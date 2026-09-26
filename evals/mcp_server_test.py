"""
Week 9: automated smoke test for mcp_server.py -- confirms the MCP server
is genuinely discoverable and callable, not just "it ran once when I typed
a question into agent.py by hand."

Requires mcp_server.py already running (python mcp_server.py, http
transport, the default) and Qdrant up, same as any other eval script here.

Checks:
  1. Both tools (search_tickets, list_ticket_sources) are discoverable via
     tools/list, with the expected input schema.
  2. Both tools are callable and return the expected shape.
  3. A request with a missing/wrong shared-secret header is rejected.

Usage:
    python evals/mcp_server_test.py
"""
import asyncio
import os
import sys
from contextlib import asynccontextmanager

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from config import MCP_SERVER_URL, MCP_SHARED_SECRET

FAILURES = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)


@asynccontextmanager
async def _connect(secret):
    # streamable_http_client only closes an httpx client it created itself --
    # passing one in (needed here, for the shared-secret header) means we
    # must close it ourselves, or it leaks one open client per connection.
    async with create_mcp_http_client(headers={"X-MCP-Shared-Secret": secret} if secret else {}) as http_client:
        async with streamable_http_client(MCP_SERVER_URL, http_client=http_client) as streams:
            yield streams


async def check_auth_rejected():
    for label, secret in [("missing secret", None), ("wrong secret", "definitely-not-the-real-secret")]:
        rejected = False
        try:
            async with _connect(secret) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
        except Exception:
            rejected = True
        check(f"unauthenticated request rejected ({label})", rejected)


async def check_tools():
    async with _connect(MCP_SHARED_SECRET) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            check("search_tickets is discoverable", "search_tickets" in names)
            check("list_ticket_sources is discoverable", "list_ticket_sources" in names)

            search_tool = next((t for t in tools.tools if t.name == "search_tickets"), None)
            check(
                "search_tickets schema declares 'query'",
                search_tool is not None and "query" in search_tool.input_schema.get("properties", {}),
            )

            result = await session.call_tool("search_tickets", {"query": "password lockout", "top_k": 2})
            items = (result.structured_content or {}).get("result", [])
            check("search_tickets returns chunk-shaped results", bool(items) and all(
                {"text", "source", "chunk_index"} <= item.keys() for item in items
            ))

            result = await session.call_tool("list_ticket_sources", {})
            items = (result.structured_content or {}).get("result", [])
            check("list_ticket_sources returns source-shaped results", bool(items) and all(
                {"filename", "chunk_count", "preview"} <= item.keys() for item in items
            ))


async def main():
    print(f"Testing MCP server at {MCP_SERVER_URL}\n")
    await check_auth_rejected()
    await check_tools()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
