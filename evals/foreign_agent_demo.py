"""
Week 9: stands in for "someone else's agent" -- the mentor's literal
checklist item is "did they build their own server that another person's
agent could call?" This script proves it concretely instead of just
asserting it: it deliberately imports NOTHING from this repo except the
two facts a stranger would actually be handed -- a URL and a shared
secret -- not `retrieval`, not `agent`, not `ingestion`, not even
`config`. Everything else it knows about the server it learns the same
way any other MCP client would: by asking it.

Requires mcp_server.py already running (python mcp_server.py).

Usage:
    python evals/foreign_agent_demo.py <server-url> <shared-secret>
    python evals/foreign_agent_demo.py   # uses the two env vars below
"""
import asyncio
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client


async def main():
    if len(sys.argv) >= 3:
        url, secret = sys.argv[1], sys.argv[2]
    else:
        url = os.environ.get("FOREIGN_DEMO_MCP_URL", "http://127.0.0.1:8830/mcp")
        secret = os.environ.get("FOREIGN_DEMO_MCP_SECRET", "")
        print(f"(no CLI args given -- using {url!r} and a secret from FOREIGN_DEMO_MCP_SECRET)")

    # streamable_http_client only closes an httpx client it created itself --
    # passing one in (needed here, for the shared-secret header) means we
    # must close it ourselves, or it leaks one open client per connection.
    async with create_mcp_http_client(headers={"X-MCP-Shared-Secret": secret} if secret else {}) as http_client, \
               streamable_http_client(url, http_client=http_client) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("\n--- Step 1: discover what this server can do (no prior knowledge) ---")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")
                print(f"    input schema: {tool.input_schema}")

            print("\n--- Step 2: call a discovered tool by name, with a made-up question ---")
            first_tool = tools.tools[0]
            args = {"query": "why was I charged twice"} if "query" in first_tool.input_schema.get("properties", {}) else {}
            result = await session.call_tool(first_tool.name, args)
            print(f"Called '{first_tool.name}' -- got {len((result.structured_content or {}).get('result', []))} result(s):")
            for item in (result.structured_content or {}).get("result", [])[:2]:
                print(f"  {item}")

    print("\nA client that has never read this repo's source discovered and called "
          "the server successfully -- this is the same mechanism any other MCP agent would use.")


if __name__ == "__main__":
    asyncio.run(main())
