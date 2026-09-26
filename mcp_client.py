"""
Week 9: a small sync-friendly-ish wrapper around one MCP client session,
used by agent.py so the agent loop's own diff for "call tools via MCP"
stays minimal -- the transport/session plumbing lives here, not inside the
ReAct loop.

One MCPToolSession per run_agent() call: opened once at the start of a run,
reused for every step in that run, closed when the run ends.
"""
import json
import os
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from config import MCP_SERVER_URL, MCP_SHARED_SECRET, MCP_TRANSPORT

_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")


class MCPToolError(Exception):
    """A tool call failed in a recoverable way. agent.py catches this and
    turns it into an observation -- the same non-crashing treatment already
    given to an unrecognized action or malformed JSON."""


class MCPToolSession:
    """
    Async context manager wrapping one MCP client session.

        async with MCPToolSession() as mcp:
            tools = await mcp.discover_tools()
            chunks = await mcp.call_tool("search_tickets", {"query": "..."})

    Pass `existing_session=` an already-initialized mcp.ClientSession to use
    it directly instead of opening a new transport connection -- this is the
    seam evals/prompt_injection_test.py uses to run the real, unmodified
    agent loop against an isolated in-memory MCP server (a poisoned test
    corpus) with zero risk to the real data/ directory or Qdrant collection.
    """

    def __init__(self, transport=MCP_TRANSPORT, url=MCP_SERVER_URL,
                 shared_secret=MCP_SHARED_SECRET, existing_session=None):
        self._transport = transport
        self._url = url
        self._shared_secret = shared_secret
        self._existing_session = existing_session
        self._stack = None
        self.session = None

    async def __aenter__(self):
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        if self._existing_session is not None:
            # Caller already built + initialized this session (the
            # in-memory test seam) -- don't reconnect or re-initialize.
            self.session = self._existing_session
            return self

        if self._transport == "stdio":
            params = StdioServerParameters(command=sys.executable, args=[_SERVER_PATH, "--transport", "stdio"])
            read, write = await self._stack.enter_async_context(stdio_client(params))
        else:
            # streamable_http_client only closes an httpx client it created
            # itself -- passing one in (needed here, to attach the shared-
            # secret header) means WE own its lifecycle. Found live: not
            # entering it into the stack leaked one unclosed httpx2.AsyncClient
            # per agent run, which surfaced as every run past the ~12th in a
            # 20-case eval batch failing with an opaque anyio TaskGroup
            # ExceptionGroup -- a resource exhaustion threshold, not a flake.
            http_client = await self._stack.enter_async_context(
                create_mcp_http_client(
                    headers={"X-MCP-Shared-Secret": self._shared_secret} if self._shared_secret else {}
                )
            )
            read, write = await self._stack.enter_async_context(
                streamable_http_client(self._url, http_client=http_client)
            )

        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc_info):
        return await self._stack.__aexit__(*exc_info)

    async def discover_tools(self):
        """Returns [{"name", "description", "input_schema"}, ...] -- the
        source of truth agent.py's prompt builds its action list from,
        instead of a hard-coded string."""
        result = await self.session.list_tools()
        return [
            {"name": t.name, "description": t.description or "", "input_schema": t.input_schema or {}}
            for t in result.tools
        ]

    async def call_tool(self, name, arguments):
        """
        Calls an MCP tool and returns its result as a list of dicts -- the
        same shape retrieval.retrieve()/ingestion.grouped_sources() already
        produce, so agent.py's downstream logic (dedup, _validate_sources)
        needs zero changes regardless of which tool was called.
        """
        result = await self.session.call_tool(name, arguments)
        if result.is_error:
            text = "; ".join(getattr(block, "text", str(block)) for block in result.content)
            raise MCPToolError(f"tool '{name}' failed: {text}")

        if result.structured_content and "result" in result.structured_content:
            return result.structured_content["result"]

        # Fallback for a tool result with no structured content (older
        # client/server negotiation, or a tool with no output schema) --
        # the text block is JSON-encoded (see mcp_server.py's tools, which
        # return plain Python lists/dicts and let the SDK serialize them).
        for block in result.content:
            if getattr(block, "type", None) == "text":
                return json.loads(block.text)
        return []
