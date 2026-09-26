"""
Week 9: the app's own MCP server -- exposes ticket search and source listing
as standard MCP tools, callable by any MCP client, not just this repo's
agent.py.

Two transports, different purposes:
    python mcp_server.py                     # streamable-http (default) --
                                              # a long-lived process, same
                                              # operational shape as Qdrant,
                                              # so the embedding model/BM25
                                              # index stay warm across calls
    python mcp_server.py --transport stdio   # local subprocess -- no
                                              # network exposure; used to
                                              # inspect the raw JSON-RPC
                                              # handshake once and as the
                                              # simplest possible
                                              # zero-infrastructure demo

Built on the official `mcp` SDK. Note: this installed version (mcp 2.x)
renamed what the SDK's 1.x line called "FastMCP" to `MCPServer`
(mcp.server.mcpserver.MCPServer) -- confirmed directly against the
installed package (the old import raises a ModuleNotFoundError with a
migration pointer), not assumed. Same decorator-based ergonomics the
brief's "Building a server (fastmcp)" topic is describing, just renamed.
"""

import argparse

import uvicorn
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp.server.mcpserver import MCPServer

from config import (
    DEFAULT_TOP_K,
    MCP_SERVER_HOST,
    MCP_SERVER_PORT,
    MCP_SHARED_SECRET,
    MCP_TRANSPORT,
)
from ingestion import get_client, get_embedding_model, grouped_sources
from retrieval import retrieve

# Load-once, module-level cache -- same pattern as api.py's _get_model()/
# get_client(). The whole point of the http transport is that this process
# stays warm across calls instead of reloading the embedding model per call.
_model = None
_client = None


def _get_model():
    global _model
    if _model is None:
        _model = get_embedding_model()
    return _model


def _get_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


mcp_app = MCPServer(name="ask-my-tickets")


@mcp_app.tool()
def search_tickets(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Search the customer-support ticket knowledge base for chunks relevant
    to ONE specific topic. Use a focused, single-topic query, not a whole
    compound question at once -- call this once per distinct topic before
    giving a final answer. Returns the closest matches, each with its source
    document, page (if any), and retrieval distance (lower = more relevant)."""
    return retrieve(query, top_k=top_k, model=_get_model(), client=_get_client())


@mcp_app.tool()
def list_ticket_sources() -> list[dict]:
    """List every distinct source document in the ticket knowledge base,
    with how many chunks it contributed and a short text preview."""
    return grouped_sources(_get_client())


class SharedSecretMiddleware(BaseHTTPMiddleware):
    """
    Gates every HTTP request behind a shared-secret header -- the brief's
    own "keeping it safe: access control" topic, made concrete rather than
    just written about. Only wraps the HTTP transport: stdio has no network
    exposure to begin with (a caller must already be able to launch a local
    subprocess to reach it at all), so there's no remote party to gate out
    there.
    """

    async def dispatch(self, request: Request, call_next):
        if request.headers.get("X-MCP-Shared-Secret") != MCP_SHARED_SECRET:
            return JSONResponse(
                {"error": "missing or invalid X-MCP-Shared-Secret header"},
                status_code=401,
            )
        return await call_next(request)


def main():
    parser = argparse.ArgumentParser(description="Ask My Tickets -- MCP server (Week 9)")
    parser.add_argument("--transport", choices=["http", "stdio"], default=MCP_TRANSPORT)
    args = parser.parse_args()

    if args.transport == "stdio":
        print("Serving over stdio (local subprocess only -- no auth, no network exposure to gate).")
        mcp_app.run(transport="stdio")
        return

    if not MCP_SHARED_SECRET:
        raise SystemExit(
            "MCP_SHARED_SECRET is not set in .env -- the HTTP transport refuses to start "
            "without one, since it's the only thing standing between this server and an "
            "arbitrary caller on the network. Set it and try again."
        )

    http_app = mcp_app.streamable_http_app(host=MCP_SERVER_HOST)
    http_app.add_middleware(SharedSecretMiddleware)
    print(f"Serving over streamable-http at http://{MCP_SERVER_HOST}:{MCP_SERVER_PORT}/mcp")
    uvicorn.run(http_app, host=MCP_SERVER_HOST, port=MCP_SERVER_PORT)


if __name__ == "__main__":
    main()
