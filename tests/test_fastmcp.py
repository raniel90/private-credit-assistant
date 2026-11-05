"""The FastMCP server exposes the same first-party contract to MCP clients."""

import asyncio

from fastmcp import Client

import mcp_server


def test_fastmcp_discovers_and_calls_tools(monkeypatch):
    def fake_backend_request(method, path, payload=None):
        return {"method": method, "path": path, "payload": payload}

    monkeypatch.setattr(mcp_server, "backend_request", fake_backend_request)

    async def check():
        async with Client(mcp_server.mcp) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools} == {
                "search_documents", "get_trade_scenario", "calculate_exposure"
            }
            result = await client.call_tool(
                "search_documents", {"question": "What is the Alpha limit?", "limit": 3}
            )
            assert result.data == {
                "method": "POST",
                "path": "/search",
                "payload": {"question": "What is the Alpha limit?", "limit": 3},
            }

    asyncio.run(check())
