"""Local MCP adapter that exposes the lab backend's first-party operations."""

import os

import httpx
from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from demo_identity import access_record


mcp = FastMCP("RAG Interview Lab")
BACKEND_URL = os.getenv("RAG_LAB_BACKEND_URL", "http://127.0.0.1:8000")


def backend_request(method: str, path: str, payload: dict | None = None) -> dict:
    token = get_http_headers().get("x-demo-token")
    if not token:
        return {"error": "access_denied", "_access": access_record(
            None, "denied", "A request-specific demo identity is required.")}
    response = httpx.request(
        method,
        f"{BACKEND_URL}{path}",
        headers={"X-Demo-Token": token},
        json=payload,
        timeout=30,
    )
    if response.status_code in {401, 403}:
        return {"error": "access_denied", "_access": response.json()["detail"]}
    response.raise_for_status()
    return response.json()


@mcp.tool()
def search_documents(question: str, limit: int = 3) -> dict:
    """Find source passages from documents the current demo user may read."""
    return backend_request(
        "POST", "/search", {"question": question, "limit": limit}
    )


@mcp.tool()
def get_trade_scenario() -> dict:
    """Get the proposed Portfolio Alpha trade and current position values."""
    return backend_request("GET", "/trade-scenario")


@mcp.tool()
def calculate_exposure(
    net_asset_value_usd: int,
    current_exposure_usd: int,
    proposed_purchase_usd: int,
) -> dict:
    """Calculate post-trade single-name exposure as a percentage of NAV."""
    return backend_request(
        "POST",
        "/calculate-exposure",
        {
            "net_asset_value_usd": net_asset_value_usd,
            "current_exposure_usd": current_exposure_usd,
            "proposed_purchase_usd": proposed_purchase_usd,
        },
    )


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8001)
