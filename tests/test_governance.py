"""Authorization, session separation and denied-tool behavior for the demo."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

import api
import chat_app
import agent_graph
import mcp_server
import retrieval
from core import COLLECTION
from demo_identity import PROFILES


@pytest.fixture
def backend(monkeypatch):
    # Exercise the real tenant AND group filter with Qdrant's in-memory engine.
    client = QdrantClient(":memory:")
    client.create_collection(COLLECTION, vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    documents = json.loads((Path(__file__).parents[1] / "data/documents.json").read_text())
    client.upsert(COLLECTION, points=[models.PointStruct(
        id=i, vector=[1.0, 0.0], payload={**doc, "text": "test evidence", "chunk_index": 0}
    ) for i, doc in enumerate(documents)])
    monkeypatch.setattr(retrieval, "get_client", lambda: client)
    monkeypatch.setattr(retrieval, "get_embedder", lambda: SimpleNamespace(embed=lambda texts: [[1.0, 0.0]]))
    with TestClient(api.app) as test_client:
        yield test_client
    client.close()


@pytest.mark.parametrize("user,documents,status", [
    ("alice", {"alpha-risk"}, 200),
    ("bob", {"alpha-research", "beta-risk"}, 403),
    ("carol", set(), 403),
])
def test_same_query_different_authorization(backend, user, documents, status):
    headers = {"X-Demo-Token": PROFILES[user].token}
    response = backend.post("/search", json={"question": "policy and credit memorandum"}, headers=headers)
    data = response.json()
    assert {hit["document_id"] for hit in data["results"]} == documents
    assert {hit["document_id"] for hit in data["_access"]["documents"]} == documents
    assert data["_access"]["user_id"] == user
    assert backend.get("/trade-scenario", headers=headers).status_code == status
    calculation = backend.post("/calculate-exposure", headers=headers, json={
        "net_asset_value_usd": 100, "current_exposure_usd": 7, "proposed_purchase_usd": 1})
    assert calculation.status_code == status


def test_missing_identity_is_denied(backend, monkeypatch):
    assert backend.post("/search", json={"question": "policy"}).status_code == 401
    monkeypatch.setenv("RAG_LAB_DEMO_TOKEN", PROFILES["alice"].token)
    monkeypatch.setattr(mcp_server, "get_http_headers", lambda: {})
    # A global Alice token must never elevate an unauthenticated MCP request.
    assert mcp_server.backend_request("GET", "/trade-scenario")["error"] == "access_denied"


def test_mcp_forwards_each_request_identity(backend, monkeypatch):
    monkeypatch.setattr(mcp_server.httpx, "request", lambda method, url, **kwargs:
                        backend.request(method, url, headers=kwargs["headers"], json=kwargs["json"]))
    for user in ["alice", "bob", "carol", "alice"]:
        monkeypatch.setattr(mcp_server, "get_http_headers", lambda user=user: {"x-demo-token": PROFILES[user].token})
        result = mcp_server.backend_request("GET", "/trade-scenario")
        assert result["_access"]["user_id"] == user
        assert (result.get("error") == "access_denied") == (user != "alice")


def test_chat_sessions_scope_memory_and_reject_forged_identity(monkeypatch):
    calls = []

    async def fake_turn(question, thread_id, emit=None, demo_token=None):
        calls.append((thread_id, demo_token))
        return {"answer": "test", "trace": []}

    monkeypatch.setattr(chat_app, "run_turn", fake_turn)
    body = {"question": "policy", "thread_id": "same-thread", "profile_id": "alice"}
    with TestClient(chat_app.app) as client, TestClient(chat_app.app) as other:
        assert client.post("/chat", json=body).status_code == 401
        session = client.get("/demo/session").json()
        assert "token" not in session["profile"]
        assert client.post("/chat", json=body).status_code == 200
        alice_thread = calls[-1][0]
        assert client.post("/demo/session", json={"profile_id": "unknown"}).status_code == 422
        assert client.post("/demo/session", json={"profile_id": "bob", "groups": ["risk"]}).status_code == 422
        client.post("/demo/session", json={"profile_id": "bob"})
        assert client.post("/chat", json=body).status_code == 409
        assert client.post("/chat", json={**body, "profile_id": "bob", "tenant_id": "fund"}).status_code == 422
        response = client.post("/chat/stream", json={**body, "profile_id": "bob"})
        assert response.status_code == 200
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[0]["profile"]["id"] == "bob"
        assert calls[-1][1] == PROFILES["bob"].token
        assert calls[-1][0] != alice_thread
        client.post("/demo/session", json={"profile_id": "alice"})
        client.post("/chat", json=body)
        assert calls[-1][0] == alice_thread
        other.get("/demo/session")
        other.post("/chat", json=body)
        assert calls[-1][0] != alice_thread


def test_denied_tool_stops_assessment(monkeypatch):
    from langchain_core.messages import HumanMessage

    calls = []

    class DeniedClient:
        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(name=n, description=n) for n in agent_graph.ALLOWED_TOOLS])

        async def call_tool(self, name, args):
            calls.append(name)
            return SimpleNamespace(is_error=False, content=[SimpleNamespace(type="text", text=json.dumps({
                "error": "access_denied", "_access": {"decision": "denied", "reason": "Risk access required."}
            }))])

    async def model(messages, format_schema=None, emit=None):
        return json.dumps({"intent": "trade_assessment"} if "intent" in format_schema["properties"]
                          else {"action": "get_trade_scenario"})

    monkeypatch.setattr(agent_graph, "ask_model", model)

    async def check():
        events = []

        async def emit(event):
            events.append(event)

        graph = agent_graph.build_graph(DeniedClient(), emit=emit)
        result = await graph.ainvoke({"messages": [HumanMessage(content="Assess this loan")]},
                                    {"configurable": {"thread_id": "denied-test"}})
        assert calls == ["get_trade_scenario"]
        assert result["answer"].startswith("Access denied.")
        assert any(e.get("status") == "denied" for e in events)

    asyncio.run(check())
