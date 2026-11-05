# Private Credit Assistant

A local chat application demonstrating RAG, MCP tool calling and tenant/group
access controls with fictional private credit data. Responses stream to the UI,
which displays tool durations, returned sources and backend access decisions.
This project has no association with any real institution.

## Architecture

```text
Browser: chat + demo profiles
    -> Chat FastAPI :8002 — sessions, SSE, conversation isolation
    -> LangGraph + Ollama :11434 — classification and tool selection
    -> FastMCP :8001/mcp — tools with per-request identity
    -> Business FastAPI :8000
         ├─ retrieval -> Qdrant :6333 (tenant AND group filters)
         ├─ loan scenario -> synthetic JSON fixture
         └─ exposure calculation -> deterministic Python

Documents + access manifest
    -> extraction -> chunks -> FastEmbed embeddings -> Qdrant
```

The model participates in tool selection; the backend enforces authorization.
FastMCP publishes and executes tools; LangGraph controls state and transitions.
Calculations and the human-review requirement are implemented in code.

## Run locally

Requirements: Python 3.11+, uv, Docker with Compose, and Ollama. Allow disk space
for dependencies, embedding weights and the selected model.

From the project directory:

```bash
uv sync --locked
docker compose up -d
```

Start Ollama in another terminal if it is not already running:

```bash
ollama serve
```

Download the default model and index the sample documents:

```bash
ollama pull qwen3:1.7b
uv run python ingest.py
```

The first ingestion also downloads the FastEmbed embedding model. Original
documents remain in `data/`; Qdrant stores a rebuildable derived index.

Run each application service in its own terminal:

```bash
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

```bash
uv run python mcp_server.py
```

```bash
uv run uvicorn chat_app:app --host 127.0.0.1 --port 8002
```

Open [the chat](http://127.0.0.1:8002). Suggested questions fill the composer;
press Send to run the agent. Services are intended for localhost only.

Optional overrides are documented in `.env.example`. Copy it to `.env` and
load it explicitly, for example to run the CLI:

```bash
uv run --env-file .env python agent_graph.py
```

The CLI requires `RAG_LAB_DEMO_TOKEN`. The web application instead resolves the
selected fictional profile on the server and passes its credential on each MCP
request. Environment variable names retain the original prefix for compatibility.

## Demonstrate access controls

| Profile | Tenant / group | Document access | Loan assessment |
| --- | --- | --- | --- |
| Alice | fund / risk | Alpha risk policy | Allowed |
| Bob | fund / credit | Acme memo and Beta policy | Denied |
| Carol | other_fund / risk | No documents in the sample dataset | Denied |

1. Select Alice and **Analyze the loan**. Exposure rises to $8M against $100M
   NAV: 8%, above the fictional 7% limit. Human exception review is required.
2. Select Bob and submit the same question. The backend denies loan assessment.
3. Start a new conversation as Bob and choose **Review credit memo**.
4. Repeat the memo question as Carol: no accessible evidence is returned.
5. Expand each tool's access details to inspect identity, tenant, groups,
   returned sources, request ID and execution duration.

Profiles simulate identities; they do not provide production login. Chat
requests cannot specify arbitrary groups or tenants. Memory is scoped to browser
session + user + conversation, and browser history is partitioned by session and
profile. Sessions expire after eight hours; server restarts clear sessions and
in-memory checkpoints.

## Files

| File | Responsibility |
| --- | --- |
| `chat_ui.html` | English-language chat, profiles, history and tool feedback |
| `chat_app.py` | Demo sessions, request validation and SSE |
| `demo_identity.py` | Fictional identities and access evidence metadata |
| `agent_graph.py` | Agent graph, MCP client and Ollama streaming |
| `mcp_server.py` | Three tools and per-request identity forwarding |
| `api.py` | Backend authorization and business operations |
| `retrieval.py` | Tenant/group-filtered vector search |
| `ingest.py` | TXT/Markdown/PDF extraction and idempotent upserts |
| `core.py` | Identity types and embeddings |
| `risk.py` | Deterministic concentration assessment |
| `data/` | Synthetic documents, ACL manifest and loan fixture |
| `tests/` | Tool contracts, authorization, isolation and financial rules |

## Tests

```bash
uv run --locked pytest -q
```

Tests do not require Ollama, a running Qdrant server or downloaded embedding
weights. Authorization tests use Qdrant's in-memory engine and controlled
embeddings to exercise the real tenant AND group filters. Other checks cover
per-request MCP identity, missing identity, denied operations, session/user
separation and rejection of forged profile or permission fields.

## Scope and limitations

- All organizations, users, amounts and policies are fictional.
- The calculation checks borrower concentration only, not collateral value,
  repayment capacity or full lending eligibility.
- Model explanations can be inaccurate. Inspect the separate verified
  calculation and retrieved evidence; no loan is approved automatically.
- Fixed demo credentials are intentionally public fixtures, not secrets or real
  authentication. Do not expose this application to the public internet as-is.
- Browser history is not a secure data store. Access details are per-run
  evidence, not a durable audit log. Production requires verified identity,
  protected persistence, audit retention and operational hardening.
- Group permissions are illustrative. Beta and the Acme memo share the
  `credit` group; this is not complete fund-by-fund entitlement management.

## Stop and reset

Stop the Python services with Ctrl+C, then stop Qdrant:

```bash
docker compose down
```

While Ollama is running, unload or remove the demo model as needed:

```bash
ollama stop qwen3:1.7b
ollama rm qwen3:1.7b
```

Stop `ollama serve` with Ctrl+C if started manually. To reset retrieval,
remove only this project's `qdrant_storage/` after stopping Qdrant and rerun
ingestion. The `.gitignore` excludes indexes, virtual environments, model files,
credentials, logs and temporary development artifacts.
