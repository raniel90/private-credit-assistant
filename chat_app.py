"""Local study UI; the agent calls the existing MCP server over HTTP."""
import asyncio
import json
import secrets
from time import monotonic
from contextlib import suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

from agent_graph import run_turn
from demo_identity import PROFILES

app = FastAPI(title="Private Credit Lab")
turn_lock = asyncio.Lock()
SESSIONS = {}
SESSION_COOKIE = "credit_demo_session"
SESSION_TTL = 8 * 60 * 60


def demo_session(request: Request):
    session = SESSIONS.get(request.cookies.get(SESSION_COOKIE))
    if not session or session["expires"] < monotonic():
        raise HTTPException(401, "Demo session expired. Reload the page.")
    return session


def session_view(session):
    return {"profile": PROFILES[session["profile_id"]].public(),
            "history_scope": session["scope"], "mode": "demo"}


@app.get("/demo/profiles")
def profiles():
    return {"profiles": [profile.public() for profile in PROFILES.values()]}


@app.get("/demo/session")
def current_session(request: Request, response: Response):
    try:
        session = demo_session(request)
    except HTTPException:
        for key in list(SESSIONS):
            if SESSIONS[key]["expires"] < monotonic():
                del SESSIONS[key]
        token = secrets.token_urlsafe(32)
        session = {"scope": secrets.token_hex(16), "profile_id": "alice",
                   "expires": monotonic() + SESSION_TTL}
        SESSIONS[token] = session
        response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="strict", max_age=SESSION_TTL)
    response.headers["Cache-Control"] = "no-store"
    return session_view(session)


class ProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_id: str


@app.post("/demo/session")
def switch_profile(body: ProfileRequest, request: Request, response: Response):
    session = demo_session(request)
    if body.profile_id not in PROFILES:
        raise HTTPException(422, "Unknown demo profile.")
    session["profile_id"] = body.profile_id
    response.headers["Cache-Control"] = "no-store"
    return session_view(session)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(min_length=1, max_length=100)
    profile_id: str


def turn_identity(body: ChatRequest, request: Request):
    session = demo_session(request)
    if body.profile_id != session["profile_id"]:
        raise HTTPException(409, "The active profile changed. Reload the page before sending.")
    profile = PROFILES[session["profile_id"]]
    # Include the browser session and server-resolved user, never a client-supplied tenant.
    thread = json.dumps([session["scope"], profile.principal.user_id, body.thread_id])
    return profile, thread


@app.post("/chat/stream")
async def stream_chat(request: ChatRequest, http_request: Request):
    profile, thread = turn_identity(request, http_request)
    if not request.question.strip():
        raise HTTPException(422, "Enter a question.")
    async def events():
        queue = asyncio.Queue(maxsize=128)
        async def produce():
            try:
                async with turn_lock:
                    state = await run_turn(request.question, thread, emit=queue.put, demo_token=profile.token)
                await queue.put({"type": "done", "answer": state["answer"], "trace": state["trace"]})
            except Exception:
                await queue.put({"type": "error", "text": "Unable to finish. Check the local services and try again."})
        task = asyncio.create_task(produce())
        try:
            yield "data: " + json.dumps({"type": "identity", "profile": profile.public()}) + "\n\n"
            yield 'data: {"type":"status","step":"queued"}\n\n'
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield "data: " + json.dumps(event) + "\n\n"
                if event["type"] in {"done", "error"}:
                    break
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/", response_class=HTMLResponse)
def index():
    return Path(__file__).with_name("chat_ui.html").read_text()


@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    profile, thread = turn_identity(request, http_request)
    if not request.question.strip():
        raise HTTPException(422, "Enter a question.")
    try:
        async with turn_lock:
            state = await run_turn(request.question, thread, demo_token=profile.token)
        return {"answer": state["answer"], "trace": state["trace"]}
    except Exception as exc:
        raise HTTPException(503, "Unable to complete the request. Check Ollama, MCP, API, and Qdrant in the terminal.") from exc
