"""Small HTTP adapter for the shared retrieval function.

Demo tokens stand in for verified identity; they are not production auth.
"""

import json
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from core import Principal
from demo_identity import DEMO_USERS, access_record
from retrieval import search_documents


app = FastAPI(title="Research RAG Lab")


class SearchRequest(BaseModel):
    question: str
    limit: int = Field(default=3, ge=1, le=20)


class ExposureRequest(BaseModel):
    net_asset_value_usd: int = Field(gt=0)
    current_exposure_usd: int = Field(ge=0)
    proposed_purchase_usd: int = Field(gt=0)


def risk_user(token: str | None) -> Principal:
    principal = DEMO_USERS.get(token)
    if principal is None:
        raise HTTPException(status_code=401, detail=access_record(None, "denied", "A valid demo identity is required."))
    if principal.tenant_id != "fund" or "risk" not in principal.groups:
        raise HTTPException(status_code=403, detail=access_record(principal, "denied", "This operation requires the fund tenant and risk group."))
    return principal


@app.post("/search")
def search(request: SearchRequest, x_demo_token: str | None = Header(default=None)):
    principal = DEMO_USERS.get(x_demo_token)
    if principal is None:
        raise HTTPException(status_code=401, detail=access_record(None, "denied", "A valid demo identity is required."))
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty")

    results = search_documents(request.question, principal, request.limit)
    return {
        "results": results,
        "_access": access_record(principal, "allowed", "Tenant and group filters applied before retrieval.",
                                 documents=[{"document_id": hit["document_id"], "title": hit["title"],
                                             "source_path": hit["source_path"]} for hit in results]),
    }


@app.get("/trade-scenario")
def get_trade_scenario(x_demo_token: str | None = Header(default=None)):
    principal = risk_user(x_demo_token)
    path = Path(__file__).parent / "data" / "trade-scenario.json"
    return {**json.loads(path.read_text(encoding="utf-8")),
            "_access": access_record(principal, "allowed", "Risk access verified for the fund tenant.")}


@app.post("/calculate-exposure")
def calculate_exposure(
    request: ExposureRequest, x_demo_token: str | None = Header(default=None)
):
    principal = risk_user(x_demo_token)
    post_trade_usd = request.current_exposure_usd + request.proposed_purchase_usd
    return {
        "_access": access_record(principal, "allowed", "Risk access verified for the fund tenant."),
        "post_trade_exposure_usd": post_trade_usd,
        "post_trade_exposure_percent": round(
            100 * post_trade_usd / request.net_asset_value_usd, 4
        ),
    }
