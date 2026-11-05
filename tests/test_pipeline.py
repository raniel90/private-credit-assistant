"""Regression checks for the two failures we observed while building the lab."""

import json
from pathlib import Path

from ingest import prepare_chunks, read_text
from risk import assess_trade


def test_policy_limit_stays_intact_in_a_chunk():
    source = Path(__file__).parents[1] / "data" / "alpha-risk.txt"
    chunks = prepare_chunks(read_text(source))
    assert any("7 percent of net asset value" in chunk for chunk in chunks)
    assert any("risk officer must review an exception" in chunk for chunk in chunks)


def test_trade_assessment_requires_human_review():
    root = Path(__file__).parents[1]
    scenario = json.loads((root / "data/trade-scenario.json").read_text())
    results = {
        "search_documents": {
            "results": [{
                "text": " ".join(prepare_chunks(read_text(root / "data/alpha-risk.txt"))),
                "source_path": "data/alpha-risk.txt",
            }]
        },
        "get_trade_scenario": scenario,
        "calculate_exposure": {
            "post_trade_exposure_usd": 8_000_000,
            "post_trade_exposure_percent": 8.0,
        },
    }
    answer = assess_trade(results)
    assert "8% ($8,000,000 / $100,000,000)" in answer
    assert "above the 7%" in answer
    assert "Do not approve automatically" in answer
    assert "risk-officer review" in answer
