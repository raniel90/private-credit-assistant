"""Deterministic assessment of the fictional portfolio trade."""

import re


def assess_trade(results: dict) -> str:
    hits = results["search_documents"].get("results", [])
    if not hits:
        return "No accessible policy evidence was found; approval cannot be assessed."

    policy_text = " ".join(hit["text"] for hit in hits)
    match = re.search(
        r"single-name exposure to\s+(\d+(?:\.\d+)?)\s+percent",
        policy_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return "The retrieved policy did not state a verifiable limit."

    limit = float(match.group(1))
    scenario = results["get_trade_scenario"]
    calculation = results["calculate_exposure"]
    actual = calculation["post_trade_exposure_percent"]
    expected = round(
        100
        * (scenario["current_exposure_usd"] + scenario["proposed_purchase_usd"])
        / scenario["net_asset_value_usd"],
        4,
    )
    if actual != expected:
        raise ValueError("Backend calculation does not match the scenario")

    source = next(hit["source_path"] for hit in hits if match.group(1) in hit["text"])
    math = (
        f"{actual:g}% (${calculation['post_trade_exposure_usd']:,} / "
        f"${scenario['net_asset_value_usd']:,})"
    )
    if actual <= limit:
        return (
            f"Post-trade exposure is {math}, within the {limit:g}% single-name "
            f"limit. This checks only that policy rule. Source: {source}"
        )

    review_required = all(
        phrase in policy_text.lower()
        for phrase in ("risk officer", "exception", "before a trade is approved")
    )
    review = (
        "The policy requires risk-officer review of an exception before approval."
        if review_required
        else "The retrieved passages do not establish an exception process."
    )
    return (
        f"Post-trade exposure is {math}, above the {limit:g}% single-name "
        f"limit. Do not approve automatically. {review} Source: {source}"
    )
