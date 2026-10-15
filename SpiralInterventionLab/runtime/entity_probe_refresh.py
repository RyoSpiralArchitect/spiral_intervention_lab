"""Bounded discovery offers for missing terms absent from prior entity probes.

Offers do not run a probe, choose an intervention, or grant apply authority.
Episode coverage is independent of the retained diagnostic-result window.
"""
from collections.abc import Mapping, Sequence
from typing import Any

from .response_probe import identity


NAME = "target_entity_insertion_probe"
MAX_TERMS = 4
UNMEASURED_STATUSES = frozenset({"stale_refresh_request", "error"})


def is_recorded_probe(row: Any) -> bool:
    """A probe receipt that measured readout rows; rejected, failed or empty probes are not evidence."""
    if not isinstance(row, Mapping) or row.get("diagnostic") != NAME or row.get("status") in UNMEASURED_STATUSES:
        return False
    rows = row.get("term_readout_rows")
    return isinstance(rows, (list, tuple)) and any(isinstance(item, Mapping) for item in rows)


def terms(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    unique: dict[str, str] = {}
    for item in value:
        if isinstance(item, str):
            term = " ".join(item.split())
            if term:
                unique.setdefault(term.casefold(), term)
    return list(unique.values())


def coverage(history: Sequence[Mapping[str, Any]], attempted: set[str]) -> set[str]:
    covered = set(attempted)
    for receipt in history:
        if receipt.get("diagnostic") != NAME:
            continue
        rows = receipt.get("term_readout_rows") or ()
        covered.update(term.casefold() for term in terms(
            [row.get("term") for row in rows if isinstance(row, Mapping)]))
        covered.update(term.casefold() for term in terms(receipt.get("requested_terms")))
    return covered


def report(*, missing_terms: Sequence[str], missing_known: bool,
           history: Sequence[Mapping[str, Any]], attempted: set[str],
           pending_blueprint_count: int, budget_left: int) -> dict[str, Any]:
    missing = terms(missing_terms)
    covered = coverage(history, attempted)
    unseen = [term for term in missing if term.casefold() not in covered]
    probe_seen = bool(covered) or any(row.get("diagnostic") == NAME for row in history)
    if not missing_known:
        status = "missing_terms_unknown"
    elif not missing:
        status = "no_missing_terms"
    elif not probe_seen:
        status = "awaiting_initial_probe"
    elif pending_blueprint_count:
        status = "recorded_candidates_pending"
    elif not unseen:
        status = "all_missing_terms_already_probed"
    elif budget_left <= 0:
        status = "diagnostic_budget_exhausted"
    else:
        status = "refresh_available"
    selected = unseen[:MAX_TERMS]
    refresh_id = identity("entity-probe-refresh:", {
        "missing": sorted(term.casefold() for term in missing),
        "attempted": sorted(covered), "terms": selected,
    })
    request = {
        "diagnostic": NAME, "terms": selected, "probe_refresh_id": refresh_id,
        "next_evidence_needed": "unprobed_missing_term_readout",
    } if status in {"refresh_available", "diagnostic_budget_exhausted"} else None
    return {"status": status, "probe_refresh_id": refresh_id,
        "unprobed_missing_terms": unseen, "probe_terms": selected,
        "attempted_term_count": len(covered), "pending_blueprint_count": pending_blueprint_count,
        "diagnostic_budget_left": budget_left, "request": request,
        "scope": "unprobed_missing_terms_once_per_episode_not_prefix_retry",
        "production_apply_allowed": False, "certified_for_apply": False}
