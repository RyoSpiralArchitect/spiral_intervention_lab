"""Identity and accounting for blueprint reviews, not readout measurements."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .response_probe import identity


NAME = "entity_insertion_operator_candidate_review"
MISSING_TERM_KEYS = (
    "entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms",
)


def recorded_step(result: Mapping[str, Any] | None) -> int:
    value = result.get("recorded_step") if isinstance(result, Mapping) else None
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else -1


def missing_terms_known(feedback: Mapping[str, Any]) -> bool:
    return any(isinstance(feedback.get(key), (list, tuple)) for key in MISSING_TERM_KEYS)


def blueprint_ids(result: Mapping[str, Any]) -> list[str]:
    """Structural identity, not a measurement or a grant of apply authority."""
    blueprints = result.get("candidate_blueprints")
    if not isinstance(blueprints, (list, tuple)):
        return []
    keys = ("kind", "objective_term", "first_piece", "first_token_id", "source_provenance",
            "source_span", "recommended_operator_families")
    return sorted({identity("entity-blueprint:", {k: row.get(k) for k in keys})
                   for row in blueprints if isinstance(row, Mapping)})


def annotate_review(
    result: dict[str, Any], *, probe: Mapping[str, Any] | None,
    request: Mapping[str, Any], missing_terms: Sequence[str], missing_known: bool,
    history: Sequence[Mapping[str, Any]],
    known_blueprint_ids: Sequence[str] | set[str] = (),
) -> dict[str, Any]:
    rows = probe.get("term_readout_rows", ()) if isinstance(probe, Mapping) else ()
    rows = [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, (list, tuple)) else []
    terms = {str(row.get("term") or "").strip().casefold() for row in rows}
    # Prefix advancement is not a fresh probe. Only the evidence and feedback
    # actually consumed by this pure review can reopen it.
    probe_id = identity("entity-probe:", {
        "recorded_step": recorded_step(probe), "rows": rows,
        "status": probe.get("status") if isinstance(probe, Mapping) else None,
    })
    review_id = identity("entity-review:", {
        "probe_id": probe_id, "missing_known": missing_known,
        "missing_probe_terms": sorted(terms.intersection(str(term).casefold() for term in missing_terms)),
        "objective": request.get("objective_bundle_key") or request.get("bundle_key"),
    })
    result.update(
        entity_candidate_review_input_id=review_id,
        source_probe_id=probe_id,
        source_probe_recorded_step=recorded_step(probe),
        source_probe_is_current_prefix=recorded_step(probe) == recorded_step(result) >= 0,
        evidence_scope="recorded_probe_and_current_task_feedback_not_new_measurement",
        new_measurement_count=0,
        physical_replay_count=0,
        diagnostic_only=True,
        policy_candidate_ready=False,
    )
    account_review(result, history, known_blueprint_ids=known_blueprint_ids)
    return result


def account_review(result: dict[str, Any], history: Sequence[Mapping[str, Any]], *,
                   known_blueprint_ids: Sequence[str] | set[str] = ()) -> None:
    previous = next((row for row in reversed(history) if isinstance(row, Mapping)
                     and row.get("diagnostic") == NAME), None)
    reused = bool(previous and previous.get("entity_candidate_review_input_id")
                  == result["entity_candidate_review_input_id"])
    ids = blueprint_ids(result)
    seen = set(known_blueprint_ids)
    for row in history:
        if isinstance(row, Mapping) and row.get("diagnostic") == NAME:
            seen.update(blueprint_ids(row))
    new_ids = sorted(set(ids) - seen)
    # Feedback can change the review input without discovering a new candidate.
    # A fresh probe or a changed candidate set can still refresh the review for
    # free; it does not reuse or suppress a current-prefix physical measurement.
    outputs_unchanged = bool(previous and result.get("source_probe_id")
        and previous.get("source_probe_id") == result["source_probe_id"]
        and blueprint_ids(previous) == ids)
    result.update(
        review_reused=reused,
        review_outputs_unchanged=outputs_unchanged,
        candidate_blueprint_ids=ids,
        new_candidate_blueprint_ids=new_ids,
        new_candidate_blueprint_count=len(new_ids),
        reused_candidate_blueprint_count=len(ids) - len(new_ids),
        candidate_novelty_scope="episode_blueprint_structure_not_measurement",
        diagnostic_call_cost=int(bool(new_ids)),
        diagnostic_budget_charge_reason=("unchanged_review_inputs" if reused else
            "new_candidate_blueprints" if new_ids else
            "no_new_candidate_blueprints" if ids else str(result["status"])),
    )
