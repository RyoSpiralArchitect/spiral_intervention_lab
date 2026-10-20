"""Source-bound next-token observations, not a clause planner or actuator.

The lexical filter is fixed before measurement. Source order, including the
natural connective and terminal punctuation, is retained; no rescue outcomes,
reference role frames, or preferred output tokens enter the offer.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from . import diagnostic_budget
from .response_probe import identity, state_identity


NAME = "source_clause_structure_probe"
MAX_CANDIDATES = 12
SOURCE_LEXEMES = frozenset({
    ",", ";", ":", ".", "and", "or", "but", "while", "whereas", "before",
    "after", "when", "then", "although", "because", "until", "so",
})


def source_candidates(worker: Any) -> list[dict[str, Any]]:
    """Use actual source token IDs, never retokenized names or header hints."""
    candidates: dict[int, dict[str, Any]] = {}
    records = worker._token_position_records()
    prompt_text = "".join(str(r.get("piece") or "") for r in records if r.get("segment_kind") == "prompt")
    offset = 0
    for record in records:
        piece = str(record.get("piece") or "")
        start = offset
        if record.get("segment_kind") == "prompt":
            offset += len(piece)
        lexeme = piece.strip().casefold()
        token_id = record.get("token_id")
        if (record.get("provenance_class") != "source_body"
                or lexeme not in SOURCE_LEXEMES
                or isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0):
            continue
        if lexeme.isalpha():
            left = start + len(piece) - len(piece.lstrip())
            right = start + len(piece.rstrip())
            if (left > 0 and (prompt_text[left - 1].isalnum() or prompt_text[left - 1] in "_'")
                    or right < len(prompt_text) and (prompt_text[right].isalnum() or prompt_text[right] in "_'")):
                continue
        if token_id not in candidates:
            if len(candidates) >= MAX_CANDIDATES:
                continue
            candidates[token_id] = {
                "token_id": token_id, "piece": piece, "source_positions": [],
                "source_provenance": "source_body",
            }
        positions = candidates[token_id]["source_positions"]
        if len(positions) < 8:
            positions.append(int(record["position"]))
    return list(candidates.values())


def context_id(worker: Any, candidates: list[dict[str, Any]]) -> str:
    # Avoid hashing the whole activation cache on every packet construction.
    # Execution separately checks full state restoration.
    return identity("clause-context:", {
        "run": worker.run_id, "episode": worker.episode_id, "prompt": worker.prompt,
        "tokens": worker._current_token_tensor().detach().cpu().tolist(),
        "model": str(getattr(worker.model, "cfg", "unknown")),
        "patches": worker._collect_active_edits(),
        "decoder_control_mode": worker.decoder_control_mode,
        "allowed_token_ids": worker.allowed_token_ids,
        "guardrails": worker._decoder_guardrail_constraints(),
        "decoder_feedback": worker._last_task_feedback if worker.decoder_control_mode != "off" else None,
        "no_progress_steps": worker._no_progress_steps if worker.decoder_control_mode != "off" else None,
        "candidates": candidates,
    })


def offer(worker: Any) -> dict[str, Any]:
    enabled = getattr(worker, "clause_structure_diagnostics", False)
    candidates = source_candidates(worker) if enabled else []
    context = context_id(worker, candidates) if candidates else None
    attempted = getattr(worker, "_clause_structure_attempted_contexts", set())
    if not enabled:
        status = "disabled"
    elif not candidates:
        status = "no_source_structure_tokens"
    elif not worker._output_token_ids():
        status = "awaiting_nonempty_prefix"
    elif worker.done():
        status = "generation_finished"
    elif worker._collect_active_edits():
        status = "active_edits_present"
    elif context in attempted:
        status = "already_attempted_at_context"
    elif diagnostic_budget.left(worker) <= 0:
        status = "diagnostic_budget_exhausted"
    else:
        status = "available"
    request = {"diagnostic": NAME, "structure_context_id": context} if context else None
    return {
        "status": status, "structure_context_id": context, "request": request,
        "source_candidates": candidates, "generated_tokens": len(worker._output_token_ids()),
        "ordering": "source_occurrence_not_expected_utility",
        "evidence_scope": "current_prefix_readout_not_clause_rescue_or_operator_effect",
        "diagnostic_only": True, "production_apply_allowed": False,
    }


def readout_rows(logits: torch.Tensor, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    values = logits.detach().cpu().float().flatten()
    finite = torch.isfinite(values)
    if not finite.any() or torch.isnan(values).any() or torch.isposinf(values).any():
        raise ValueError("invalid_structure_readout")
    if any(row["token_id"] >= values.numel() for row in candidates):
        raise ValueError("source_token_outside_readout_vocabulary")
    winner = int(values.argmax())
    top_ids = set(values.topk(min(20, int(finite.sum()))).indices.tolist())
    probabilities = values.softmax(0)
    rows = []
    for candidate in candidates:
        token_id = candidate["token_id"]
        masked = not bool(finite[token_id])
        rows.append({
            **candidate,
            "status": "masked_by_decode_constraints" if masked else "measured",
            "rank": None if masked else int((values > values[token_id]).sum()) + 1,
            "prob": float(probabilities[token_id]),
            "logit": None if masked else float(values[token_id]),
            "gap_to_top1": None if masked else float(values[winner] - values[token_id]),
            "top20_hit": token_id in top_ids,
            "is_greedy_token": token_id == winner,
        })
    return rows, winner


def execute(worker: Any, request: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    current = offer(worker)
    result: dict[str, Any] = {
        "diagnostic": NAME, "source": source, "recorded_step": worker._steps,
        "structure_context_id": current["structure_context_id"],
        "requested_structure_context_id": request.get("structure_context_id"),
        "generated_tokens": current["generated_tokens"],
        "active_edit_count": len(worker._collect_active_edits()),
        "evidence_scope": current["evidence_scope"],
        "diagnostic_only": True, "production_apply_allowed": False,
        "certified_for_apply": False, "policy_candidate_ready": False,
        "diagnostic_call_cost": 0, "new_measurement_count": 0,
        "physical_replay_count": 0, "edited_replay_count": 0,
    }
    if request.get("structure_context_id") != current["structure_context_id"] or not current["structure_context_id"]:
        return {**result, "status": "stale_structure_request", "blocked_reason": "current_context_id_required"}
    if current["status"] != "available":
        return {**result, "status": "unavailable", "blocked_reason": current["status"]}
    if not hasattr(worker, "_clause_structure_attempted_contexts"):
        worker._clause_structure_attempted_contexts = set()
    worker._clause_structure_attempted_contexts.add(current["structure_context_id"])
    before = state_identity(worker)
    before_step = worker._steps
    result.update(diagnostic_call_cost=1, physical_replay_count=1)
    try:
        baseline = worker._simulate_decode(
            max_new_tokens=1, top_k=6, score_candidate_text=False, score_observer_check=False)
        if not isinstance(baseline, Mapping) or not isinstance(baseline.get("first_logits"), torch.Tensor):
            raise ValueError(getattr(worker, "_last_simulate_decode_error", None) or "structure_readout_failed")
        rows, winner = readout_rows(baseline["first_logits"], current["source_candidates"])
        result.update(
            status="ok", new_measurement_count=1, structure_readout_rows=rows,
            greedy_token_id=winner, greedy_piece=worker.codec.decode([winner]),
            readout_stage="post_constraints_and_decoder_control",
            ordering=current["ordering"],
        )
    except Exception as exc:
        result.update(status="error", error=str(exc), new_measurement_count=0)
    finally:
        result["state_restored"] = before == state_identity(worker) and before_step == worker._steps
    if not result["state_restored"]:
        # A mutated diagnostic context must not be used as a new generation state.
        raise RuntimeError("source_clause_structure_probe_changed_worker_state")
    return result
