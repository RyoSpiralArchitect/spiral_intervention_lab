"""Prefix-only position facts, never a scheduler or an efficacy prediction."""
from __future__ import annotations

from typing import Any
import unicodedata


def output_tokens(worker: Any) -> tuple[int, ...]:
    return tuple(int(token) for segment in worker._segments if segment.kind == "output"
                 for token in segment.token_ids)


def describe(worker: Any) -> dict[str, Any]:
    tokens = output_tokens(worker)
    prefix = worker.final_text()
    if not prefix:
        trailing = "empty"
    elif prefix[-1].isspace():
        trailing = "whitespace"
    elif unicodedata.category(prefix[-1]).startswith("P"):
        trailing = "punctuation"
    else:
        trailing = "open_text"
    return {"worker_step": worker._steps, "generated_token_count": len(tokens),
            "prefix_tail": prefix[-64:], "last_token_piece": worker.codec.decode(tokens[-1:])[-32:] if tokens else "",
            "trailing_surface": trailing, "word_completion": "unknown_without_lookahead",
            "scope": "observed_prefix_only_not_position_quality"}


def history(worker: Any, candidate_id: str, context: str) -> dict[str, Any]:
    records = [(key, row) for key, row in worker._candidate_measurements.items() if key[0] == candidate_id]
    current = output_tokens(worker)
    recent = []
    for key, row in records[-2:]:
        previous = worker._candidate_measurement_positions.get(key)
        extends = previous is not None and current[:len(previous)] == previous
        relation = "unknown" if previous is None else "same" if current == previous else "extended" if extends else "diverged"
        metrics = row.get("metrics") or {}
        recent.append({"measurement_context_id": key[1], "same_runtime_context": key[1] == context,
            "position": row.get("measurement_position"), "prefix_relation": relation,
            "new_output_tokens": len(current) - len(previous) if extends else None,
            "metrics": {k: metrics[k] for k in (
                "target_top20_threshold_gap_delta", "target_piece_prob_delta", "target_piece_logit_delta",
                "target_rank_before", "target_rank_after", "bound_token_top20_hit_delta") if k in metrics},
            "state_restored": row.get("state_restored") is True})
    return {"scope": "same_frozen_candidate_only", "measured_context_count": len(records),
            "omitted_context_count": max(0, len(records) - 2), "recent": recent}
