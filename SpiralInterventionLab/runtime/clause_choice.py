"""Offline raw-readout audit against a fixed, full-vocabulary baseline winner.

Relative logit movement is not probability lift, semantic success, or permission.
This consumes retained trajectories; it never runs a model or selects an edit.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import torch

from . import clause_trajectory


def summarize(records: Sequence[Mapping[str, Any]], *, component_labels: Sequence[str] = (),
              decode_token: Callable[[int], str] | None = None) -> dict[str, Any]:
    validated = clause_trajectory.summarize(records, component_labels=component_labels)
    baseline = clause_trajectory.replay_frames(records[0])[0]["vectors"][clause_trajectory.RAW_LOGITS].double()
    winner = int(baseline.argmax())
    baseline_ids = [int(f["vectors"][clause_trajectory.RAW_LOGITS].argmax())
                    for r in records for f in clause_trajectory.replay_frames(r)[:2]]
    result = {
        "schema_version": 1, "status": "ok", "fixed_baseline_winner_token_id": winner,
        "fixed_baseline_winner_piece": decode_token(winner) if decode_token else None,
        "baseline_winner_ids": baseline_ids,
        "baseline_winner_stable_across_conditions_and_repeats": all(t == winner for t in baseline_ids),
        "baseline_argmax_tie_count": int((baseline == baseline.max()).sum()),
        "anchor_contract": "first_no_edit_raw_argmax_never_reanchor_to_edited_winner",
        "probability_contract": "pre_constraint_full_vocabulary_softmax_cpu_float64",
        "margin_contract": "raw_target_logit_minus_fixed_baseline_winner_logit",
        "repeat_contract": "paired_deterministic_replays_not_independent_samples",
        "target_readout_effect_certified": False, "production_apply_allowed": False,
        "semantic_safety": "unassessed", "surfaces": [],
    }
    for index, verified in enumerate(validated["surfaces"]):
        responses = records[0]["result"]["operator_rows"][index]["structure_response_rows"]
        pieces = {r["token_id"]: r["piece"] for r in responses}
        if len(pieces) != len(responses) or any(t < 0 or t >= baseline.numel() for t in pieces):
            raise ValueError("invalid_or_duplicate_structure_readout_ids")
        entry = {"surface": verified["surface"], "structure_readout_token_ids": list(pieces),
                 "baseline_winner_in_structure_readout": winner in pieces, "conditions": []}
        for record in records:
            rows = record["result"]["operator_rows"][index]["structure_response_rows"]
            if {r["token_id"]: r["piece"] for r in rows} != pieces:
                raise ValueError("structure_readout_bindings_differ_across_conditions")
            frames = clause_trajectory.replay_frames(record)
            before = [f["vectors"][clause_trajectory.RAW_LOGITS].double() for f in frames[:2]]
            after = [f["vectors"][clause_trajectory.RAW_LOGITS].double()
                     for f in frames[2 + 2 * index:4 + 2 * index]]
            if any(v.shape != baseline.shape for v in before + after):
                raise ValueError("readout_vocabulary_shape_mismatch")
            probabilities = [(torch.softmax(b, dim=0), torch.softmax(a, dim=0))
                             for b, a in zip(before, after, strict=True)]
            edited_ids = [int(a.argmax()) for a in after]
            condition = {"label": record["label"], "kind": record["kind"],
                         "baseline_winner_ids": [int(b.argmax()) for b in before],
                         "edited_winner_ids": edited_ids,
                         "edited_winner_pieces": [decode_token(t) if decode_token else None for t in edited_ids],
                         "winner_changed_vs_fixed_baseline": [t != winner for t in edited_ids],
                         "edited_winner_in_structure_readout": [t in pieces for t in edited_ids],
                         "readout_rows": []}
            for tid in dict.fromkeys([*pieces, winner]):
                margins_before = [float(b[tid] - b[winner]) for b in before]
                margins_after = [float(a[tid] - a[winner]) for a in after]
                condition["readout_rows"].append({
                    "token_id": tid, "piece": pieces.get(tid, result["fixed_baseline_winner_piece"]),
                    "in_structure_readout": tid in pieces,
                    "baseline_logit_margins_vs_fixed_winner": margins_before,
                    "edited_logit_margins_vs_fixed_winner": margins_after,
                    "logit_margin_deltas_vs_fixed_winner": [a - b for b, a in zip(margins_before, margins_after)],
                    "raw_logit_deltas": [float(a[tid] - b[tid]) for b, a in zip(before, after)],
                    "raw_probabilities_before": [float(b[tid]) for b, _ in probabilities],
                    "raw_probabilities_after": [float(a[tid]) for _, a in probabilities],
                    "raw_probability_deltas": [float(a[tid] - b[tid]) for b, a in probabilities],
                })
            entry["conditions"].append(condition)
        result["surfaces"].append(entry)
    return result
