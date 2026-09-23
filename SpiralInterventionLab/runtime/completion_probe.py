"""Offline competition and conditional name-completion measurements.

Conditioned branches use only the recorded required-term tokenization. They
are not generated successes, candidate sources, or production permissions.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import re
from typing import Any

import torch

from .prefix_probe import _one_token_then_unedited, _release_probe_buffers
from .response_probe import logit_variation, state_identity


def completion_binding(codec: Any, *, term: str, surface: str,
                       first_token_id: int, competitor_token_id: int) -> dict[str, Any]:
    if not term or surface.strip() != term or not re.fullmatch(r"\s*" + re.escape(term), surface):
        raise ValueError("completion_surface_must_be_only_the_recorded_term")
    ids = codec.encode(surface).detach().reshape(-1).cpu().tolist()
    if not 2 <= len(ids) <= 8 or ids[0] != first_token_id or codec.decode(ids) != surface:
        raise ValueError("completion_tokenization_does_not_match_frozen_first_piece")
    for token in [*ids, first_token_id, competitor_token_id]:
        if isinstance(token, bool) or not isinstance(token, int) or token < 0:
            raise ValueError("invalid_completion_token_id")
    if competitor_token_id == first_token_id:
        raise ValueError("competitor_is_target")
    return {"term": term, "surface": surface, "token_ids": ids,
            "pieces": [codec.decode([token]) for token in ids],
            "first_token_id": first_token_id, "competitor_token_id": competitor_token_id,
            "competitor_piece": codec.decode([competitor_token_id]),
            "competitor_source": "recorded_no_edit_greedy_next_token",
            "forced_prefix_source": "recorded_required_term_tokenization",
            "conditional_branch_is_task_result": False, "production_apply_allowed": False}


def token_readout(logits: torch.Tensor, token_id: int, codec: Any) -> dict[str, Any]:
    logits = logits.detach().cpu().flatten().double()
    if not 0 <= token_id < logits.numel() or not torch.isfinite(logits[token_id]):
        raise ValueError("unavailable_completion_token")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("nonfinite_completion_readout")
    logp = logits.log_softmax(0)
    winner = int(logits.argmax())
    return {"token_id": token_id, "piece": codec.decode([token_id]),
            "logit": float(logits[token_id]), "log_probability": float(logp[token_id]),
            "probability": float(logp[token_id].exp()),
            "rank": int((logits > logits[token_id]).sum()) + 1,
            "greedy_token_id": winner, "greedy_piece": codec.decode([winner])}


def competition_report(before: torch.Tensor, after: torch.Tensor, binding: Mapping[str, Any],
                       codec: Any) -> dict[str, Any]:
    logit_variation(before, after)  # Reject changed masks, shapes, and nonfinite logits.
    target, competitor = binding["first_token_id"], binding["competitor_token_id"]
    rows = {}
    for label, logits in (("baseline", before), ("edited", after)):
        t, c = token_readout(logits, target, codec), token_readout(logits, competitor, codec)
        rows[label] = {"target": t, "competitor": c,
                       "competitor_minus_target_logit_gap": c["logit"] - t["logit"],
                       "target_to_competitor_probability_ratio": math.exp(t["log_probability"] - c["log_probability"])}
    if rows["baseline"]["competitor"]["greedy_token_id"] != competitor:
        raise ValueError("recorded_competitor_is_no_longer_baseline_winner")
    gap_before = rows["baseline"]["competitor_minus_target_logit_gap"]
    gap_after = rows["edited"]["competitor_minus_target_logit_gap"]
    return {**rows, "competitor_minus_target_gap_delta": gap_after - gap_before,
            "target_vs_competitor_log_odds_delta": gap_before - gap_after,
            "target_won_edited_first_token": rows["edited"]["target"]["greedy_token_id"] == target,
            "competitor_binding_frozen": True, "production_apply_allowed": False}


def completion_at_start(codec: Any, binding: Mapping[str, Any], suffix_ids: list[int], *, stopped: bool) -> dict[str, Any]:
    ids = [binding["first_token_id"], *suffix_ids]
    text = codec.decode(ids)
    surface = binding["surface"]
    completed = ids[:len(binding["token_ids"])] == binding["token_ids"]
    if not completed or not text.startswith(surface):
        status = "not_completed" if not surface.startswith(text) else "horizon_unresolved"
    elif len(text) == len(surface):
        status = "whole_term_completed" if stopped else "word_boundary_unresolved"
    else:
        status = "whole_term_completed" if not re.match(r"\w", text[len(surface)]) else "continued_into_another_word"
    return {"status": status, "canonical_token_sequence_completed": completed,
            "whole_term_completed_at_branch_start": status == "whole_term_completed",
            "forced_first_token_is_not_a_generated_success": True}


def measure_completion(worker: Any, frozen: Any, before: torch.Tensor, after: torch.Tensor,
                       binding: Mapping[str, Any], *, horizon: int) -> dict[str, Any]:
    """Measure the entry competition, then a shared unedited conditional branch."""
    # Revalidate instead of trusting a caller-supplied chain or free-form prefix.
    verified = completion_binding(worker.codec, term=frozen.term, surface=binding["surface"],
        first_token_id=frozen.token_id, competitor_token_id=binding["competitor_token_id"])
    if dict(binding) != verified or horizon <= len(binding["token_ids"]):
        raise ValueError("invalid_completion_binding_or_insufficient_horizon")
    if worker._collect_active_edits():
        raise ValueError("conditional_branch_requires_rolled_back_edit")
    context, segments = state_identity(worker), deepcopy(worker._segments)
    saved_packet = worker._last_packet
    saved_alignment = worker.runtime_state.trace_alignment_step
    report = {"status": "incomplete", "binding": dict(binding), "measurement_context_id": context,
              "physical_replay_count": 0, "completed_replay_count": 0, "model_forward_token_count": 0,
              "conditional_only": True, "eligible_for_task_comparison": False,
              "production_apply_allowed": False, "certified_for_apply": False}
    try:
        report["competition"] = competition_report(before, after, binding, worker.codec)
        worker._append_output_token(binding["first_token_id"])
        worker._last_packet = None
        conditional = []
        for _ in range(2):
            report["physical_replay_count"] += 1
            trial = _one_token_then_unedited(worker, horizon=horizon - 1, edit=None)
            report["completed_replay_count"] += 1
            report["model_forward_token_count"] += len(trial["continuation_token_ids"])
            conditional.append(trial)
        first, repeat = conditional
        control = logit_variation(first["first_logits"], repeat["first_logits"])
        if control["max_abs_logit_delta"] != 0 or first["continuation_token_ids"] != repeat["continuation_token_ids"]:
            raise ValueError("conditional_completion_repeat_drift")
        suffix = first["continuation_token_ids"]
        branch_text = worker.final_text() + first["continuation"]
        stopped = bool(suffix[-1] in worker.stop_token_ids or (worker.stop_checker and worker.stop_checker(branch_text)))
        report["forced_first_piece_branch"] = {
            "forced_token_ids": [binding["first_token_id"]], "generated_suffix_token_ids": suffix,
            "generated_suffix": first["continuation"], "conditional_text": branch_text,
            "repeat_control": control, "stopped": stopped, "apply_count": 0,
            **completion_at_start(worker.codec, binding, suffix, stopped=stopped),
            "eligible_for_task_comparison": False, "production_apply_allowed": False}
        factors = [{"position": 1, "conditioned_on_target_token_ids": binding["token_ids"][:1],
                    **token_readout(first["first_logits"], binding["token_ids"][1], worker.codec),
                    "repeat_control": control}]
        # Longer names require teacher-forced partial-name prefixes. These
        # factors describe one tokenization, not a natural generated trajectory.
        for index in range(2, len(binding["token_ids"])):
            worker._segments = deepcopy(segments)
            for token in binding["token_ids"][:index]:
                worker._append_output_token(token)
            worker._last_packet = None
            pair = []
            for _ in range(2):
                report["physical_replay_count"] += 1
                trial = _one_token_then_unedited(worker, horizon=1, edit=None)
                pair.append(trial)
                report["completed_replay_count"] += 1
                report["model_forward_token_count"] += 1
                _release_probe_buffers(worker)
            control = logit_variation(pair[0]["first_logits"], pair[1]["first_logits"])
            if control["max_abs_logit_delta"] != 0:
                raise ValueError("conditional_factor_repeat_drift")
            factors.append({"position": index, "conditioned_on_target_token_ids": binding["token_ids"][:index],
                **token_readout(pair[0]["first_logits"], binding["token_ids"][index], worker.codec),
                "repeat_control": control})
        suffix_logp = sum(factor["log_probability"] for factor in factors)
        sequence = {}
        for label, logits in (("baseline", before), ("edited", after)):
            entry_logp = token_readout(logits, frozen.token_id, worker.codec)["log_probability"]
            sequence[label] = {"log_probability": entry_logp + suffix_logp,
                               "probability": math.exp(entry_logp + suffix_logp)}
        report.update(status="complete", conditional_token_factors=factors,
            suffix_probability_given_forced_first_piece=math.exp(suffix_logp),
            canonical_sequence_prefix_probability={**sequence,
                "delta": sequence["edited"]["probability"] - sequence["baseline"]["probability"],
                "includes_word_boundary": False, "includes_all_tokenizations": False,
                "suffix_shared_because": "TTL1 rolled back before full-recompute unedited suffix; identical forced tokens",
                "is_greedy_success_rate": False})
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}:{exc}", model_forward_token_count_is_lower_bound=True)
    finally:
        worker._segments = segments
        worker._last_packet = saved_packet
        worker.runtime_state.trace_alignment_step = saved_alignment
    report["state_restored"] = state_identity(worker) == context and not worker._collect_active_edits()
    if not report["state_restored"]:
        report.update(status="incomplete", error="conditional_state_restoration_failed")
    return report
