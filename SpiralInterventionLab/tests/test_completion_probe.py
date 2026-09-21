from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.examples.replay_prefix_response import completion_reference
from SpiralInterventionLab.examples.rewrite_ladder_baseline import sha256_file
from SpiralInterventionLab.runtime.completion_probe import (
    competition_report, completion_at_start, completion_binding,
)
from SpiralInterventionLab.runtime.prefix_probe import measure_prefix
from SpiralInterventionLab.tests.test_prefix_probe import prefix_worker


def completion_worker(long_name=False):
    worker, frozen, _ = prefix_worker()
    pieces = {0: " sending", 1: " before", 30: " Sel", 31: "im", 32: " sends", 33: ".", 34: "x"}
    term = "Selimx" if long_name else "Selim"
    ids = [30, 31, 34] if long_name else [30, 31]
    worker.codec = SimpleNamespace(decode=lambda seq: "".join(pieces.get(t, "?") for t in seq),
        encode=lambda text: torch.tensor(ids if text == " " + term else [1], device="cpu"))
    worker.stop_checker = lambda text: text.endswith(".")
    frozen = replace(frozen, term=term, token_id=30, token_piece=" Sel")
    calls = []
    def simulate(**kwargs):
        prefix = list(worker._segments[0].token_ids)
        calls.append({"prefix": prefix, **kwargs})
        logits = torch.full((40,), -10.0, device="cpu")
        if prefix[-1] == 30:
            logits[31] = 5.0
        elif prefix[-1] == 31:
            logits[32] = 5.0
        elif len(prefix) == 1:
            logits[0], logits[30] = 5.0, 1.0
        else:
            logits[33] = 5.0
        if kwargs.get("command"):
            logits[30] += 0.2
            logits[0] -= 0.3
        return {"first_logits": logits, "continuation_token_ids": [int(logits.argmax())],
                "edit_runtime_telemetry": [{"op": "activation_patch", "hook_call_count": 1}] if kwargs.get("command") else []}
    worker._simulate_decode = simulate
    binding = completion_binding(worker.codec, term=term, surface=" " + term,
                                 first_token_id=30, competitor_token_id=0)
    return worker, frozen, binding, calls


def measure(worker, frozen, binding):
    # Fixture state identity changes with any leaked forced output token.
    state = lambda w: str(w._segments[0].token_ids)
    with patch("SpiralInterventionLab.runtime.prefix_probe.state_identity", side_effect=state), \
         patch("SpiralInterventionLab.runtime.completion_probe.state_identity", side_effect=state):
        return measure_prefix(worker, frozen, horizon=8, completion=binding)


def test_competition_and_name_branch_do_not_turn_forcing_into_success():
    worker, frozen, binding, calls = completion_worker()
    segments = deepcopy(worker._segments)
    result = measure(worker, frozen, binding)
    assert result["status"] == "complete", result
    extra = result["completion_probe"]
    competition = extra["competition"]
    assert competition["competitor_minus_target_gap_delta"] == pytest.approx(-0.5)
    assert not competition["target_won_edited_first_token"]
    assert not result["baseline"]["whole_term_present"] and not result["edited"]["whole_term_present"]
    branch = extra["forced_first_piece_branch"]
    assert branch["forced_token_ids"] == [30] and branch["generated_suffix_token_ids"] == [31, 32, 33]
    assert branch["whole_term_completed_at_branch_start"]
    assert not branch["eligible_for_task_comparison"] and branch["apply_count"] == 0
    assert "legacy_score" not in branch
    assert extra["conditional_token_factors"][0]["token_id"] == 31
    assert extra["conditional_token_factors"][0]["rank"] == 1
    sequence = extra["canonical_sequence_prefix_probability"]
    expected = competition["baseline"]["target"]["probability"] * extra["suffix_probability_given_forced_first_piece"]
    assert sequence["baseline"]["probability"] == pytest.approx(expected)
    assert sequence["delta"] > 0 and not sequence["includes_word_boundary"] and not sequence["is_greedy_success_rate"]
    assert result["physical_replay_count"] == 4 and result["total_physical_replay_count"] == 6
    assert result["model_forward_token_count"] == 8 and result["total_model_forward_token_count"] == 14
    assert sum(bool(c.get("command")) for c in calls) == 2
    assert all(not c.get("command") for c in calls if 30 in c["prefix"])
    assert worker._segments == segments and worker._last_packet == {"original": True}
    assert worker.runtime_state.trace_caches == {} and result["state_restored"] and extra["state_restored"]
    assert not result["production_apply_allowed"] and not extra["certified_for_apply"]


def test_long_name_scores_forced_factors_without_inventing_a_natural_completion():
    worker, frozen, binding, calls = completion_worker(long_name=True)
    result = measure(worker, frozen, binding)
    assert result["status"] == "complete", result
    extra = result["completion_probe"]
    assert not extra["forced_first_piece_branch"]["whole_term_completed_at_branch_start"]
    factors = extra["conditional_token_factors"]
    assert [f["position"] for f in factors] == [1, 2]
    assert factors[1]["conditioned_on_target_token_ids"] == [30, 31]
    assert factors[1]["token_id"] == 34 and factors[1]["greedy_token_id"] == 32
    assert result["total_physical_replay_count"] == 8


def test_conditional_failure_is_incomplete_and_restores_primary_state():
    worker, frozen, binding, calls = completion_worker()
    original = worker._simulate_decode
    worker._last_simulate_decode_error = "conditional failure"
    worker._simulate_decode = lambda **kwargs: None if 30 in worker._segments[0].token_ids else original(**kwargs)
    result = measure(worker, frozen, binding)
    assert result["status"] == "incomplete" and result["primary_natural_probe_status"] == "complete"
    assert result["completion_probe"]["status"] == "incomplete" and result["state_restored"]
    assert worker._segments[0].token_ids == [1] and worker.runtime_state.trace_caches == {}
    assert result["completion_probe"]["physical_replay_count"] == 1


@pytest.mark.parametrize("change", ["free_text", "wrong_first_piece", "bool_id", "competitor_is_target"])
def test_completion_binding_rejects_retargeting_or_answer_text(change):
    worker, _, _, _ = completion_worker()
    kwargs = dict(term="Selim", surface=" Selim", first_token_id=30, competitor_token_id=0)
    if change == "free_text":
        kwargs["surface"] = " Selim sends a report"
    elif change == "wrong_first_piece":
        kwargs["first_token_id"] = 31
    elif change == "bool_id":
        kwargs["competitor_token_id"] = True
    else:
        kwargs["competitor_token_id"] = 30
    with pytest.raises(ValueError):
        completion_binding(worker.codec, **kwargs)


def test_word_boundary_and_horizon_are_not_mistaken_for_name_completion():
    worker, _, binding, _ = completion_worker()
    assert completion_at_start(worker.codec, binding, [], stopped=False)["status"] == "horizon_unresolved"
    assert completion_at_start(worker.codec, binding, [31], stopped=False)["status"] == "word_boundary_unresolved"
    assert completion_at_start(worker.codec, binding, [31], stopped=True)["whole_term_completed_at_branch_start"]
    assert completion_at_start(worker.codec, binding, [31, 34], stopped=False)["status"] == "continued_into_another_word"
    assert completion_at_start(worker.codec, binding, [31, 33], stopped=True)["whole_term_completed_at_branch_start"]


def test_competition_refuses_a_different_recorded_winner_or_invalid_logits():
    worker, _, binding, _ = completion_worker()
    logits = torch.zeros(40, device="cpu")
    logits[2] = 10
    with pytest.raises(ValueError, match="no_longer"):
        competition_report(logits, logits, binding, worker.codec)
    logits[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        competition_report(logits, logits, binding, worker.codec)


def test_completion_reference_is_hash_bound_and_does_not_search_for_better_prefix(tmp_path):
    plan = {"points": [{"worker_step": 9, "prefix": "before"}],
            "candidate": {"target_piece_token_id": 30, "source_tensor_identity": {"sha256": "s"},
                          "activation_patch_step_size": 0.16, "activation_patch_alpha": 0.04}}
    row = {"worker_step": 9, "prefix": "before", "status": "complete", "state_restored": True,
           "candidate_id": "f", "target_piece_token_id": 30, "source_tensor_identity": {"sha256": "s"},
           "measurement_context_id": "c", "baseline": {"token_ids": [0]}}
    (tmp_path / "rows.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "status.json").write_text(json.dumps({"status": "complete", "rows_sha256": sha256_file(tmp_path / "rows.jsonl")}))
    (tmp_path / "manifest.json").write_text(json.dumps({"source_jsonl_sha256": "log"}))
    candidate = {"candidate_id": "f", "edit": {"budget": {"step_size": 0.16}, "op": {"alpha": 0.04}}}
    (tmp_path / "frozen_candidate.json").write_text(json.dumps(candidate))
    ref = completion_reference(tmp_path, plan, worker_step=9, source_sha256="log")
    assert ref["competitor_token_id"] == 0 and ref["measurement_context_id"] == "c"
    with pytest.raises(ValueError, match="not_in_frozen_plan"):
        completion_reference(tmp_path, plan, worker_step=10, source_sha256="log")
    with pytest.raises(ValueError, match="hash_verified"):
        completion_reference(tmp_path, plan, worker_step=9, source_sha256="different")
    (tmp_path / "rows.jsonl").write_text(json.dumps({**row, "prefix": "changed"}) + "\n")
    with pytest.raises(ValueError, match="hash_verified"):
        completion_reference(tmp_path, plan, worker_step=9, source_sha256="log")
