from copy import deepcopy

import pytest
import torch

from SpiralInterventionLab.runtime import clause_choice, clause_trajectory
from SpiralInterventionLab.tests.test_clause_trajectory import worker_and_records


def test_fixed_full_vocabulary_winner_is_not_the_selected_structure_token():
    _, records = worker_and_records()
    baseline = torch.tensor([-5., 0., -5., -1., -5., 2., 1.], device="cpu")
    edited = baseline.clone()
    edited[1] += 1.1
    edited[5] += 1.
    edited[6] += 3.
    for record in records:
        frames = clause_trajectory.replay_frames(record)
        for i, frame in enumerate(frames):
            frame["vectors"][clause_trajectory.RAW_LOGITS] = (
                edited if i >= 2 and record["kind"] == "source" else baseline).clone()
    summary = clause_choice.summarize(records, decode_token=lambda tid: f"piece:{tid}")
    assert summary["fixed_baseline_winner_token_id"] == 5
    assert summary["baseline_winner_stable_across_conditions_and_repeats"]
    surface = summary["surfaces"][0]
    assert not surface["baseline_winner_in_structure_readout"]
    condition = surface["conditions"][0]
    assert condition["edited_winner_ids"] == [6, 6]
    target = next(r for r in condition["readout_rows"] if r["token_id"] == 1)
    assert target["logit_margin_deltas_vs_fixed_winner"] == pytest.approx([0.1, 0.1])
    assert all(d < 0 for d in target["raw_probability_deltas"])
    assert next(r for r in condition["readout_rows"] if r["token_id"] == 5)["piece"] == "piece:5"
    assert not summary["target_readout_effect_certified"] and not summary["production_apply_allowed"]


def test_winner_instability_is_visible_without_reanchoring():
    _, records = worker_and_records()
    baseline = clause_trajectory.replay_frames(records[0])[0]["vectors"][clause_trajectory.RAW_LOGITS]
    first_winner = int(baseline.argmax())
    records[1]["frames"][0]["vectors"][clause_trajectory.RAW_LOGITS][5] = 100.
    summary = clause_choice.summarize(records)
    assert summary["fixed_baseline_winner_token_id"] == first_winner
    assert not summary["baseline_winner_stable_across_conditions_and_repeats"]
    assert 5 in summary["baseline_winner_ids"]


def test_choice_audit_reuses_trajectory_validation_and_rejects_readout_rebinding():
    _, records = worker_and_records()
    bad = deepcopy(records)
    bad[0]["result"]["state_restored"] = False
    with pytest.raises(ValueError, match="unrestored"):
        clause_choice.summarize(bad)
    bad = deepcopy(records)
    bad[1]["result"]["operator_rows"][0]["structure_response_rows"][0]["piece"] = "other"
    with pytest.raises(ValueError, match="readout_bindings"):
        clause_choice.summarize(bad)
