from __future__ import annotations

from SpiralInterventionLab.examples.digit_transform_e2e import create_task_env
from SpiralInterventionLab.examples.replay_operator_dose_response import (
    RELEASED,
    minimal_winning_dose,
    released_replay_policy,
)
from SpiralInterventionLab.examples.replay_orphan_blueprint_ceiling import _scored
from SpiralInterventionLab.examples.replay_structure_token_probe import retarget_readout, single_token_id, token_rank
from SpiralInterventionLab.runtime.policy import HarnessPolicy


def test_minimal_winning_dose_ignores_order_and_losses():
    points = [{"dose": 5.12, "won": True}, {"dose": 2.56, "won": False}, {"dose": 10.24, "won": True}]

    assert minimal_winning_dose(points) == 5.12
    assert minimal_winning_dose([{"dose": 0.08, "won": False}]) is None


def test_released_policy_only_lifts_main_alpha_and_cost_budgets():
    default = HarnessPolicy.default_v0()
    released = released_replay_policy(max_edits_per_step_override=2)

    assert released.global_budget.max_total_alpha == RELEASED
    assert released.global_budget.max_total_edit_cost == RELEASED
    assert released.global_budget.max_edits_per_step == 2
    assert released.global_budget.max_production_trial_total_alpha == default.global_budget.max_production_trial_total_alpha
    assert HarnessPolicy.default_v0().global_budget.max_total_alpha == default.global_budget.max_total_alpha


def test_scored_continuations_carry_role_binding_for_ladder_tasks():
    task = create_task_env("constrained_rewrite_l1")
    task.reset(0)

    correct = _scored(task, " Nora will take the sample to Ivo before Selim sends the report to Yara.")
    swapped = _scored(task, " Nora will take the sample to Ivo before sending Selim's report to Yara before dusk.")

    assert correct["role_binding_ok"] and correct["role_violations"] == []
    assert swapped["score"] == 1.0 and not swapped["role_binding_ok"]


class _Codec:
    def __init__(self, table):
        self.table = table

    def encode(self, piece):
        import torch

        return torch.tensor([[0, *self.table[piece]]])


def test_single_token_id_strips_special_ids_and_rejects_multi_token_pieces():
    codec = _Codec({",": [11], " whereas": [1405, 300]})

    assert single_token_id(codec, ",", special_ids={0}) == 11
    assert single_token_id(codec, " whereas", special_ids={0}) is None


def test_retarget_readout_rewrites_every_direction_node_only():
    edit = {"source": {"dtype": "vector", "expr": {"fn": "clip_norm", "arg": {
        "fn": "readout_direction", "target_token_ids": [24082], "negative_token_ids": [279]}}}}

    assert retarget_readout(edit, 11) == 1
    node = edit["source"]["expr"]["arg"]
    assert node["target_token_ids"] == [11] and node["negative_token_ids"] == []
    assert retarget_readout({"source": {"expr": {"ref": {}}}}, 11) == 0


def test_token_rank_is_exact_over_the_full_vocabulary():
    import torch

    logits = torch.tensor([0.5, 3.0, 2.0, 3.0])

    assert token_rank(logits, 2) == 3
    assert token_rank(logits, 1) == 1


def test_survey_rescue_requires_full_score_no_violations_and_correct_roles():
    from SpiralInterventionLab.examples.survey_clause_structure import is_rescue

    assert is_rescue({"score": 1.0, "constraint_violations": [], "role_binding_ok": True})
    assert not is_rescue({"score": 1.0, "constraint_violations": [], "role_binding_ok": False})
    assert not is_rescue({"score": 1.0, "constraint_violations": ["word_budget"], "role_binding_ok": True})
    assert not is_rescue({"score": 0.938889, "constraint_violations": [], "role_binding_ok": True})


def test_survey_first_piece_takes_the_first_non_special_token_of_multi_piece_names():
    from SpiralInterventionLab.examples.survey_clause_structure import first_piece_id

    codec = _Codec({" Selim": [24082, 318]})

    assert first_piece_id(codec, " Selim", special_ids={0}) == 24082


def test_patch_edit_size_reads_activation_patch_telemetry_only():
    from SpiralInterventionLab.examples.replay_structure_token_probe import patch_edit_size

    simulated = {"edit_runtime_telemetry": [
        {"op": "resid_add", "target_norm_before": 1.0, "blend_delta_norm": 9.0},
        {"op": "activation_patch", "target_norm_before": 63.445, "blend_delta_norm": 19.8125},
    ]}

    assert patch_edit_size(simulated) == {"site_norm_before": 63.445, "blend_delta_norm": 19.8125,
                                          "blend_delta_over_site_norm": 0.31228}
    assert patch_edit_size({"edit_runtime_telemetry": []}) == {}
