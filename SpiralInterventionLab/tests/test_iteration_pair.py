from pathlib import Path

import pytest

from SpiralInterventionLab.examples.digit_transform_e2e import _build_parser, create_task_env
from SpiralInterventionLab.examples.run_iteration_pair import PROFILES, build_argv, validate_action_identity


@pytest.mark.parametrize("profile", PROFILES)
def test_iteration_pair_preserves_model_local_fixture_and_apply_policy(profile):
    checkpoint = Path("/supplied/model")
    argv = build_argv(profile, checkpoint, Path("/new/results"), device="cpu", controller="test-controller")
    args = _build_parser().parse_args(argv)
    spec = PROFILES[profile]
    assert args.worker_model_path == str(checkpoint) and args.worker_hf_offline
    assert args.worker_model == spec["worker_model"] and args.worker_dtype == spec["dtype"]
    assert args.task == spec["task"] and args.seed == spec["seed"]
    assert args.worker_first_n_layers is None and args.num_seeds == 1
    assert args.no_b1 and not args.c1_only
    assert args.max_diagnostic_calls_per_run == 12 and args.diagnostic_result_window == 12
    assert args.worker_decoder_control_mode == "off" and args.worker_loop_rescue_edits_per_run == 0
    env = create_task_env(args.task)
    env.reset(args.seed)
    assert env.worker_runtime_kwargs()["max_generated_tokens"] == (18 if profile == "gpt2_control" else 64)


def test_pair_audit_distinguishes_new_normal_cap_candidate_from_parent():
    offered = {"action": "measure_normal_cap_current_prefix", "candidate_id": "parent", "new_candidate_id": "child"}
    row = {"candidate_id": "child", "parent_candidate_id": "parent", "inherits_measurement": False,
           "executed_action": offered["action"]}
    validate_action_identity(row, offered)
    validate_action_identity({"candidate_id": "parent", "executed_action": None}, offered)
    for invalid in ({**row, "candidate_id": "parent"}, {**row, "parent_candidate_id": "wrong"},
                    {**row, "inherits_measurement": True}):
        with pytest.raises(AssertionError):
            validate_action_identity(invalid, offered)


def test_pair_audit_accepts_bounded_investigation_child_identity():
    offered = {"action": "investigate_normal_cap_current_prefix", "candidate_id": "parent",
               "new_candidate_id": "child"}
    row = {"candidate_id": "child", "parent_candidate_id": "parent", "inherits_measurement": False,
           "executed_action": offered["action"]}
    validate_action_identity(row, offered)
    with pytest.raises(AssertionError):
        validate_action_identity({**row, "candidate_id": "parent"}, offered)
