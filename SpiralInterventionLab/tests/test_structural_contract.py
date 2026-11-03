from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from SpiralInterventionLab.runtime import structural_contract
from SpiralInterventionLab.runtime.compiler import StepContext, _resolved_step_size, compile_command
from SpiralInterventionLab.runtime.edit_budget import (
    MAIN_EDIT_BUDGET_POOL,
    NORM_RELATIVE_WRITE_CONTRACT,
    classify_edit_budget_pool,
    clip_relative_write,
    relative_write_cap,
)
from SpiralInterventionLab.runtime.policy import (
    GlobalBudget,
    HarnessPolicy,
    PolicyViolation,
    WriteContract,
    command_budget_usage,
    validate_command_against_packet,
)
from SpiralInterventionLab.runtime.schema import parse_controller_command, parse_observation_packet
from SpiralInterventionLab.tests import test_controller_runtime as fixtures


CONTRACT_POLICY = HarnessPolicy(write_contract=WriteContract(
    relative_write_cap=0.08, readout_token_ids=(5,), source_positions=(7,)))


def readout_command(**changes):
    edit = {
        "id": "e_contract",
        "target": {"surface_id": "s_resid_l11_last"},
        "source": {"dtype": "vector",
                   "expr": {"fn": "readout_direction", "target_token_ids": [5], "normalize": True}},
        "op": {"kind": "resid_add", "alpha": 3.0},
        "budget": {"ttl_steps": 1, "revertible": True},
        "meta": {"write_contract": NORM_RELATIVE_WRITE_CONTRACT, "relative_write_cap": 0.08},
    }
    for key, value in changes.items():
        edit[key] = value
    return {"version": "0.1", "decision": "apply", "edits": [edit]}


def patch_command(token):
    command = readout_command(
        op={"kind": "activation_patch", "mode": "blend", "alpha": 1.0},
        source={"dtype": "vector", "expr": {"fn": "mean", "args": [
            {"ref": {"scope": "runtime", "worker": "os_0", "tensor": "hidden", "layer": 11, "token": token}},
            {"ref": {"scope": "runtime", "worker": "os_0", "tensor": "hidden", "layer": 11,
                     "token": {"mode": "last"}}}]}})
    return command


def packet():
    return fixtures._make_packet()


def test_default_policy_rejects_contract_and_contract_policy_supersedes_absolute_alpha():
    with pytest.raises(PolicyViolation, match="does not enable"):
        validate_command_against_packet(readout_command(), packet())
    # alpha 3.0 is far above the surface's 0.2 cap; the hook-time relative cap replaces it.
    validate_command_against_packet(readout_command(), packet(), policy=CONTRACT_POLICY)


def test_contract_policy_keeps_absolute_caps_for_undeclared_edits():
    plain = readout_command(meta={}, op={"kind": "resid_add", "alpha": 0.3})
    with pytest.raises(PolicyViolation, match="surface alpha cap"):
        validate_command_against_packet(plain, packet(), policy=CONTRACT_POLICY)


@pytest.mark.parametrize("meta", [
    {"write_contract": "other_contract", "relative_write_cap": 0.08},
    {"write_contract": NORM_RELATIVE_WRITE_CONTRACT},
    {"relative_write_cap": 0.08},
    {"write_contract": NORM_RELATIVE_WRITE_CONTRACT, "relative_write_cap": True},
    {"write_contract": NORM_RELATIVE_WRITE_CONTRACT, "relative_write_cap": 1.5},
    {"write_contract": NORM_RELATIVE_WRITE_CONTRACT, "relative_write_cap": float("nan")},
])
def test_partial_or_malformed_declarations_fail_closed(meta):
    with pytest.raises(PolicyViolation):
        validate_command_against_packet(readout_command(meta=meta), packet(), policy=CONTRACT_POLICY)


@pytest.mark.parametrize("change", [
    {"source": {"dtype": "vector", "expr": {"fn": "readout_direction", "target_token_ids": [4]}}},
    {"source": {"dtype": "vector",
                "expr": {"fn": "readout_direction", "target_token_ids": [5], "negative_token_ids": [4]}}},
    {"meta": {"write_contract": NORM_RELATIVE_WRITE_CONTRACT, "relative_write_cap": 0.09}},
    {"meta": {"write_contract": NORM_RELATIVE_WRITE_CONTRACT, "relative_write_cap": 0.08,
              "apply_kind": "production_trial"}},
    {"budget": {"ttl_steps": 2, "revertible": True}},
    {"budget": {"ttl_steps": 1, "step_size": 0.05, "revertible": True}},
    {"op": {"kind": "resid_add", "alpha": 2048.0}},
    {"op": {"kind": "activation_patch", "mode": "blend", "alpha": 1.5}},
])
def test_contract_rejects_targets_doses_and_lifetimes_outside_the_contract(change):
    with pytest.raises(PolicyViolation):
        validate_command_against_packet(readout_command(**change), packet(), policy=CONTRACT_POLICY)


def test_contract_admits_only_allowlisted_runtime_positions_and_one_edit():
    validate_command_against_packet(patch_command({"mode": "index", "value": 7}), packet(), policy=CONTRACT_POLICY)
    for token in ({"mode": "index", "value": 8}, {"mode": "span", "start": 7, "end": 8, "pool": "mean"}):
        with pytest.raises(PolicyViolation, match="allowlist"):
            validate_command_against_packet(patch_command(token), packet(), policy=CONTRACT_POLICY)
    traced = patch_command({"mode": "last"})
    traced["edits"][0]["source"]["expr"]["args"][0]["ref"].update(scope="trace", trace_id="best_success")
    with pytest.raises(PolicyViolation, match="allowlist"):
        validate_command_against_packet(traced, packet(), policy=CONTRACT_POLICY)
    pair = readout_command()
    pair["edits"].append({**deepcopy(pair["edits"][0]), "id": "e_second"})
    roomy = packet()
    roomy["budget"]["edits_left_this_step"] = 2
    policy = HarnessPolicy(global_budget=GlobalBudget(max_edits_per_step=2),
                           write_contract=CONTRACT_POLICY.write_contract)
    with pytest.raises(PolicyViolation, match="exactly one edit"):
        validate_command_against_packet(pair, roomy, policy=policy)


def test_contract_rejects_weight_surfaces():
    command = readout_command(target={"surface_id": "s_weight_l7_mlp"},
                              op={"kind": "rank1_patch", "alpha": 0.05},
                              source={"dtype": "rank1",
                                      "u": {"fn": "readout_direction", "target_token_ids": [5]},
                                      "v": {"fn": "readout_direction", "target_token_ids": [5]}})
    with pytest.raises(PolicyViolation, match="last-token residual"):
        validate_command_against_packet(command, packet(), policy=CONTRACT_POLICY)


def test_contract_usage_is_charged_in_relative_units():
    usage = command_budget_usage(readout_command(), packet(), policy=CONTRACT_POLICY)
    assert usage[MAIN_EDIT_BUDGET_POOL]["alpha"] == pytest.approx(0.08)
    assert usage[MAIN_EDIT_BUDGET_POOL]["edit_cost"] == pytest.approx(0.08)
    usage = command_budget_usage(patch_command({"mode": "last"}), packet(), policy=CONTRACT_POLICY)
    assert usage[MAIN_EDIT_BUDGET_POOL]["edit_cost"] == pytest.approx(0.08 * 1.25)
    # Without an enabling policy the raw request is charged, so validation fails closed.
    assert command_budget_usage(readout_command(), packet())[MAIN_EDIT_BUDGET_POOL]["alpha"] == 3.0


def test_contract_edits_never_enter_the_absolute_loop_rescue_pool():
    small = readout_command(op={"kind": "resid_add", "alpha": 0.05})
    edit = parse_controller_command(small).edits[0]
    parsed = parse_observation_packet(packet())
    assert classify_edit_budget_pool(edit=edit, packet=parsed, surface=parsed.surface_catalog[0]) == "main"
    plain = parse_controller_command(readout_command(op={"kind": "resid_add", "alpha": 0.05}, meta={},
                                                     budget={"ttl_steps": 1, "step_size": 0.05,
                                                             "revertible": True})).edits[0]
    assert classify_edit_budget_pool(edit=plain, packet=parsed, surface=parsed.surface_catalog[0]) == "loop_rescue"


def test_relative_write_cap_reads_only_explicit_declarations():
    assert relative_write_cap(SimpleNamespace(meta={})) is None
    assert relative_write_cap(SimpleNamespace(meta={"write_contract": NORM_RELATIVE_WRITE_CONTRACT,
                                                    "relative_write_cap": 0.08})) == 0.08
    with pytest.raises(ValueError):
        relative_write_cap(SimpleNamespace(meta={"relative_write_cap": 0.08}))


def test_guard_clips_each_written_position_to_its_own_norm():
    before = torch.tensor([[[3.0, 4.0], [0.0, 10.0]]])
    after = before.clone()
    after[0, 1] += torch.tensor([6.0, 8.0])
    telemetry = {}
    out = clip_relative_write(before, after, relative_cap=0.08, telemetry=telemetry)
    assert torch.equal(out[0, 0], before[0, 0])
    assert float((out[0, 1] - before[0, 1]).norm()) == pytest.approx(0.8)
    assert torch.allclose((out[0, 1] - before[0, 1]) / 0.8, torch.tensor([0.6, 0.8]))
    assert telemetry["written_positions"] == 1 and telemetry["relative_clip_saturated"] is True
    assert telemetry["site_norm"] == pytest.approx(10.0) and telemetry["raw_write_norm"] == pytest.approx(10.0)
    assert telemetry["write_limit_norm"] == pytest.approx(0.8)
    assert telemetry["relative_write_norm"] == pytest.approx(0.08)


def test_guard_returns_unsaturated_output_object_and_fails_closed_on_nonfinite():
    before = torch.ones(1, 2, 4)
    after = before.clone()
    after[0, -1, 0] += 0.1
    telemetry = {}
    assert clip_relative_write(before, after, relative_cap=0.08, telemetry=telemetry) is after
    assert telemetry["relative_clip_saturated"] is False
    assert telemetry["relative_write_norm"] == pytest.approx(0.05)
    untouched = {}
    clip_relative_write(before, before.clone(), relative_cap=0.08, telemetry=untouched)
    assert untouched == {"written_positions": 0, "relative_clip_saturated": False}
    broken = before.clone()
    broken[0, -1, 0] = float("inf")
    with pytest.raises(RuntimeError, match="non-finite"):
        clip_relative_write(before, broken, relative_cap=0.08)


def test_guard_reports_realized_write_after_half_precision_cast():
    before = torch.full((1, 1, 64), 7.0, dtype=torch.float16)
    after = before + torch.linspace(-3, 3, 64, dtype=torch.float16)
    telemetry = {}
    out = clip_relative_write(before, after, relative_cap=0.08, telemetry=telemetry)
    realized = float((out.float() - before.float()).norm()) / float(before.float().norm())
    assert out.dtype == torch.float16
    assert telemetry["relative_write_norm"] == pytest.approx(realized, abs=1e-6)
    assert realized == pytest.approx(0.08, abs=2e-3)


def test_compiled_contract_resid_add_is_guarded_and_charged_in_relative_units():
    runtime_state = fixtures.FakeRuntimeState()
    runtime_state.model = SimpleNamespace(W_U=torch.eye(3).repeat(1, 3))
    parsed = parse_observation_packet(packet())
    ctx = StepContext(packet=parsed, runtime_state=runtime_state, traces={}, stats={},
                      adapter=fixtures.FakeAdapter())
    compiled = compile_command(readout_command(), parsed, ctx, policy=CONTRACT_POLICY)
    compiled[0].apply(ctx)
    record = runtime_state.hooks["e_contract"]
    metadata = record["metadata"]
    assert metadata["alpha"] == 0.08 and metadata["op_alpha"] == 3.0 and metadata["step_size"] is None
    assert metadata["edit_cost"] == pytest.approx(0.08)
    assert metadata["write_contract"] == NORM_RELATIVE_WRITE_CONTRACT
    act = torch.full((1, 4, 3), 10.0)
    out = record["hook_fn"](act, None)
    telemetry = metadata["write_contract_telemetry"]
    assert torch.equal(out[0, :3], act[0, :3])
    assert float((out[0, -1] - act[0, -1]).norm()) == pytest.approx(0.08 * float(act[0, -1].norm()), rel=1e-5)
    assert telemetry["guard_call_count"] == 1 and telemetry["relative_clip_saturated"] is True


def test_contract_step_size_has_no_surface_fallback_but_plain_edits_keep_it():
    payload = {**packet(), "surface_catalog": fixtures.TestWorkerRuntimeAndBaselines()._surface_catalog()}
    surface = parse_observation_packet(payload).surface_map()["s_resid_pre_l3_last"]
    contract = parse_controller_command(readout_command()).edits[0]
    plain = parse_controller_command(readout_command(meta={})).edits[0]
    assert _resolved_step_size(contract, surface) is None
    assert _resolved_step_size(plain, surface) == 0.08


def test_plain_registration_metadata_is_unchanged_by_the_contract_code_path():
    runtime_state = fixtures.FakeRuntimeState()
    parsed = parse_observation_packet(packet())
    ctx = StepContext(packet=parsed, runtime_state=runtime_state, traces={}, stats={},
                      adapter=fixtures.FakeAdapter())
    compile_command(fixtures._activation_patch_command(), parsed, ctx)[0].apply(ctx)
    metadata = runtime_state.hooks["e_activation_patch"]["metadata"]
    assert not {"op_alpha", "write_contract", "relative_write_cap", "write_contract_telemetry"} & set(metadata)
    assert metadata["alpha"] == 0.2
    out = runtime_state.hooks["e_activation_patch"]["hook_fn"](torch.full((1, 4, 3), 10.0), None)
    assert out.shape == (1, 4, 3)
    assert "guard_call_count" not in metadata["activation_patch_telemetry"]


def test_protected_lexemes_cover_structural_words_inside_required_terms():
    assert structural_contract.protected_lexemes(["before dusk", "Nora", "sample"]) == {"before"}
    assert structural_contract.protected_lexemes(["Nora", "report"]) == frozenset()


def contract_worker():
    worker = fixtures.TestWorkerRuntimeAndBaselines()._make_worker_runtime()
    worker.reset("pacab")
    worker._append_output_token(1)
    worker.runtime_state.model = SimpleNamespace(W_U=torch.eye(3).repeat(1, 3)[:, :7])
    worker.runtime_state.run_with_cache(worker._current_token_tensor())
    worker._token_position_records = lambda: [
        {"position": 0, "token_id": 0, "piece": "SOURCE:", "provenance_class": "constraint_header",
         "segment_kind": "prompt"},
        {"position": 1, "token_id": 1, "piece": " before", "provenance_class": "source_body", "segment_kind": "prompt"},
        {"position": 2, "token_id": 3, "piece": ",", "provenance_class": "source_body", "segment_kind": "prompt"},
        {"position": 3, "token_id": 1, "piece": " Nora", "provenance_class": "source_body", "segment_kind": "prompt"},
        {"position": 4, "token_id": 2, "piece": " while", "provenance_class": "source_body", "segment_kind": "prompt"},
    ]

    def read_ref(ref, ctx):
        tensor = ctx.runtime_state.last_cache[f"{ref['tensor']}:{ref['layer']}"][0]
        token = ref["token"]
        return (tensor[-1] if token["mode"] == "last" else tensor[token["value"]]).detach().clone()

    worker.adapter.read_ref = read_ref
    return worker


def test_offered_recipes_execute_under_the_contract_and_report_bounded_writes():
    worker = contract_worker()
    terms = ("before dusk", "Nora")
    sources = structural_contract.eligible_sources(worker, terms)
    assert [s["piece"] for s in sources] == [",", " while"]
    policy = structural_contract.policy_for(worker, terms)
    assert policy.write_contract.readout_token_ids == (3, 2)
    assert policy.write_contract.source_positions == (2, 4)
    where = structural_contract.surface(worker, 11)
    recipes = (structural_contract.patch_recipes(worker, sources, where)
               + structural_contract.readout_recipes(worker, sources, where))
    assert [r["component"] for r in recipes] == ["common", "common_parallel", "common_orthogonal",
                                                 "readout", "readout"]
    assert len({r["recipe_id"] for r in recipes}) == len(recipes)
    for recipe in recipes:
        command = structural_contract.command(recipe)
        assert "step_size" not in command["edits"][0]["budget"]
        assert worker._simulate_decode(max_new_tokens=1, top_k=3, command=command, score_candidate_text=False,
                                       score_observer_check=False) is None
        assert worker._last_simulate_decode_error.startswith("PolicyViolation")
        simulated = worker._simulate_decode(max_new_tokens=1, top_k=3, command=command, policy_override=policy,
                                            score_candidate_text=False, score_observer_check=False)
        assert simulated is not None, worker._last_simulate_decode_error
        (row,) = simulated["edit_runtime_telemetry"]
        assert row["guard_call_count"] == 1 and row["written_positions"] == 1
        assert row["relative_write_norm"] <= 0.08 + 1e-6
        assert row["relative_write_norm"] == pytest.approx(min(recipe["requested_relative_write"], 0.08), rel=1e-4)
        assert not worker._collect_active_edits()


def test_source_positions_must_still_hold_their_token():
    worker = contract_worker()
    records = worker._token_position_records()
    records[2]["token_id"] = 4
    worker._token_position_records = lambda: records
    with pytest.raises(ValueError, match="token_mismatch"):
        structural_contract.eligible_sources(worker, ("before dusk",))
