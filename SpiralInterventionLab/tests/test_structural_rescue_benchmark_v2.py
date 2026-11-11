import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch
import time

import pytest

from SpiralInterventionLab.examples import structural_rescue_benchmark_v2 as v2
from SpiralInterventionLab.examples.structural_rescue_benchmark import ForwardLimit, Meter
from SpiralInterventionLab.examples import (
    structural_rescue_screen, structural_rescue_screen_v2, structural_rescue_screen_v3,
)
from SpiralInterventionLab.runtime import structural_contract
from SpiralInterventionLab.tasks import structural_rescue
from SpiralInterventionLab.tests.test_structural_contract import contract_worker


TERMS = ("before dusk", "Nora")


def toy_catalog(worker):
    sources = structural_contract.eligible_sources(worker, TERMS)
    where = structural_contract.surface(worker, 11)
    recipes = (structural_contract.patch_recipes(worker, sources, where)
               + structural_contract.readout_recipes(worker, sources, where))
    return recipes, structural_contract.policy_for(worker, TERMS)


def test_v1_manifest_is_unchanged_and_v2_manifest_is_separate_and_stable():
    v1_digest = hashlib.sha256(structural_rescue.manifest_bytes()).hexdigest()
    assert v1_digest.startswith("81326474db5ae819")
    assert v2.manifest_bytes() == v2.manifest_bytes()
    frozen = v2.manifest()
    assert frozen["version"] != structural_rescue.benchmark_manifest()["version"]
    assert frozen["cases"] == structural_rescue.benchmark_manifest()["cases"]
    assert frozen["contract"]["relative_write_cap"] == 0.08
    assert frozen["controller_system_prompt_sha256"] == hashlib.sha256(v2.SYSTEM_PROMPT.encode()).hexdigest()


def test_contract_pilot_records_its_disk_floor_deviation_and_meter_enforces_it(tmp_path):
    assert v2.manifest()["disk_floor_gib"] == 6 and v2.manifest()["protocol_deviations"]
    worker = SimpleNamespace(runtime_state=SimpleNamespace(), _collect_active_edits=lambda: [])
    meter = Meter(worker, deadline=time.monotonic() + 10, disk_path=tmp_path, disk_floor_gib=6)
    with patch("shutil.disk_usage", return_value=SimpleNamespace(free=7 * 2**30)):
        meter.check_limits()
    with patch("shutil.disk_usage", return_value=SimpleNamespace(free=5 * 2**30)):
        with pytest.raises(ForwardLimit, match="6_gib"):
            meter.check_limits()
    v1_meter = Meter(worker, deadline=time.monotonic() + 10, disk_path=tmp_path)
    with patch("shutil.disk_usage", return_value=SimpleNamespace(free=7 * 2**30)):
        with pytest.raises(ForwardLimit, match="10_gib"):
            v1_meter.check_limits()


def test_packet_offers_ids_and_geometry_but_no_expressions_tensors_or_answers():
    worker = contract_worker()
    recipes, _policy = toy_catalog(worker)
    packet = v2.compact_packet(worker, recipes, "case", {"structure_readout_rows": []})
    assert [r["recipe_id"] for r in packet["recipes"]] == [r["recipe_id"] for r in recipes]
    for row in packet["recipes"]:
        assert not {"source_expression", "input_identity", "alpha", "site_norm"} & set(row)
        assert ("geometry" in row) == (row["kind"] == "patch")
    assert "frames" not in json.dumps(packet) and packet["production_apply_allowed"] is False


def test_fixed_rules_pick_first_pair_orthogonal_and_first_source_readout():
    worker = contract_worker()
    recipes, _policy = toy_catalog(worker)
    fixed = v2.fixed_ids(recipes)
    patch_recipe = next(r for r in recipes if r["recipe_id"] == fixed["fixed"])
    push = next(r for r in recipes if r["recipe_id"] == fixed["fixed_readout"])
    assert patch_recipe["component"] == "common_orthogonal"
    assert [s["piece"] for s in patch_recipe["sources"]] == [",", " while"]
    assert push["kind"] == "readout" and push["sources"][0]["piece"] == ","
    assert v2.fixed_ids([]) == {"fixed": None, "fixed_readout": None, "fixed_clause_form": None,
                                "fixed_verb_slot": None}


def test_clause_form_gate_is_recorded_and_matches_only_the_participial_template():
    assert v2.manifest(clause_form=True)["clause_form_applicability_rule"] == \
        v2.CLAUSE_FORM_APPLICABILITY_RULE
    assert "clause_form_applicability_rule" not in v2.manifest()
    participial = ("In order to start before lunch, Olga should bring the drawing to Nikos, "
                   "with Stefan giving the voucher to Ilse.")
    relative = ("Before lunch, Stefan should bring the manual to Pablo; meanwhile, Celia, "
                "who should give the drawing to Tilda, also finishes.")
    assert v2.clause_form_applies(participial) and not v2.clause_form_applies(relative)
    assert v2.manifest_bytes(clause_form=True) == v2.manifest_bytes(clause_form=True)


def test_verb_slot_plan_locates_the_substituted_verb_and_inflected_target():
    import torch
    pieces = {1: " before", 2: " Boris", 3: " finishes", 4: " roster"}
    codec = SimpleNamespace(decode=lambda ids: pieces.get(ids[0], " ?"),
                            encode=lambda s: torch.tensor([5097]))
    worker = SimpleNamespace(codec=codec, _simulate_decode=lambda **kw: {
        "continuation_token_ids": [1, 2, 3, 4]})
    task = SimpleNamespace(case=SimpleNamespace(
        frames=[("Pablo", "send", "receipt", "Nikos"), ("Boris", "take", "roster", "Edda")]))
    plan = v2.verb_slot_plan(worker, task)
    assert plan["apply_at_step"] == 2 and plan["pre_roll_token_ids"] == [1, 2]
    assert plan["substituted_piece"] == " finishes" and plan["target_piece"] == " takes"
    worker._simulate_decode = lambda **kw: {"continuation_token_ids": [1, 2, 4]}
    assert v2.verb_slot_plan(worker, task) is None


def test_b0_preview_reuses_lookahead_even_without_a_verb_slot_and_excludes_evaluation():
    worker = SimpleNamespace(codec=SimpleNamespace(decode=lambda ids: " observed continuation"),
                             _simulate_decode=Mock(return_value={"continuation_token_ids": [1, 2],
                                                                "joint_success": True, "role_report": {}}))
    task = SimpleNamespace(case=SimpleNamespace(frames=[(), ("Boris", "take")]))
    preview = {}
    assert v2.verb_slot_plan(worker, task, max_new_tokens=2, preview=preview) is None
    worker._simulate_decode.assert_called_once_with(max_new_tokens=2, top_k=6,
        score_candidate_text=False, score_observer_check=False)
    assert preview == {"continuation": " observed continuation", "observed_tokens": 2,
                       "max_tokens": 2, "reached_token_limit": True}


def test_b0_preview_and_plain_selectors_have_same_prompt_and_offers():
    provider = SimpleNamespace(provider_name="fake", model_name="fake", complete=Mock(return_value=
        SimpleNamespace(text='{"decision":"noop"}', usage={}, metadata={})))
    plain = {"case_id": "case", "recipes": [{"recipe_id": "r1", "kind": "readout"}]}
    preview = {**plain, "b0_preview": {"continuation": " observed continuation"}}
    for arm, packet in (("controller", plain), (v2.B0_PREVIEW_POLICY, preview)):
        chosen, trace = v2.select(arm, packet, provider=provider, fixed={}, b0_preview_study=True)
        assert chosen is None and trace["status"] == "noop" and trace["provider_calls"] == 1
    first, second = (call.args[0] for call in provider.complete.call_args_list)
    assert first.system_prompt == second.system_prompt
    assert first.payload == {k: v for k, v in second.payload.items() if k != "b0_preview"}
    assert first.max_output_tokens == second.max_output_tokens == 1600
    with pytest.raises(ValueError, match="requires_predeclared"):
        v2.select(v2.B0_PREVIEW_POLICY, preview, provider=provider, fixed={})


def test_preview_study_freezes_screen_selection_and_checkpoint(tmp_path):
    screen = tmp_path / "screen"
    screen.mkdir()
    raw = structural_rescue_screen_v3.manifest_bytes()
    (screen / "manifest.json").write_bytes(raw)
    rows = [{"case_id": case["case_id"], "boundary_found": True, "joint_success": i != 0}
            for i, case in enumerate(json.loads(raw)["candidates"])]
    checkpoint = {"config.json": "fake"}
    report = {"status": "complete", "manifest_sha256": hashlib.sha256(raw).hexdigest(),
              "selected_case_ids": ["gen_000"], "rows": rows, "checkpoint_sha256_verified": checkpoint}
    receipt = screen / "results.json"
    receipt.write_text(json.dumps(report))
    output = tmp_path / "preview"
    args = ["--prepare", "--compare-b0-preview", "--screen-dir", str(screen),
            "--checkpoint-manifest", str(receipt)]
    assert v2.main([*args, "--output-dir", str(output)]) == 0
    frozen = json.loads((output / "manifest.json").read_bytes())
    assert frozen["policies"] == [*v2.HOLDOUT_POLICIES, v2.B0_PREVIEW_POLICY]
    assert frozen["b0_preview"]["additional_worker_forwards"] == 0
    assert frozen["checkpoint_sha256"] == checkpoint and frozen["max_controller_calls_per_case"] == 2
    assert [case["case_id"] for case in frozen["cases"]] == ["gen_000"]
    assert frozen["controller_system_prompt_sha256"] == hashlib.sha256(
        v2.system_prompt(True, v2.HOLDOUT_CAP, True, True, b0_preview_study=True).encode()).hexdigest()
    report["selected_case_ids"] = ["gen_001"]
    receipt.write_text(json.dumps(report))
    with pytest.raises(SystemExit):
        v2.main([*args, "--output-dir", str(tmp_path / "tampered")])


def test_verb_slot_manifest_records_the_rule_and_its_own_policy():
    vs = v2.manifest(verb_slot=True)
    assert vs["verb_slot_rule"] == v2.VERB_SLOT_RULE and "fixed_verb_slot" in vs["policies"]
    assert vs["version"].endswith("_clauseform_verbslot")
    assert "verb_slot_rule" not in v2.manifest(clause_form=True)


def test_heldout_freeze_uses_new_cases_without_filtering_and_binds_checkpoint(tmp_path):
    checkpoint = tmp_path / "prior_receipt.json"
    checkpoint.write_text(json.dumps({"checkpoint_sha256_verified": {"config.json": "hash"}}))
    output = tmp_path / "heldout"
    args = ["--heldout", "--output-dir", str(output), "--checkpoint-manifest", str(checkpoint)]
    assert v2.main(["--prepare", *args]) == 0
    frozen = json.loads((output / "manifest.json").read_bytes())
    assert len(frozen["cases"]) == 10
    prior = {c["prompt_sha256"] for screen in (structural_rescue_screen,
             structural_rescue_screen_v2, structural_rescue_screen_v3)
             for c in screen.manifest()["candidates"]}
    heldout = {c["prompt_sha256"] for c in frozen["cases"]}
    assert len(heldout) == 10 and not heldout & prior
    assert frozen["case_set"]["selection_rule"] == "all_generated_in_order_no_b0_screen"
    assert frozen["policies"] == ["b0", "fixed_routed", "controller"]
    assert frozen["checkpoint_sha256"] == {"config.json": "hash"}
    assert frozen["contract"]["relative_write_cap"] == 0.12
    assert frozen["setup_forward_cap_per_policy"] == 40 and frozen["disk_floor_gib"] == 10
    with pytest.raises(SystemExit):
        v2.main(["--prepare", *args, "--case-ids", "heldout_000"])
    with pytest.raises(SystemExit):
        v2.main(["--prepare", *args, "--device", "cpu"])


def test_fixed_routed_uses_only_applicable_offers_and_never_calls_provider():
    provider = Mock()
    rows = [{"recipe_id": "readout", "kind": "readout"},
            {"recipe_id": "starter", "kind": "clause_form", "applies_to_this_case": True},
            {"recipe_id": "verb", "kind": "verb_slot", "applies_to_this_case": True}]
    packet = {"recipes": rows}
    for expected in ("verb", "starter", "readout", None):
        chosen, trace = v2.select("fixed_routed", packet, provider=provider, fixed={})
        assert chosen == expected and trace["provider_calls"] == 0
        if rows:
            rows.pop()
    provider.complete.assert_not_called()
    packet["recipes"] = [{"recipe_id": "starter", "kind": "clause_form", "applies_to_this_case": False}]
    assert v2.select("fixed_routed", packet, provider=provider, fixed={})[0] is None


def test_heldout_pair_counts_exclude_provider_failure_but_keep_unavailable_boundary():
    passed = {"status": "ok", "joint_success": True}
    failed = {"status": "ok", "joint_success": False}
    rows = [
        {"comparison_eligible": False, "policies": {"fixed_routed": passed, "controller": passed}},
        {"policies": {"fixed_routed": failed, "controller": passed}},
        {"policies": {"fixed_routed": passed, "controller": failed}},
        {"policies": {"fixed_routed": passed, "controller": {**passed, "selection": {"status": "error"}}}},
    ]
    assert v2.paired_heldout(rows) == dict(expected=10, paired=3, both_pass=1, both_fail=0,
                                          controller_only=1, fixed_routed_only=1)


def test_prepare_unembed_storage_avoids_per_forward_copy_without_changing_linear():
    import torch

    unembed = torch.nn.Module()
    unembed.W_U = torch.nn.Parameter(
        (torch.arange(24, device="cpu") / 7).reshape(3, 8).half(), requires_grad=False)
    model = SimpleNamespace(unembed=unembed)
    before = unembed.W_U.detach().clone()
    hidden = torch.tensor([[0.1, -0.3, 0.4]], dtype=before.dtype, device="cpu")
    expected = torch.nn.functional.linear(hidden, before.T.contiguous())
    v2.prepare_unembed_storage(model)
    assert torch.equal(unembed.W_U, before)
    assert not unembed.W_U.requires_grad and unembed.W_U.T.is_contiguous()
    linear_weight = unembed.W_U.T.contiguous()
    assert linear_weight.data_ptr() == unembed.W_U.data_ptr()
    assert torch.equal(torch.nn.functional.linear(hidden, linear_weight), expected)
    laid_out = unembed.W_U
    v2.prepare_unembed_storage(model)
    assert unembed.W_U is laid_out


def test_datasheet_manifest_is_a_separate_frozen_version():
    base, sheet = v2.manifest(), v2.manifest(datasheet=True)
    assert sheet["version"] == base["version"] + "_datasheet"
    assert {k: v for k, v in sheet.items() if k not in {"version", "controller_datasheet",
                                                      "controller_system_prompt_sha256"}} == {
        k: v for k, v in base.items() if k not in {"version", "controller_system_prompt_sha256"}}
    slopes = sheet["controller_datasheet"]["readout_closure_slope"]
    assert set(slopes) == set(structural_contract.READOUT_CLOSURE_SLOPE)
    assert (sheet["controller_system_prompt_sha256"]
            == hashlib.sha256(v2.system_prompt(True).encode()).hexdigest()
            != base["controller_system_prompt_sha256"])
    assert v2.manifest_bytes(datasheet=True) == v2.manifest_bytes(datasheet=True)


def test_datasheet_packet_adds_reach_only_to_measured_readout_recipes():
    worker = contract_worker()
    recipes, _policy = toy_catalog(worker)
    plain = v2.compact_packet(worker, recipes, "case", {"structure_readout_rows": []})
    assert all("reach_estimate" not in r for r in plain["recipes"])
    packet = v2.compact_packet(worker, recipes, "case", {"structure_readout_rows": []},
                               datasheet=True)
    by_id = {r["recipe_id"]: r for r in recipes}
    for row in packet["recipes"]:
        recipe = by_id[row["recipe_id"]]
        estimate = row.get("reach_estimate")
        slope = structural_contract.READOUT_CLOSURE_SLOPE.get(recipe["sources"][0]["piece"])
        if recipe["kind"] != "readout" or slope is None:
            assert estimate is None
            continue
        assert estimate["slope_logits_per_unit_cap"] == slope
        assert estimate["estimated_gap_closure"]["min"] == slope["min"] * 0.08
        assert estimate["basis"] == structural_contract.REACH_BASIS


def test_controller_selection_uses_the_datasheet_prompt_when_enabled():
    packet = {"case_id": "case", "recipes": [{"recipe_id": "r1"}]}
    provider = SimpleNamespace(provider_name="fake", model_name="fake", complete=Mock(return_value=
        SimpleNamespace(text='{"decision":"noop"}', usage={}, metadata={})))
    _chosen, trace = v2.select("controller", packet, provider=provider,
                               fixed={"fixed": None, "fixed_readout": None}, datasheet=True)
    assert "reach_estimate" in v2.system_prompt(True)
    assert trace["system_prompt_sha256"] == hashlib.sha256(v2.system_prompt(True).encode()).hexdigest()


def test_controller_selection_uses_the_contract_prompt():
    packet = {"case_id": "case", "recipes": [{"recipe_id": "r1"}]}
    provider = SimpleNamespace(provider_name="fake", model_name="fake", complete=Mock(return_value=
        SimpleNamespace(text='{"decision":"probe","recipe_id":"r1"}', usage={}, metadata={})))
    chosen, trace = v2.select("controller", packet, provider=provider, fixed={"fixed": None, "fixed_readout": "r1"})
    assert chosen == "r1"
    assert trace["system_prompt_sha256"] == hashlib.sha256(v2.SYSTEM_PROMPT.encode()).hexdigest()
    assert provider.complete.call_args[0][0].system_prompt == v2.SYSTEM_PROMPT
    assert v2.select("fixed_readout", packet, provider=None, fixed={"fixed": None, "fixed_readout": "r1"})[0] == "r1"
    assert v2.select("fixed", packet, provider=None, fixed={"fixed": None, "fixed_readout": "r1"})[0] is None


def run_branch(worker, recipe, policy, tmp_path):
    worker.stop_checker = lambda text: len(text) > 4
    task = SimpleNamespace(evaluate=lambda text: {"output": text})
    meter = Meter(worker, deadline=time.monotonic() + 30, disk_path=tmp_path)
    meter.reset("diagnostic")
    meter.boundary_length = worker._current_token_tensor().shape[-1]
    with patch("shutil.disk_usage", return_value=SimpleNamespace(free=20 * 2**30)):
        with meter.install():
            return v2.branch(worker, task, recipe, meter, policy)


@pytest.mark.parametrize("kind", ["patch", "readout"])
def test_contract_branch_confirms_one_guarded_write_and_restores_prefix(kind, tmp_path):
    worker = contract_worker()
    recipes, policy = toy_catalog(worker)
    recipe = next(r for r in recipes if r["kind"] == kind)
    before = worker.final_text()
    result, raw = run_branch(worker, recipe, policy, tmp_path)
    assert raw is not None and result["state_restored"]
    (row,) = result["runtime_telemetry"]
    assert row["guard_call_count"] == 1 and row["relative_write_norm"] <= 0.08 + 1e-6
    assert "step_size" not in result["command"]["edits"][0]["budget"]
    assert not worker._collect_active_edits() and worker.final_text() == before


def test_contract_branch_fails_closed_without_the_enabling_policy(tmp_path):
    worker = contract_worker()
    recipes, _policy = toy_catalog(worker)
    with pytest.raises(Exception, match="does not enable"):
        run_branch(worker, recipes[0], None, tmp_path)
    assert not worker._collect_active_edits()


@pytest.mark.parametrize("mismatch", [False, True])
def test_delayed_readout_compares_at_the_actual_slot_and_rejects_prefix_mismatch(tmp_path, mismatch):
    import torch
    worker = contract_worker()
    recipe, = structural_contract.verb_slot_recipe(
        worker, structural_contract.surface(worker, 11), token_id=3, piece="c",
        apply_at_step=2, pre_roll_token_ids=[1, 2], substituted_piece="b")
    policy = structural_contract.policy_for(worker, TERMS, extra_readout_token_ids=[3])
    start_length = worker._current_token_tensor().shape[-1]
    original = worker.runtime_state.run_with_cache

    def run(tokens, *args, **kwargs):
        logits, cache = original(tokens, *args, **kwargs)
        winner = 1
        if tokens.shape[-1] == start_length + 2:
            winner = 3 if worker._collect_active_edits() else 2
        logits = torch.zeros_like(logits)
        logits[..., winner] = 5.0
        return logits, cache

    worker.runtime_state.run_with_cache = run
    if mismatch:
        simulate = worker._simulate_decode

        def corrupt_edited_prefix(**kwargs):
            if kwargs["command"]:
                kwargs["pre_roll_token_ids"] = [2, 1]
            return simulate(**kwargs)

        worker._simulate_decode = corrupt_edited_prefix
        with pytest.raises(RuntimeError, match="branch_readout_prefix_mismatch"):
            run_branch(worker, recipe, policy, tmp_path)
    else:
        result, _raw = run_branch(worker, recipe, policy, tmp_path)
        readout = result["readout"]
        assert readout["decode_step"] == 2 and result["token_id_at_comparison"] == 3
        assert readout["prefix_identity"]["shape"][-1] == start_length + 2
        assert readout["fixed_baseline_winner"] == 2 and readout["edited_winner"] == 3
        target, = readout["rows"]
        assert target["token_id"] == 3 and target["baseline_gap_to_fixed_winner"] == 5.0
        assert result["state_restored"] and result["forwards"]["diagnostic"]["completed"] == 3


def test_failed_branch_keeps_attempts_and_selection_metadata():
    meter = SimpleNamespace(counts={"diagnostic": {"attempted": 3, "completed": 2}})
    outcome = {"selection": {"status": "selected", "provider_calls": 1}}
    with patch.object(v2, "branch", side_effect=ForwardLimit("cap")):
        with pytest.raises(ForwardLimit):
            v2.record_branch(None, None, None, meter, outcome, None)
    assert outcome["status"] == "error" and outcome["forwards"] == meter.counts
    assert outcome["selection"]["provider_calls"] == 1


def test_pre_model_failure_is_persisted_without_any_provider_call(tmp_path):
    output = tmp_path / "run"
    assert v2.main(["--prepare", "--output-dir", str(output)]) == 0
    digest = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"checkpoint_sha256": {"config.json": "hash"}}))
    with patch.object(v2, "verify_checkpoint", side_effect=ValueError("checkpoint_mismatch")), \
            patch.object(v2, "create_controller_provider") as provider:
        with pytest.raises(ValueError, match="checkpoint_mismatch"):
            v2.main(["--run", "--output-dir", str(output), "--manifest-sha256", digest,
                     "--worker-model-path", str(tmp_path), "--checkpoint-manifest", str(checkpoint)])
    assert provider.call_count == 0
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == "error" and report["cases"] == []
    with pytest.raises(ValueError, match="manifest"):
        v2.main(["--run", "--output-dir", str(output), "--manifest-sha256", "0" * 64,
                 "--worker-model-path", str(tmp_path), "--checkpoint-manifest", str(checkpoint)])
