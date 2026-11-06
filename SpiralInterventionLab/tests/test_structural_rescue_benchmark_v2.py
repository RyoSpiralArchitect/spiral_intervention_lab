import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch
import time

import pytest

from SpiralInterventionLab.examples import structural_rescue_benchmark_v2 as v2
from SpiralInterventionLab.examples.structural_rescue_benchmark import ForwardLimit, Meter
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
    assert v2.fixed_ids([]) == {"fixed": None, "fixed_readout": None}


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
