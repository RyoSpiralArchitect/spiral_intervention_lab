import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.examples import run_hygiene
from SpiralInterventionLab.examples import structural_first_token_table as table
from SpiralInterventionLab.runtime import structural_contract
from SpiralInterventionLab.tests.test_structural_contract import contract_worker

RECEIPTS = Path(__file__).resolve().parents[2] / "results" / "structural_rescue_contract_20261103"
TERMS = ("before dusk", "Nora")


def test_case_sets_rebuild_the_known_cases_and_the_calibration_pool():
    known, binding = table.case_set("known16", screen1_dir=RECEIPTS / "screen_r2")
    assert len(known) == 16 and binding["screen1_manifest_sha256"].startswith("e018bbcc")
    assert [c.case_id for c in known[9:]] == ["gen_002", "gen_009", "gen_018", "gen_045", "gen_047", "gen_058",
                                              "gen_060"]
    pool, _ = table.case_set("pool62", screen1_dir=RECEIPTS / "screen_r2")
    splits = [c.split for c in pool]
    assert len(pool) == 62 and splits.count("calibration") == 3 and splits.count("screen1_rescue") == 7
    assert not any(c.case_id.startswith("confirm") for c in pool)
    frozen = table.manifest_bytes(pool, binding, caps=list(table.SWEEP_CAPS), patch_caps=[0.12],
                                  patch_scope="orthogonal_plus_spot_checks", packet_caps=[], references={})
    assert frozen == table.manifest_bytes(pool, binding, caps=list(table.SWEEP_CAPS), patch_caps=[0.12],
                                          patch_scope="orthogonal_plus_spot_checks", packet_caps=[], references={})
    assert b'"evaluator_sha256"' in frozen


def test_outcome_cap_prefers_the_contract_cap_then_the_smaller_neighbour():
    assert table.outcome_cap([0.04, 0.08, 0.12]) == 0.08
    assert table.outcome_cap([0.07, 0.09]) == 0.07
    assert table.outcome_cap([0.10, 0.12]) == 0.10


def test_a_prebuilt_packet_skips_the_rebuild_and_changes_nothing():
    worker = contract_worker()
    sources = structural_contract.eligible_sources(worker, TERMS)
    where = structural_contract.surface(worker, 11)
    recipe = structural_contract.readout_recipes(worker, sources, where)[0]
    policy = structural_contract.policy_for(worker, TERMS)
    command = structural_contract.command(recipe)
    packet = worker.build_controller_packet()
    fresh = worker._simulate_decode(max_new_tokens=1, top_k=6, command=command, policy_override=policy,
                                    score_candidate_text=False, score_observer_check=False)
    with patch.object(worker, "build_controller_packet", side_effect=AssertionError("rebuilt")):
        reused = worker._simulate_decode(max_new_tokens=1, top_k=6, command=command, policy_override=policy,
                                         score_candidate_text=False, score_observer_check=False, packet=packet)
    assert reused["continuation_token_ids"] == fresh["continuation_token_ids"]
    assert torch.equal(reused["first_logits"], fresh["first_logits"])
    strip = lambda rows: [{k: v for k, v in r.items() if k != "edit_id"} for r in rows]
    assert strip(reused["edit_runtime_telemetry"]) == strip(fresh["edit_runtime_telemetry"])
    assert packet == worker.build_controller_packet()


def fake_worker(realized):
    telemetry = [{"guard_call_count": 1, "relative_write_norm": realized}]
    meter = SimpleNamespace(phase=None, capture_edited=None, raw_first=None, counts={})

    def simulate(**_):
        # The Meter captures the edited forward's raw logits.
        meter.raw_first = torch.tensor([0.0, 1.0], device="cpu")
        return {"continuation_token_ids": [1], "edit_runtime_telemetry": telemetry}

    worker = SimpleNamespace(_simulate_decode=simulate, _collect_active_edits=lambda: [],
                             _last_simulate_decode_error=None)
    return worker, meter


@pytest.mark.parametrize("kind,realized,ok", [("readout", 0.0800, True), ("readout", 0.0779, False),
                                              ("readout", 0.0821, False), ("patch", 0.05, True),
                                              ("patch", 0.0821, False)])
def test_realized_writes_are_checked_two_sided_for_readouts(kind, realized, ok):
    worker, meter = fake_worker(realized)
    recipe = {"recipe_id": "r", "kind": kind, "component": kind, "sources": [], "input_identity": "same",
              "op": "resid_add" if kind == "readout" else "activation_patch", "alpha": 1.0,
              "surface": {"surface_id": "s"}, "source_expression": {}, "relative_write_cap": 0.08}
    with patch.object(table, "compile_expr", return_value=lambda ctx: None), \
            patch.object(table, "tensor_identity", return_value="same"), \
            patch.object(table.structural_contract, "_context", return_value=None):
        if ok:
            row, raw = table.push(worker, recipe, meter, None, 0.08, {})
            assert row["first_token_id"] == 1 and row["realized_relative_write"] == realized
        else:
            with pytest.raises(RuntimeError, match="tolerance"):
                table.push(worker, recipe, meter, None, 0.08, {})


def test_push_records_rank_gap_and_whether_the_target_won():
    b0 = torch.tensor([3.0, 1.0, 2.5, 0.0], dtype=torch.float64, device="cpu")
    edited = torch.tensor([2.0, 1.0, 2.6, 0.0], dtype=torch.float64, device="cpu")
    row = table.push_record({"first_token_id": 2}, edited, b0, [2, 3])
    assert row["edited_top5"][:2] == [2, 0] and row["emitted_b0_rank"] == 2
    assert row["emitted_b0_gap"] == pytest.approx(0.5) and row["edited_winner_matches_first_token"]


def test_preflight_refuses_low_disk_battery_and_a_modified_tree(tmp_path):
    full = SimpleNamespace(free=20 * 2**30)
    with patch("shutil.disk_usage", return_value=SimpleNamespace(free=10.9 * 2**30)):
        with pytest.raises(RuntimeError, match="free_disk"):
            run_hygiene.preflight(tmp_path, floor_gib=6)
    with patch("shutil.disk_usage", return_value=full), patch.object(run_hygiene, "power_source", return_value="battery"):
        with pytest.raises(RuntimeError, match="ac_power"):
            run_hygiene.preflight(tmp_path, floor_gib=6)
    with patch("shutil.disk_usage", return_value=full), patch.object(run_hygiene, "power_source", return_value="ac"), \
            patch.object(run_hygiene, "_git", return_value=" M file.py"):
        with pytest.raises(RuntimeError, match="modified"):
            run_hygiene.preflight(tmp_path, floor_gib=6)
        run_hygiene.preflight(tmp_path, floor_gib=6, allow_dirty=True)


def test_evaluator_hash_covers_the_evaluator_sources():
    digest = hashlib.sha256()
    for relative in run_hygiene.EVALUATOR_SOURCES:
        digest.update(relative.encode() + b"\0" + (run_hygiene.REPO / relative).read_bytes() + b"\0")
    assert run_hygiene.evaluator_sha256() == digest.hexdigest()
