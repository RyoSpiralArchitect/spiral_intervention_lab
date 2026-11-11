import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from SpiralInterventionLab.examples import structural_rescue_benchmark_v2 as v2
from SpiralInterventionLab.examples import structural_rescue_screen as screen
from SpiralInterventionLab.examples import structural_rescue_screen_v2 as s2
from SpiralInterventionLab.examples import structural_rescue_screen_v3 as s3
from SpiralInterventionLab.examples import structural_rescue_screen_v4 as s4


def test_fresh_screen_is_disjoint_bounded_and_keeps_prior_protocols():
    checkpoint = {"config.json": "hash"}
    frozen = s4.manifest(checkpoint)
    assert frozen["count"] == 30 and frozen["disk_floor_gib"] == 10
    assert frozen["forward_caps"] == {"setup": 40, "diagnostic": 24, "continuation": 64}
    assert frozen["checkpoint_sha256"] == checkpoint and frozen["run"]["mps_memory_fraction"] == 0.7
    prior = {row["prompt_sha256"] for source in (screen, s2, s3) for row in source.manifest()["candidates"]}
    prior.update(row["prompt_sha256"] for row in v2.heldout_manifest(v2.heldout_cases(), checkpoint)["cases"])
    current = {row["prompt_sha256"] for row in frozen["candidates"]}
    assert len(current) == 30 and not current & prior
    assert s4.manifest_bytes(checkpoint) == s4.manifest_bytes(checkpoint)
    assert hashlib.sha256(v2.manifest_bytes()).hexdigest().startswith("9df274ca19caf621")


@pytest.mark.parametrize("tamper", ["none", "incomplete", "selection", "rows"])
def test_fresh_screen_receipt_is_complete_bound_and_not_cherry_picked(tmp_path, tamper):
    checkpoint = {"config.json": "hash"}
    frozen = s4.manifest_bytes(checkpoint)
    (tmp_path / "manifest.json").write_bytes(frozen)
    rows = [{"case_id": row["case_id"], "boundary_found": True, "joint_success": i != 0}
            for i, row in enumerate(s4.manifest(checkpoint)["candidates"])]
    report = {"status": "complete", "manifest_sha256": hashlib.sha256(frozen).hexdigest(),
              "checkpoint_sha256_verified": checkpoint, "rows": rows, "selected_case_ids": ["gen_000"]}
    if tamper == "incomplete":
        report["status"] = "running"
    if tamper == "selection":
        report["selected_case_ids"] = ["gen_001"]
    if tamper == "rows":
        report["rows"] = rows[:-1]
    (tmp_path / "results.json").write_text(json.dumps(report))
    if tamper != "none":
        with pytest.raises(ValueError):
            s4.selected_cases(tmp_path)
    else:
        cases, binding = screen.load_selected_cases(tmp_path)
        assert cases[0].case_id == "gen_000" and cases[0].split == "rescue_v4"
        assert binding["selected_case_ids"] == ["gen_000"]


def test_fresh_screen_does_not_restore_unused_full_prefix_cache():
    state = SimpleNamespace(last_cache={"large": object()}, last_logits=object())
    worker = SimpleNamespace(
        runtime_state=state, done=lambda: False, _output_token_ids=lambda: [1],
        step=Mock(), final_text=lambda: " A sends parcel to B",
        _current_token_tensor=lambda: SimpleNamespace(shape=(1, 8)),
        codec=SimpleNamespace(decode=lambda ids: "Observed continuation."))

    def simulate(**kwargs):
        assert state.last_cache is None and state.last_logits is None
        assert kwargs.get("command") is None
        return {"continuation_token_ids": [2]}

    worker._simulate_decode = Mock(side_effect=simulate)
    task = SimpleNamespace(evaluate=lambda text: {"output": text})
    case = SimpleNamespace(case_id="case", frames=[("A", "send", "parcel", "B")])
    meter = SimpleNamespace()
    with patch.object(screen, "_release_memory"):
        row = screen._decode_case(worker, task, case, meter=meter)
    assert row["boundary_found"] and row["output"] == "Observed continuation."
    worker._simulate_decode.assert_called_once()
