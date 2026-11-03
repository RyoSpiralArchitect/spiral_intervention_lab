import hashlib
import json

import pytest

from SpiralInterventionLab.examples import structural_rescue_screen_v2 as screen2


def test_screen2_quotas_follow_generation_order_and_ids_are_fresh():
    kept, excluded = screen2.candidates()
    assert len(kept) == 144 and not excluded and kept[0].case_id == "s2_000"
    rows = [{"case_id": f"s2_{i:03d}", "boundary_found": i % 5 != 0, "joint_success": i % 3 != 0}
            for i in range(80)]
    chosen = screen2.select(rows)
    assert len(chosen["rescue"]) == 12 and len(chosen["harm"]) == 24
    assert chosen["rescue"] == sorted(chosen["rescue"]) and not set(chosen["rescue"]) & set(chosen["harm"])
    assert screen2.manifest_bytes({"a": 1}) == screen2.manifest_bytes({"a": 1}) != screen2.manifest_bytes({"a": 2})


def test_screen2_selection_is_rebuilt_and_bound(tmp_path):
    treatment = {"r_star": 0.08}
    (tmp_path / "manifest.json").write_bytes(screen2.manifest_bytes(treatment))
    rows = [{"case_id": f"s2_{i:03d}", "boundary_found": True, "joint_success": i >= 12} for i in range(36)]
    results = {"status": "complete", "manifest_sha256": hashlib.sha256(screen2.manifest_bytes(treatment)).hexdigest(),
               "rows": rows, "selected": screen2.select(rows)}
    (tmp_path / "results.json").write_text(json.dumps(results))
    rescue, harm, binding = screen2.selected_cases(tmp_path, treatment)
    assert [c.split for c in rescue] == ["rescue"] * 12 and [c.split for c in harm] == ["harm"] * 24
    assert binding["rescue"][0] == "s2_000"
    with pytest.raises(ValueError, match="differs"):
        screen2.selected_cases(tmp_path, {"r_star": 0.1})
    results["selected"] = {"rescue": [], "harm": []}
    (tmp_path / "results.json").write_text(json.dumps(results))
    with pytest.raises(ValueError, match="rule"):
        screen2.selected_cases(tmp_path, treatment)
