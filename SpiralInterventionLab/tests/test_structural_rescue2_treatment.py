import hashlib
import json

import pytest

from SpiralInterventionLab.examples import structural_rescue2_treatment as treat
from SpiralInterventionLab.runtime import reach_datasheet


def write_sweep(tmp_path, *, r_star=0.08, mismatches=()):
    digests = {}
    for cap in (0.08, 0.10):
        sheet = reach_datasheet.build([], cap=cap, provenance={"t": 1})
        (tmp_path / f"datasheet_cap_{cap:.2f}.json").write_text(json.dumps(sheet, indent=2, sort_keys=True) + "\n")
        digests[f"{cap:.2f}"] = sheet["datasheet_sha256"]
    analysis = {"reproduction": {"mismatches": list(mismatches)}, "missing_pool_indices": [],
                "invariants": {"gap_mismatch": []}, "r_star": {"r_star": r_star, "reason": "x"},
                "datasheets": digests}
    (tmp_path / "analysis_sweep.json").write_text(json.dumps(analysis))


def test_treatment_binds_r_star_both_datasheets_and_every_prompt(tmp_path):
    write_sweep(tmp_path, r_star=0.10)
    frozen = treat.treatment(tmp_path)
    assert frozen["r_star"] == 0.10 and frozen["table"]["caps"] == [0.08, 0.10]
    assert frozen["datasheets"]["control"]["cap"] == 0.08
    assert frozen["sweep_analysis_sha256"] == hashlib.sha256((tmp_path / "analysis_sweep.json").read_bytes()).hexdigest()
    assert set(frozen["system_prompt_sha256"]) == {"controller", "controller_mechanics", "controller_datasheet"}


def test_treatment_refuses_an_unclean_sweep(tmp_path):
    write_sweep(tmp_path, mismatches=[["c", "push"]])
    with pytest.raises(ValueError, match="clean"):
        treat.treatment(tmp_path)
