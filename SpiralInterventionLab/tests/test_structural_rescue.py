import json

import pytest

from SpiralInterventionLab.tasks.structural_rescue import CASES, StructuralRescueEnv, manifest_bytes


def test_manifest_covers_success_controls_and_unfiltered_confirmation_variants():
    manifest = json.loads(manifest_bytes())
    assert len(manifest["cases"]) == 9
    assert [c["split"] for c in manifest["cases"]].count("confirmation") == 6
    assert manifest_bytes() == manifest_bytes()
    for case in CASES:
        task = StructuralRescueEnv(case)
        prompt = task.reset(0)
        assert prompt == task.reset(99)
        assert "frames_evaluation_only" not in prompt
        assert "joint_success" not in task.task_feedback("")


@pytest.mark.parametrize("text,status", [
    ("Nora takes the sample to Ivo, while Selim sends the report to Yara before dusk.", "pass"),
    ("Selim takes the sample to Ivo, while Nora sends the report to Yara before dusk.", "fail"),
    ("Nora takes the sample to Ivo before sending Selim's report to Yara before dusk.", "fail"),
    ("Nora takes the sample to Ivo, while Selim sends the report to Ivo before dusk.", "fail"),
    ("Nora takes the sample to Ivo, while Selim sends the report before dusk.", "unknown"),
    ("Nora takes the sample to Ivo, while Selim sends the report to Yara and sends before dusk.", "unknown"),
    ("Nora takes the sample to Ivo, who asks Selim to send the report to Yara before dusk.", "unknown"),
])
def test_role_aware_evaluation_retains_unknown_and_rejects_role_swaps(text, status):
    task = StructuralRescueEnv(CASES[1])
    task.reset(0)
    report = task.evaluate(text)
    assert report["role_status"] == status
    assert report["joint_success"] == (status == "pass")


def test_lexical_full_score_does_not_promote_a_role_swap():
    task = StructuralRescueEnv(CASES[1])
    task.reset(0)
    row = task.evaluate("Selim takes sample to Ivo, while Nora sends report to Yara before dusk.")
    assert row["legacy_score"] == 1.0
    assert not row["joint_success"]


def test_trailing_unfinished_text_is_not_a_single_completed_sentence():
    task = StructuralRescueEnv(CASES[1])
    task.reset(0)
    output = "Nora takes sample to Ivo, while Selim sends report to Yara before dusk. More"
    row = task.evaluate(output)
    assert row["legacy_score"] == 1.0
    assert not row["single_sentence_ok"] and not row["joint_success"]
