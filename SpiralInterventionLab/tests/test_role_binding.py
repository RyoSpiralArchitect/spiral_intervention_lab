from __future__ import annotations

import pytest

from SpiralInterventionLab.examples.digit_transform_e2e import create_task_env
from SpiralInterventionLab.tasks.role_binding import action_forms, check_role_frames

REVIEW = (("Nora", "take", "sample", "Ivo"), ("Selim", "send", "report", "Yara"))
PREFIX = "Nora will take the sample to Ivo"


@pytest.mark.parametrize("output", [
    f"{PREFIX} before Selim sends the report to Yara.",
    f"{PREFIX} Selim will send the report to Yara before dusk.",
    f"{PREFIX}, and Selim sent the report to Yara before dusk.",
])
def test_role_correct_rewrites_bind_both_frames(output):
    report = check_role_frames(output, REVIEW)

    assert report["role_binding_ok"], report
    assert [row["status"] for row in report["frames"]] == ["bound", "bound"]


@pytest.mark.parametrize("output,expected", [
    # Recorded natural output: Nora, not Selim, sends the report.
    (f"{PREFIX} before sending the report to Yara before dusk.", ["send: agent nora, expected selim"]),
    # Forced at token 10: the report becomes Selim's, the sender becomes Nora.
    (f"{PREFIX} before sending Selim's report to Yara before dusk.", ["send: agent nora, expected selim"]),
    # Forced at token 6: the sample goes to Selim instead of Ivo.
    ("Nora will take the sample to Selim before dusk, who will send the report to Yara.",
     ["sample: recipient selim, expected ivo"]),
    # Forced at token 13: Nora sends the report to Selim.
    (f"{PREFIX} before sending the report to Selim before dusk.",
     ["send: agent nora, expected selim", "report: recipient selim, expected yara"]),
])
def test_role_violations_from_recorded_and_forced_outputs(output, expected):
    report = check_role_frames(output, REVIEW)

    assert not report["role_binding_ok"]
    assert report["role_violations"] == expected


def test_missing_action_is_unbound_not_a_violation():
    report = check_role_frames("Nora will take the sample to Ivo before dusk.", REVIEW)

    assert report["role_violations"] == []
    assert report["frames"][1]["status"] == "action_missing"
    assert not report["role_binding_ok"]


def test_action_forms_cover_inflections_and_irregular_pasts():
    assert {"take", "takes", "taking", "took", "taken"} <= action_forms("take")
    assert {"send", "sends", "sending", "sent"} <= action_forms("send")
    assert "brought" in action_forms("Bring")


def test_ladder_env_reports_roles_without_changing_score_or_feedback():
    task = create_task_env("constrained_rewrite_l1")
    task.reset(0)
    natural = " Nora will take the sample to Ivo before sending the report to Yara before dusk."

    report = task.role_binding_report(natural)

    assert report["case_id"] == "review"
    assert report["role_violations"] == ["send: agent nora, expected selim"]
    assert task.score(natural) == 0.938889
    assert "role_binding_ok" not in task.task_feedback(natural)
