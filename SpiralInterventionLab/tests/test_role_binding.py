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


def test_theme_and_recipient_are_not_borrowed_from_a_later_clause():
    report = check_role_frames(
        "Nora will take a break before Selim sends the sample to Ivo and the report to Yara.", REVIEW)

    assert not report["role_binding_ok"]
    assert report["frames"][0]["status"] == "partially_bound"
    assert report["frames"][0]["theme_in_action_clause"] is False
    assert report["frames"][1]["status"] == "bound"


def test_recipient_after_to_does_not_close_the_clause():
    report = check_role_frames("Nora will take the sample to Selim, who will send the report to Yara.", REVIEW)

    assert report["role_violations"] == ["sample: recipient selim, expected ivo"]


def test_missing_action_is_unbound_not_a_violation():
    report = check_role_frames("Nora will take the sample to Ivo before dusk.", REVIEW)

    assert report["role_violations"] == []
    assert report["frames"][1]["status"] == "action_missing"
    assert not report["role_binding_ok"]


def test_double_object_dative_binds_the_indirect_object():
    report = check_role_frames(
        "Nora will take the sample to Ivo before dusk, while Selim sends Yara the report.", REVIEW)

    assert report["role_binding_ok"], report
    row = report["frames"][1]
    assert row["bound_recipient"] == "yara" and row["bound_recipient_mode"] == "double_object"


def test_double_object_dative_still_catches_a_wrong_recipient():
    report = check_role_frames(
        "Nora will take the sample to Yara, while Selim sends Ivo the report.", REVIEW)

    assert report["role_violations"] == ["sample: recipient yara, expected ivo",
                                       "report: recipient ivo, expected yara"]


def test_another_frames_agent_after_the_action_still_closes_the_clause():
    # "sends Nora the report" is ambiguous; the check stays conservative rather
    # than binding an agent name into the dative slot.
    report = check_role_frames(
        "Nora will take the sample to Ivo, while Selim sends Nora the report.", REVIEW)

    row = report["frames"][1]
    assert row["bound_recipient"] is None and row["theme_in_action_clause"] is False


def test_a_possessive_between_action_and_theme_is_not_a_dative_recipient():
    report = check_role_frames(
        "Nora will take the sample to Ivo, while Selim sends Nora's report to Yara.", REVIEW)

    row = report["frames"][1]
    assert row["bound_recipient"] == "yara" and row["bound_recipient_mode"] == "to_dative"


def test_to_dative_is_preferred_over_a_pre_theme_name():
    # "sends Ivo the report to Yara" is odd English, but the explicit to-form wins.
    report = check_role_frames(
        "Nora will take the sample to Ivo, while Selim sends Ivo the report to Yara.", REVIEW)

    row = report["frames"][1]
    assert row["bound_recipient"] == "yara" and row["bound_recipient_mode"] == "to_dative"
    assert report["role_violations"] == []


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
