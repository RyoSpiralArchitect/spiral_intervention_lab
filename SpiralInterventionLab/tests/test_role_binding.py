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


GIVE = (("Rosa", "send", "blueprint", "Celia"), ("Yusuf", "give", "badge", "Wanda"))


def test_double_object_recipient_binds_only_when_enabled():
    output = "Rosa sends the blueprint to Celia before dawn, while Yusuf gives Wanda the badge."
    v1 = check_role_frames(output, GIVE)
    v2 = check_role_frames(output, GIVE, v2=True)

    assert v1["frames"][1]["status"] == "partially_bound" and "agent_index" not in v1["frames"][1]
    assert v2["role_binding_ok"] and v2["frames"][1]["recipient_form"] == "dative"
    assert v2["frames"][0]["recipient_form"] == "to"


@pytest.mark.parametrize("output,expected", [
    ("Rosa sends the blueprint to Celia, while Yusuf gives Rosa the badge.", ["badge: recipient rosa, expected wanda"]),
    ("Rosa sends the blueprint to Celia, while Yusuf gives Wanda's badge to Celia.",
     ["badge: recipient celia, expected wanda"]),
])
def test_double_object_reading_still_catches_wrong_recipients(output, expected):
    assert check_role_frames(output, GIVE, v2=True)["role_violations"] == expected


def test_a_name_after_the_action_closes_nothing_but_needs_the_theme_after_it():
    # The dative recipient is not a new subject, so the theme is still found.
    agents_and_recipients = (("Rosa", "send", "blueprint", "Yusuf"), ("Yusuf", "give", "badge", "Wanda"))
    report = check_role_frames("Rosa sends Yusuf the blueprint; Yusuf gives Wanda the badge.",
                               agents_and_recipients, v2=True)
    assert report["role_binding_ok"], report
    # Without a following theme the name is not read as a recipient.
    report = check_role_frames("Rosa sends the blueprint to Celia; Yusuf gives Wanda.", GIVE, v2=True)
    assert report["frames"][1]["status"] == "partially_bound"
