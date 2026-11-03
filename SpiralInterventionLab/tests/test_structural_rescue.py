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


GEN = StructuralRescueEnv(__import__("SpiralInterventionLab.tasks.structural_rescue", fromlist=["RescueCase"]).RescueCase(
    "gen_x", "rescue", "Before dusk, Boris should send the roster to Tilda, while Edda should give the permit to Alma.",
    (("Boris", "send", "roster", "Tilda"), ("Edda", "give", "permit", "Alma")), "before dusk"))


@pytest.mark.parametrize("text,kind,joint", [
    ("Boris sends the roster to Tilda before dusk, while Edda gives the permit to Alma.", "conjunction", True),
    ("Boris sends the roster to Tilda before dusk; meanwhile, Edda gives the permit to Alma.", "semicolon", True),
    # The source's tasks are simultaneous; a join that orders them changes its meaning.
    ("Boris sends the roster to Tilda before Edda gives the permit to Alma before dusk.", "meaning_changing_join", False),
    # The deadline phrase is not a joining word.
    ("Boris sends the roster to Tilda before dusk, Edda gives the permit to Alma.", "comma_splice", False),
    ("Boris sends the roster to Tilda, then Edda gives the permit to Alma before dusk.", "comma_splice", False),
    ("Boris sends the roster to Tilda Edda gives the permit to Alma before dusk.", "run_on", False),
])
def test_v2_judges_how_the_two_clauses_are_joined(text, kind, joint):
    GEN.reset(0)
    row = GEN.evaluate_v2(text)
    assert [j["kind"] for j in row["clause_joins"]] == [kind]
    assert row["joint_success"] is joint and row["clause_join_ok"] is joint
    # v1 accepts every one of them; its verdict is kept alongside.
    assert row["evaluation_v1"]["joint_success"] is True


def test_v2_rejects_a_dangling_final_word_and_keeps_v1_fields():
    GEN.reset(0)
    row = GEN.evaluate_v2("Boris sends the roster to Tilda before dusk, while Edda gives the permit to Alma before.")
    assert row["dangling_final_word"] == "before" and not row["joint_success"]
    v1 = GEN.evaluate("Boris sends the roster to Tilda before dusk, while Edda gives the permit to Alma before.")
    assert {k: row[k] for k in ("output", "legacy_score", "constraints_ok", "constraint_violations")} == \
        {k: v1[k] for k in ("output", "legacy_score", "constraints_ok", "constraint_violations")}
    assert row["evaluator_version"] == "structural_rescue_eval_v2"


@pytest.mark.parametrize("text", [
    "Boris sends the roster to Tilda before dusk, while Edda gives Alma the permit.",
    "Boris sends the roster to Tilda, and Edda gives Alma the permit before dusk.",
])
def test_v2_passes_double_object_datives_that_v1_left_unknown(text):
    GEN.reset(0)
    row = GEN.evaluate_v2(text)
    assert row["evaluation_v1"]["role_status"] == "unknown"
    assert row["role_status"] == "pass" and row["joint_success"]
    assert [j["kind"] for j in row["clause_joins"]] == ["conjunction"]
