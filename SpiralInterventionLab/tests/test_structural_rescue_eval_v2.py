"""Evaluator v2 against every output stored in the contract receipts."""
import json

import pytest
from pathlib import Path

from SpiralInterventionLab.tasks.structural_rescue import CASES, StructuralRescueEnv
from SpiralInterventionLab.tasks.structural_rescue_generator import generate

RECEIPTS = Path(__file__).resolve().parents[2] / "results" / "structural_rescue_contract_20261103"


def stored_rows():
    for run in ("precheck_r4", "pilot", "rescue"):
        report = json.loads((RECEIPTS / run / "results.json").read_text())
        for case in report["cases"]:
            for row in list(case["policies"].values()) + case.get("calibration_matrix", []):
                if row.get("status") == "ok":
                    yield case["case_id"], row
    for row in json.loads((RECEIPTS / "screen_r2" / "results.json").read_text())["rows"]:
        yield row["case_id"], row


def envs():
    cases = {c.case_id: c for c in CASES} | {c.case_id: c for c in generate(20261103, 72)}
    out = {}
    for case_id, case in cases.items():
        out[case_id] = StructuralRescueEnv(case)
        out[case_id].reset(0)
    return out


def test_v1_reproduces_every_stored_verdict_and_v2_keeps_it_alongside():
    tasks = envs()
    checked = 0
    for case_id, row in stored_rows():
        v1 = tasks[case_id].evaluate(row["output"])
        v2 = tasks[case_id].evaluate_v2(row["output"])
        for key in ("joint_success", "role_status", "constraints_ok", "role_unknown_reasons", "legacy_score"):
            assert v1[key] == row[key], (case_id, key)
            assert v2["evaluation_v1"].get(key, v1[key]) == row[key]
        assert v1["role_report"] == row["role_report"]
        checked += 1
    assert checked == 236


def test_v2_changes_on_the_screen_are_splices_datives_and_one_dangling_word():
    tasks = envs()
    screen = json.loads((RECEIPTS / "screen_r2" / "results.json").read_text())["rows"]
    lost, gained = [], []
    for row in screen:
        v2 = tasks[row["case_id"]].evaluate_v2(row["output"])
        if row["joint_success"] and not v2["joint_success"]:
            lost.append((row["case_id"], [j["kind"] for j in v2["clause_joins"]], v2["dangling_final_word"]))
        if not row["joint_success"] and v2["joint_success"]:
            gained.append(row["case_id"])
    assert [c for c, joins, _ in lost if joins == ["comma_splice"]] == [
        "gen_003", "gen_006", "gen_012", "gen_027", "gen_029", "gen_030", "gen_032", "gen_044", "gen_063", "gen_065"]
    assert [c for c, _, word in lost if word] == ["gen_026"] and len(lost) == 11
    # Double-object datives, nine of them in candidates without a natural boundary.
    assert gained == ["gen_009", "gen_015", "gen_018", "gen_020", "gen_024", "gen_031", "gen_052", "gen_057",
                      "gen_059", "gen_066", "gen_070", "gen_071"]


# Hand-labelled outputs from the evaluator review (2026-11-03); True means a joint success.
# Decisions fixed before any v2 run: non-finite adjuncts pass, meaning-changing joins fail,
# colons and dashes join, articles are not checked.
ADVERSARIAL = [
    ("gen_018", " Tilda mails Boris the badge before dawn; Kira sends the roster to Olga.", True),
    ("gen_018", " Tilda mails the badge to Boris, Kira sends the roster to Olga before dawn.", False),
    ("gen_018", " Kira sends the roster to Olga before dawn while Tilda mails the badge to Boris.", True),
    ("gen_018", " Kira sends the roster to Olga, before dawn, Tilda mails the badge to Boris.", False),
    ("gen_018", " Kira sends the roster to Olga, and, before dawn, Tilda mails the badge to Boris.", True),
    ("gen_026", " Marta will mail the letter to Olga before, Boris will take the blueprint to Wanda before midnight.", False),
    ("gen_026", " Marta will mail the letter to Olga before; Boris will take the blueprint to Wanda before midnight.", False),
    ("gen_018", " Kira sends the roster to Olga after lunch, Tilda mails the badge to Boris before dawn.", False),
    ("gen_018", " Kira sends the roster to Olga as agreed, Tilda mails the badge to Boris before dawn.", False),
    ("gen_018", " Kira sends the roster to Olga, also Tilda mails the badge to Boris before dawn.", False),
    ("gen_018", " Kira sends the roster to Olga and Boris, Tilda mails the badge to Boris before dawn.", False),
    ("gen_018", " While Kira sends the roster to Olga, Tilda mails the badge to Boris before dawn.", True),
    ("gen_018", " Before dawn, while Kira sends the roster to Olga, Tilda mails the badge to Boris.", True),
    ("gen_018", " While Kira sends the roster to Olga, and Tilda mails the badge to Boris before dawn.", False),
    ("gen_018", " Before dawn, Kira sends the roster to Olga, Tilda mails the badge to Boris.", False),
    ("gen_018", " Kira sends the roster to Olga, and then Tilda mails the badge to Boris before dawn.", True),
    ("gen_018", " Kira sends the roster to Olga, then Tilda mails the badge to Boris before dawn.", False),
    ("confirm_paraphrase_a", " Hana brings the sketch to Emil before lunch; meanwhile, Dario sends the notice to Cora.", True),
    ("confirm_paraphrase_a", " Hana brings the sketch to Emil before lunch, meanwhile, Dario sends the notice to Cora.", False),
    ("gen_018", " Kira sends the roster to Olga before dawn Tilda mails the badge to Boris.", False),
    ("gen_026", " Marta will mail the letter to Olga before midnight, while Boris will take the blueprint to Wanda before.", False),
    ("gen_018", " Kira sends the roster to Olga before dawn, while Tilda mails the badge to.", False),
    ("gen_042", " Edda takes Yusuf's ticket before dawn; Henrik sends the ledger to Felix.", False),
    ("gen_042", " Edda takes Yusuf’s ticket before dawn; Henrik sends the ledger to Felix.", False),
    ("gen_009", " Rosa sends Wanda the blueprint before dawn, while Yusuf gives Celia the badge.", False),
    ("gen_009", " Rosa sends Yusuf the blueprint before dawn, while Yusuf gives Wanda the badge.", False),
    ("gen_018", " Kira sends Olga the badge before dawn; Tilda mails Boris the roster.", False),
    ("gen_009", " Rosa sends the blueprint to Celia before dawn, while Yusuf gives her the badge.", False),
    ("gen_018", " Kira sends the roster to Olga before dawn, with Tilda mailing Boris the badge.", True),
    ("gen_058", " Boris sends the roster to Tilda before midnight, with Edda giving Alma the permit.", False),
    ("gen_018", " Kira sends the roster to Olga before dawn, Tilda mailing Boris the badge.", True),
    ("gen_036", " Tilda sends drawing to Ilse before dawn; Yusuf mails permit to Stefan.", True),
    # "handed" is an action form in v2; "took" would still miss the required term "take".
    ("gen_015", " Nikos handed Vesna the proposal before dusk, while Celia takes the permit to Tilda.", True),
    ("gen_015", " Nikos handed Vesna the proposal before dusk, while Celia took the permit to Tilda.", False),
    ("cal_review", " Nora will take the sample to Ivo before Selim sends the report to Yara before dusk.", False),
    ("gen_018", " Kira sends the roster to Olga unless Tilda mails the badge to Boris before dawn.", False),
    ("gen_018", " Kira sends the roster to Olga before dawn: Tilda mails the badge to Boris.", True),
    ("gen_018", " Kira sends the roster to Olga before dawn — Tilda mails the badge to Boris.", True),
    ("gen_018", " Kira sends the roster to Olga before dawn, and Tilda mails the badge to Boris after.", False),
    ("gen_018", " Kira sends the roster to Olga before dawn; Tilda mails Boris the badge to Olga.", False),
]


@pytest.mark.parametrize("case_id,output,expected", ADVERSARIAL)
def test_v2_on_hand_labelled_adversarial_outputs(case_id, output, expected):
    row = envs()[case_id].evaluate_v2(output)
    assert row["joint_success"] is expected, (row["clause_joins"], row["role_status"], row["role_unknown_reasons"],
                                              row["dangling_final_word"], row["constraint_violations"])


def test_v2_reports_bare_themes_and_leaves_join_unchecked_when_roles_fail():
    tasks = envs()
    row = tasks["gen_036"].evaluate_v2(" Tilda sends drawing to Ilse before dawn; Yusuf mails permit to Stefan.")
    assert row["bare_themes"] == ["drawing", "permit"] and row["joint_success"]
    failed = tasks["cal_review"].evaluate_v2(" Nora will take the sample to Ivo before sending the report to Yara before dusk.")
    assert failed["role_status"] == "fail" and failed["clause_join_ok"] is None and failed["clause_joins"] == []


def test_failure_types_name_the_first_v2_reason():
    from SpiralInterventionLab.examples.summarize_structural_rescue import failure_type
    tasks = envs()
    screen = json.loads((RECEIPTS / "screen_r2" / "results.json").read_text())["rows"]
    kinds = {}
    for row in screen:
        if row["boundary_found"]:
            kind = failure_type(tasks[row["case_id"]].evaluate_v2(row["output"]))
            if kind:
                kinds[kind] = kinds.get(kind, 0) + 1
    assert kinds == {"constraint": 5, "join_comma_splice": 10, "dangling": 1}
