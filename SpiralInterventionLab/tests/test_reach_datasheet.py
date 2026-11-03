import json

import pytest

from SpiralInterventionLab.runtime import reach_datasheet


def push(case_id, token, cap, gap, closure, first, piece=",", kind="readout"):
    return {"recipe_id": f"{case_id}:{kind}:{token}@{cap}", "kind": kind, "component": kind, "cap": cap,
            "sources": [{"token_id": token, "piece": piece}], "first_token_id": first,
            "readout": {"edited_winner": first, "rows": [{"token_id": token, "baseline_gap_to_fixed_winner": gap,
                                                          "edited_gap_to_fixed_winner": gap - closure}]}}


def case(case_id, winner=1603, pushes=(), eligible=(11, 1418, 627)):
    return {"case_id": case_id, "status": "ok", "b0_winner_id": winner, "b0": {"first_token_id": winner},
            "eligible_sources": [{"token_id": t} for t in eligible], "pushes": list(pushes)}


def pool():
    """Six ' before'-top cases where ',' closes 1.6-2.1 logits, and one ','-top case."""
    cases = [case(f"c{i}", pushes=[push(f"c{i}", 11, 0.08, 1.0 + 0.2 * i, 1.6 + 0.1 * i,
                                        11 if 1.6 + 0.1 * i > 1.0 + 0.2 * i else 1603)]) for i in range(6)]
    cases.append(case("top", winner=11, pushes=[push("top", 11, 0.08, 0.0, 0.0, 11),
                                                 push("top", 1418, 0.08, 3.0, 2.5, 11, piece=" while")]))
    return cases


def test_rows_separate_already_top_targets_and_top1_classes():
    rows = reach_datasheet.table_rows(pool())
    top = [r for r in rows if r["case_id"] == "top"]
    assert [r["already_top"] for r in top] == [True, False]
    assert {r["top1_class"] for r in top} == {"source_token"}
    assert {r["top1_class"] for r in rows if r["case_id"] != "top"} == {"other"}
    # The ' while' push left ',' on top: a change to neither B0's token nor its target.
    assert top[1]["third_token"] is False and top[1]["target_became_next"] is False


def test_build_excludes_already_top_rows_and_hashes_the_sheet():
    sheet = reach_datasheet.build(reach_datasheet.table_rows(pool()), cap=0.08, provenance={"pool": "toy"})
    comma = sheet["readout"]["11|other"]
    assert comma["closure_logits"]["n"] == 6 and comma["closure_logits"]["min"] == pytest.approx(1.6)
    assert sheet["readout"]["11|source_token"] == {"token_id": 11, "piece": ",", "top1_class": "source_token",
                                                   "closure_logits": None, "already_top": 1,
                                                   "target_became_next": 0, "third_token_changes": 0}
    reach_datasheet.verify(sheet, cap=0.08)
    with pytest.raises(ValueError, match="cap"):
        reach_datasheet.verify(sheet, cap=0.1)
    with pytest.raises(ValueError, match="tampered"):
        reach_datasheet.verify({**sheet, "min_n": 1}, cap=0.08)
    text = json.dumps({k: v for k, v in sheet.items() if k != "provenance"})
    assert "joint_success" not in text and "output" not in text and '"c1"' not in text


def test_leave_one_out_and_patch_cap_are_explicit():
    rows = reach_datasheet.table_rows(pool()) + [
        {"case_id": "c0", "kind": "patch", "cap": 0.12, "first_token_changed": False}]
    sheet = reach_datasheet.build(rows, cap=0.08, provenance={}, patch_rows=rows, patch_cap=0.12, exclude_case="c0")
    assert sheet["readout"]["11|other"]["closure_logits"]["n"] == 5
    assert sheet["patch"] == {"cap_measured": 0.12, "n": 0, "first_token_changes": 0}
    assert sheet["provenance"]["excluded_case"] == "c0"


def test_entries_band_reach_and_the_rule_skips_endings_tops_and_uncalibrated():
    sheet = reach_datasheet.build(reach_datasheet.table_rows(pool()), cap=0.08, provenance={})
    recipes = [{"recipe_id": r, "kind": "readout", "sources": [{"token_id": t, "piece": p}]}
               for r, t, p in (("comma", 11, ","), ("while", 1418, " while"), ("stop", 627, ".\n"))]
    def entries(gaps, top=None):
        rows = [{"token_id": t, "gap_to_top1": g, "is_greedy_token": t == top} for t, g in gaps.items()]
        return [reach_datasheet.recipe_entry(r, rows, sheet, case_top1_class="other") for r in recipes]
    near = entries({11: 1.2, 1418: 3.0, 627: 0.1})
    assert [e["reach"] for e in near] == ["likely", "uncalibrated", "uncalibrated"]
    assert near[2]["ends_output"] and near[0]["closure_logits"]["n"] == 6
    assert reach_datasheet.rule_choice(recipes, near) == "comma"
    assert entries({11: 1.8, 1418: 3.0, 627: 0.1})[0]["reach"] == "borderline"
    far = entries({11: 2.5, 1418: 3.0, 627: 0.1})
    assert far[0]["reach"] == "unlikely" and reach_datasheet.rule_choice(recipes, far) is None
    top = entries({11: 0.0, 1418: 3.0, 627: 0.1}, top=11)
    assert top[0]["reach"] == "already_top" and reach_datasheet.rule_choice(recipes, top) is None
    patch = reach_datasheet.recipe_entry({"kind": "patch", "sources": []}, [], sheet, case_top1_class="other")
    assert patch == {"cap_measured": None, "first_token_changes": 0, "n": 0}
