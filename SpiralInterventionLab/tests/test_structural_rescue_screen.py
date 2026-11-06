import hashlib
import json

import pytest

from SpiralInterventionLab.examples import structural_rescue_benchmark_v2 as v2
from SpiralInterventionLab.examples import structural_rescue_screen as screen
from SpiralInterventionLab.tasks.structural_rescue import CASES, StructuralRescueEnv
from SpiralInterventionLab.tasks.structural_rescue_generator import FAMILIES, generate


def test_generator_is_seeded_disjoint_and_roles_stay_checkable():
    cases = generate(screen.SEED, screen.COUNT)
    assert cases == generate(screen.SEED, screen.COUNT) and cases != generate(screen.SEED + 1, screen.COUNT)
    frozen_words = {word for case in CASES for frame in case.frames for word in frame}
    for index, case in enumerate(cases):
        (a, verb_a, obj_a, b), (c, verb_c, obj_c, d) = case.frames
        assert len({a, b, c, d}) == 4 and verb_a != verb_c and obj_a != obj_c
        assert not {a, b, c, d, obj_a, obj_c} & frozen_words
        # Required terms match case-insensitively, so a capitalized opener still counts.
        assert case.deadline.startswith("before ") and case.deadline in case.source.casefold()
        assert case.source.startswith("In order to") == (FAMILIES[index % 3] == "purpose_first")
        # Ten required words plus two articles, two "to"s and a connective fit the cap.
        assert len(StructuralRescueEnv(case)._SCENARIOS[0]["required_terms"]) == 9


def test_selection_keeps_generation_order_boundary_failures_only():
    rows = [{"case_id": f"gen_{i:03d}", "boundary_found": i % 4 != 0, "joint_success": i % 3 == 0}
            for i in range(40)]
    chosen = screen.select(rows)
    assert len(chosen) == screen.SELECTION_LIMIT
    assert chosen == sorted(chosen)
    assert all(int(c[4:]) % 4 != 0 and int(c[4:]) % 3 != 0 for c in chosen)


def write_screen(tmp_path, *, status="complete", selected=("gen_001", "gen_004")):
    (tmp_path / "manifest.json").write_bytes(screen.manifest_bytes())
    digest = hashlib.sha256(screen.manifest_bytes()).hexdigest()
    (tmp_path / "results.json").write_text(json.dumps({"status": status, "manifest_sha256": digest,
                                                       "selected_case_ids": list(selected)}))


def test_selected_cases_are_rebuilt_from_the_generator_and_bound_to_receipts(tmp_path):
    write_screen(tmp_path)
    cases, binding = screen.selected_cases(tmp_path)
    assert [c.case_id for c in cases] == ["gen_001", "gen_004"] and {c.split for c in cases} == {"rescue"}
    assert cases[0].source == generate(screen.SEED, screen.COUNT)[1].source
    assert binding["screen_results_sha256"] == hashlib.sha256((tmp_path / "results.json").read_bytes()).hexdigest()
    frozen = v2.manifest({"cases": cases, "binding": binding})
    assert frozen["case_set"] == binding and [c["case_id"] for c in frozen["cases"]] == ["gen_001", "gen_004"]
    assert v2.manifest_bytes({"cases": cases, "binding": binding}) != v2.manifest_bytes()
    # The evaluator field added 2026-11-06 intentionally moved the digest off 92f1138a.
    assert hashlib.sha256(v2.manifest_bytes()).hexdigest().startswith("9df274ca19caf621")


def test_selected_cases_refuse_incomplete_or_tampered_screens(tmp_path):
    write_screen(tmp_path, status="running")
    with pytest.raises(ValueError, match="incomplete"):
        screen.selected_cases(tmp_path)
    write_screen(tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="differs"):
        screen.selected_cases(tmp_path)
