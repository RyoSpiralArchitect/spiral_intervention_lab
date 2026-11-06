import hashlib
import json

import pytest

from SpiralInterventionLab.examples import structural_rescue_screen as screen
from SpiralInterventionLab.examples import structural_rescue_screen_v2 as screen_v2
from SpiralInterventionLab.tasks.structural_rescue import CASES
from SpiralInterventionLab.tasks.structural_rescue_generator import VERBS_V2, generate


def test_generator_v2_is_seeded_disjoint_and_to_dative_only():
    cases = generate(screen_v2.SEED, screen_v2.COUNT, VERBS_V2)
    assert cases == generate(screen_v2.SEED, screen_v2.COUNT, VERBS_V2)
    assert len(cases) == screen_v2.COUNT
    frozen_words = {word for case in CASES for frame in case.frames for word in frame}
    for case in cases:
        (a, verb_a, obj_a, b), (c, verb_c, obj_c, d) = case.frames
        assert {verb_a, verb_c} <= set(VERBS_V2) and verb_a != verb_c
        assert not {a, b, c, d, obj_a, obj_c} & frozen_words
        # Every clause keeps the to <recipient> surface the boundary rule needs.
        assert f" to {b}" in case.source and f" to {d}" in case.source


def test_screen_v2_manifest_is_separate_stable_and_frozen_before_running():
    assert screen_v2.manifest_bytes() == screen_v2.manifest_bytes()
    frozen = screen_v2.manifest()
    assert frozen["version"] != screen.manifest()["version"]
    assert frozen["generator"] != screen.manifest()["generator"]
    assert frozen["count"] == screen_v2.COUNT > screen.manifest()["count"]
    assert frozen["pools"]["verbs"] == VERBS_V2
    assert len(frozen["candidates"]) == screen_v2.COUNT
    assert all("prompt_sha256" in row for row in frozen["candidates"])
    assert frozen["evaluator_changes_since_screen_v1"]
    # The v1 manifest is unchanged so its receipts stay readable; pin its
    # digest so a silent pool or wording drift fails here, not at load time.
    assert screen.manifest()["version"] == "structural_rescue_screen_v1"
    assert (hashlib.sha256(screen.manifest_bytes()).hexdigest()
            == "e018bbcc917a79513ab88e4b28f5542447a1db5d324de54df3d1a176351416cf")


def test_generator_rejects_verb_pools_the_check_cannot_recognize():
    with pytest.raises(ValueError, match="verbs"):
        generate(1, 1, ("give",))
    with pytest.raises(ValueError, match="verbs"):
        generate(1, 1, ("send", "teleport"))
    assert generate(1, 1, VERBS_V2[:2])


def write_screen_v2(tmp_path, *, status="complete", selected=("gen_002", "gen_005")):
    (tmp_path / "manifest.json").write_bytes(screen_v2.manifest_bytes())
    digest = hashlib.sha256(screen_v2.manifest_bytes()).hexdigest()
    (tmp_path / "results.json").write_text(json.dumps({"status": status, "manifest_sha256": digest,
                                                       "selected_case_ids": list(selected)}))


def test_selected_cases_v2_are_rebuilt_and_bound_with_the_new_split(tmp_path):
    write_screen_v2(tmp_path)
    cases, binding = screen_v2.selected_cases(tmp_path)
    assert [c.case_id for c in cases] == ["gen_002", "gen_005"]
    assert {c.split for c in cases} == {"rescue_v2"}
    assert cases[0].source == generate(screen_v2.SEED, screen_v2.COUNT, VERBS_V2)[2].source
    assert binding["screen_results_sha256"] == hashlib.sha256((tmp_path / "results.json").read_bytes()).hexdigest()


def test_load_selected_cases_dispatches_on_the_screen_manifest_version(tmp_path):
    v1_dir, v2_dir = tmp_path / "v1", tmp_path / "v2"
    v1_dir.mkdir()
    (v1_dir / "manifest.json").write_bytes(screen.manifest_bytes())
    digest = hashlib.sha256(screen.manifest_bytes()).hexdigest()
    (v1_dir / "results.json").write_text(json.dumps({"status": "complete", "manifest_sha256": digest,
                                                     "selected_case_ids": ["gen_001"]}))
    v2_dir.mkdir()
    write_screen_v2(v2_dir, selected=("gen_002",))
    assert {c.split for c in screen.load_selected_cases(v1_dir)[0]} == {"rescue"}
    assert {c.split for c in screen.load_selected_cases(v2_dir)[0]} == {"rescue_v2"}
    with pytest.raises(ValueError, match="incomplete"):
        write_screen_v2(v2_dir, status="running")
        screen_v2.selected_cases(v2_dir)


def test_load_selected_cases_rejects_an_unknown_version(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"version": "structural_rescue_screen_v3"}))
    with pytest.raises(ValueError, match="unknown_screen_manifest_version"):
        screen.load_selected_cases(tmp_path)


def test_double_object_names_after_to_are_not_dative_recipients():
    from SpiralInterventionLab.tasks.role_binding import check_role_frames
    report = check_role_frames(
        "Nora will take the sample to Ivo, while Selim sends to Yara the report.",
        (("Nora", "take", "sample", "Ivo"), ("Selim", "send", "report", "Yara")))
    row = report["frames"][1]
    assert row["bound_recipient"] is None or row["bound_recipient_mode"] != "double_object"
