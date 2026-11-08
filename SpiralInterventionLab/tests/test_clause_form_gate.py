"""Model-free regressions for the opt-in, source-derived applicability rule."""
from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from SpiralInterventionLab.runtime.clause_form_gate import RULE, assess, filter_recipes


RECORDS = Path(__file__).resolve().parents[2] / "results/structural_rescue_clauseform_ds_v3_20261108"
FINITE = ("Dara should take the parcel to Ivo, while Remy should bring the atlas to Zora, "
          "in order to leave before dawn.")
PARTICIPIAL = ("In order to leave before dawn, Dara should take the parcel to Ivo, "
               "with Remy bringing the atlas to Zora.")
RELATIVE = ("Before dawn, Dara should take the parcel to Ivo; meanwhile, "
            "Remy, who should bring the atlas to Zora, also completes.")
TAIL = "Dara takes the parcel to Ivo"


@pytest.fixture(scope="module")
def recorded_inputs():
    # Read only source/prefix evidence; no saved outcomes or evaluation frames
    # are passed to the rule. IDs label regression examples, not policy inputs.
    manifest = json.loads((RECORDS / "manifest.json").read_text())
    return {row["case_id"]: (
        row["source"],
        json.loads((RECORDS / (row["case_id"] + "_observation.json")).read_text())["generated_tail"])
        for row in manifest["cases"]}


@pytest.mark.parametrize("case_id,form,eligible", [
    ("gen_008", "participial", True), ("gen_017", "finite", True),
    ("gen_018", "participial", True), ("gen_028", "participial", True),
    ("gen_039", "relative_agent", False), ("gen_058", "participial", True),
    ("gen_059", "relative_agent", False), ("gen_068", "participial", True),
    ("gen_073", "participial", True), ("gen_074", "relative_agent", False),
    ("gen_078", "participial", True), ("gen_088", "participial", True),
])
def test_fixed_twelve_are_classified_by_pending_source_syntax(recorded_inputs, case_id, form, eligible):
    assert len(recorded_inputs) == 12
    result = assess(*recorded_inputs[case_id])
    assert result["rule"] == RULE
    assert result["source_second_clause_form"] == form
    assert result["prefix_first_clause_complete"] is True
    assert result["applies"] is eligible
    if not eligible:
        assert result["reason"] == "embedded_action_with_competing_matrix_predicate"


@pytest.mark.parametrize("source", [FINITE, PARTICIPIAL, RELATIVE])
def test_names_objects_case_and_spacing_do_not_determine_applicability(source):
    replacements = {"Dara": "Élodie", "Ivo": "Jean-Luc", "Remy": "Søren", "Zora": "O'Neil",
                    "parcel": "report", "atlas": "letter", "dawn": "evening"}
    pattern = re.compile(r"\b(?:" + "|".join(replacements) + r")\b")
    renamed = lambda text: pattern.sub(lambda match: replacements[match[0]], text)
    expected = assess(source, TAIL)
    assert assess(renamed(source), renamed(TAIL)) == expected
    assert assess(source.upper(), TAIL.lower()) == expected
    assert assess(" \n\t" + source.replace(" ", "\t \n") + "\n", "  " + TAIL.replace(" ", "\n")) == expected


@pytest.mark.parametrize("action,finite,past", [
    ("send", "sends", "sent"), ("take", "takes", "took"), ("bring", "brings", "brought"),
    ("give", "gives", "gave"), ("hand", "hands", "handed"), ("mail", "mails", "mailed"),
])
def test_source_action_alignment_accepts_declared_finite_and_modal_forms(action, finite, past):
    source = FINITE.replace("should take", "should " + action)
    for predicate in (finite, past, "will " + action, "can " + action):
        result = assess(source, "Dara " + predicate + " parcel to Ivo")
        assert result["prefix_first_clause_complete"] is True
        assert result["applies"] is True


@pytest.mark.parametrize("participle", ["sending", "taking", "bringing", "giving", "handing", "mailing"])
def test_participle_conversion_is_not_tied_to_the_recorded_verbs(participle):
    result = assess(PARTICIPIAL.replace("bringing", participle), TAIL)
    assert result["source_second_clause_form"] == "participial"
    assert result["applies"] is True


@pytest.mark.parametrize("link", [", while ", ", whereas ", ", and ", "; meanwhile, ", "; at the same time, "])
def test_direct_and_relative_clauses_remain_distinct_for_every_declared_connector(link):
    direct = FINITE.replace(", while ", link)
    relative = RELATIVE.replace("; meanwhile, ", link)
    assert assess(direct, TAIL)["applies"] is True
    result = assess(relative, TAIL)
    assert result["source_second_clause_form"] == "relative_agent"
    assert result["applies"] is False


@pytest.mark.parametrize("predicate", ["rests", "works", "waits"])
def test_relative_exclusion_does_not_depend_on_the_observed_substitution_verb(predicate):
    result = assess(RELATIVE.replace("completes", predicate), TAIL)
    assert result["source_second_clause_form"] == "relative_agent"
    assert result["reason"] == "embedded_action_with_competing_matrix_predicate"
    assert result["applies"] is False


@pytest.mark.parametrize("source", [
    "Dara should take the parcel with care to Ivo, while Remy should bring the atlas to Zora.",
    "Dara should take the parcel to Ivo, with care, while Remy should bring the atlas to Zora.",
    "Dara, who should take the parcel to Ivo, also waits, while Remy should bring the atlas to Zora.",
    "Dara should take the parcel to Ivo, with Remy, who should bring the atlas to Zora, also working.",
    FINITE + " The notes mention who and with.",
    FINITE.replace("should bring", "should not bring"),
    FINITE.replace("Dara", "Mary Jane"),
    FINITE.replace("should take", "should carry"),
    FINITE.replace("to Ivo,", "for Ivo,"),
    FINITE[:-1],
])
def test_unrelated_keywords_and_unsupported_constructions_stay_unknown(source):
    result = assess(source, TAIL)
    assert result["source_second_clause_form"] == "unknown"
    assert result["reason"] == "unsupported_source_syntax"
    assert result["applies"] is False


@pytest.mark.parametrize("tail", [
    "", "Dara takes the parcel", "Dara takes the parcel to", "Dara taking the parcel to Ivo",
    "Dara sends the parcel to Ivo", "Remy takes the parcel to Ivo", "Dara takes the atlas to Ivo",
    "Dara takes the parcel to Zora", TAIL + ",", TAIL + ".", TAIL + " while", TAIL + " with",
    TAIL + " while Remy brings the atlas to Zora", TAIL + " before dawn",
])
def test_only_the_aligned_complete_first_clause_is_an_eligible_boundary(tail):
    result = assess(FINITE, tail)
    assert result["source_second_clause_form"] == "finite"
    assert result["prefix_first_clause_complete"] is False
    assert result["applies"] is False
    assert result["reason"] == "prefix_not_aligned_complete_first_clause"


def readout(recipe_id, kind="readout", token_id=1418, layer=27, cap=0.12):
    return {"recipe_id": recipe_id, "kind": kind,
            "sources": [{"token_id": token_id, "piece": " while" if token_id == 1418 else ","}],
            "surface": {"surface_id": f"s_resid_pre_l{layer}_last", "site": "resid_pre", "layer": layer},
            "relative_write_cap": cap}


def starter(evidence):
    return {**readout("starter", "clause_form"), "applicability": evidence}


def test_blocked_effect_cannot_reenter_as_a_source_readout_and_catalog_is_unchanged():
    recipes = [readout("comma", token_id=11), starter(assess(RELATIVE, TAIL)), readout("while_alias"),
               readout("other_layer", layer=14), readout("other_cap", cap=0.08),
               {**readout("patch", "patch"), "component": "common_orthogonal"}]
    before = deepcopy(recipes)
    offered, excluded = filter_recipes(recipes)
    assert [row["recipe_id"] for row in offered] == ["comma", "other_layer", "other_cap", "patch"]
    assert [row["recipe_id"] for row in excluded] == ["starter", "while_alias"]
    assert {row["reason"] for row in excluded} == {"embedded_action_with_competing_matrix_predicate"}
    assert recipes == before


def test_supported_finite_source_keeps_both_while_provenances_and_original_order():
    recipes = [readout("comma", token_id=11), readout("while_source"), starter(assess(FINITE, TAIL))]
    before = deepcopy(recipes)
    offered, excluded = filter_recipes(recipes)
    assert offered == recipes == before
    assert excluded == []


@pytest.mark.parametrize("evidence", [None, False, {}, {"rule": "legacy", "applies": True},
    {"rule": RULE}, {"rule": RULE, "applies": "true"}, {"rule": RULE, "applies": 1},
    {"rule": RULE, "applies": True},
    {"rule": RULE, "applies": True, "source_second_clause_form": "relative_agent",
     "prefix_first_clause_complete": True},
    {"rule": RULE, "applies": True, "source_second_clause_form": "finite",
     "prefix_first_clause_complete": False},
    {"rule": RULE, "applies": False, "reason": ""},
    {"rule": RULE, "applies": False, "reason": None},
    {"rule": RULE, "applies": False, "reason": False},
    {"rule": RULE, "applies": False, "reason": 0},
])
def test_missing_or_malformed_applicability_excludes_starter_and_effect_alias(evidence):
    recipes = [starter(evidence), readout("alias"), readout("comma", token_id=11)]
    before = deepcopy(recipes)
    offered, excluded = filter_recipes(recipes)
    assert [row["recipe_id"] for row in offered] == ["comma"]
    assert [row["recipe_id"] for row in excluded] == ["starter", "alias"]
    assert all(isinstance(row["reason"], str) and row["reason"] for row in excluded)
    assert recipes == before


def test_absent_applicability_field_fails_closed_but_plain_catalogs_are_unchanged():
    recipes = [readout("starter", "clause_form"), readout("alias")]
    offered, excluded = filter_recipes(recipes)
    assert offered == []
    assert {row["recipe_id"] for row in excluded} == {"starter", "alias"}
    plain = [readout("comma", token_id=11), readout("while")]
    assert filter_recipes(plain) == (plain, [])
    assert filter_recipes([]) == ([], [])
