"""Exercise the real catalog -> packet -> controller parser path without models."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from SpiralInterventionLab.examples import structural_rescue_benchmark_v2 as v2
from SpiralInterventionLab.examples.structural_rescue_screen import load_selected_cases
from SpiralInterventionLab.runtime.response_probe import identity


ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "results/structural_rescue_clauseform_ds_v3_20261108"
SCREEN = ROOT / "results/structural_rescue_screen_v3_20261107"


def fixture_packet(case_id, mode="syntax_v1", source=None):
    saved = json.loads((RECORDS / f"{case_id}_observation.json").read_text())
    row = next(c for c in json.loads((RECORDS / "manifest.json").read_text())["cases"]
               if c["case_id"] == case_id)
    # Deliberately no case_id, gold frames, deadline, or outcome on this task.
    task = SimpleNamespace(case=SimpleNamespace(source=source or row["source"]),
                           _SCENARIOS=[{"required_terms": ()}])
    worker = SimpleNamespace(final_text=lambda: saved["generated_tail"])
    recipes = deepcopy(saved["recipes"])
    with patch.object(v2.structural_contract, "eligible_sources", return_value=[]), \
         patch.object(v2.structural_contract, "policy_for", return_value="policy"), \
         patch.object(v2.structural_contract, "surface", return_value={}), \
         patch.object(v2.structural_contract, "patch_recipes",
                      return_value=[r for r in recipes if r["kind"] == "patch"]), \
         patch.object(v2.structural_contract, "readout_recipes",
                      return_value=[r for r in recipes if r["kind"] == "readout"]), \
         patch.object(v2.structural_contract, "clause_form_recipes",
                      return_value=[r for r in recipes if r["kind"] == "clause_form"]):
        catalog, _, _ = v2.catalog_for(worker, task, 28, cap=0.12, clause_form=True,
                                       clause_form_gate=mode)
    packet = v2.compact_packet(worker, catalog, "arbitrary_case_id",
        {"structure_readout_rows": saved["structure_readout"]}, datasheet=True,
        cap=0.12, clause_form=True, clause_form_gate=mode)
    return packet, v2.fixed_ids(catalog)


def provider_for(recipe_id):
    response = {"decision": "noop"} if recipe_id is None else {"decision": "probe", "recipe_id": recipe_id}
    return SimpleNamespace(provider_name="mock", model_name="mock",
        complete=Mock(return_value=SimpleNamespace(text=json.dumps(response), usage={}, metadata={})))


def select(packet, fixed, recipe_id, mode="syntax_v1"):
    provider = provider_for(recipe_id)
    selected, trace = v2.select("controller", packet, provider=provider, fixed=fixed,
        datasheet=True, cap=0.12, clause_form=True, clause_form_gate=mode)
    return selected, trace, provider


@pytest.mark.parametrize("case_id, expected", [("gen_017", True), ("gen_059", False)])
def test_catalog_assessment_reaches_packet_without_changing_the_control_catalog(case_id, expected):
    packet, fixed = fixture_packet(case_id)
    legacy, old_fixed = fixture_packet(case_id, "legacy")
    assert fixed == old_fixed
    assert [r["recipe_id"] for r in packet["recipes"]] == [r["recipe_id"] for r in legacy["recipes"]]
    starter = next(r for r in packet["recipes"] if r["kind"] == "clause_form")
    assert starter["applicability"]["applies"] is expected
    assert starter["applies_to_this_case"] is expected
    assert (starter["recipe_id"] in packet["controller_gate"]["offered_recipe_ids"]) is expected
    assert not {"frames", "outcomes", "joint_success"} & set(packet)


def test_false_advisory_is_still_selectable_in_the_explicit_legacy_condition():
    packet, fixed = fixture_packet("gen_017", "legacy")
    starter = next(r for r in packet["recipes"] if r["kind"] == "clause_form")
    assert starter["applies_to_this_case"] is False
    selected, trace, _ = select(packet, fixed, starter["recipe_id"], mode="legacy")
    assert selected == starter["recipe_id"] and trace["status"] == "selected"


def test_excluded_starter_response_is_rejected_with_the_actual_provider_payload_hash():
    packet, fixed = fixture_packet("gen_059")
    before = deepcopy(packet)
    selected, trace, provider = select(packet, fixed, fixed["fixed_clause_form"])
    assert selected is None and trace["status"] == "error"
    assert provider.complete.call_count == trace["provider_calls"] == 1
    payload = provider.complete.call_args[0][0].payload
    assert fixed["fixed_clause_form"] not in {r["recipe_id"] for r in payload["recipes"]}
    assert fixed["fixed_readout"] in {r["recipe_id"] for r in payload["recipes"]}
    assert trace["packet_sha256"] == identity("packet:", payload) != identity("packet:", packet)
    assert trace["controller_gate"] == packet["controller_gate"]
    assert packet == before


def test_supported_finite_source_preserves_both_while_recipe_provenances():
    packet, fixed = fixture_packet("gen_017")
    candidates = [r for r in packet["recipes"] if r["kind"] in ("readout", "clause_form")
                  and r["sources"][0]["piece"] == " while"]
    assert {r["kind"] for r in candidates} == {"readout", "clause_form"}
    for recipe in candidates:
        selected, trace, _ = select(packet, fixed, recipe["recipe_id"])
        assert selected == recipe["recipe_id"] and trace["status"] == "selected"


def test_source_readout_alias_cannot_bypass_relative_source_exclusion():
    source = ("Alma should take the manual to Vesna, while Yusuf, who should bring the voucher "
              "to Nikos, also completes.")
    packet, fixed = fixture_packet("gen_017", source=source)
    aliases = [r for r in packet["recipes"] if r["kind"] in ("readout", "clause_form")
               and r["sources"][0]["piece"] == " while"]
    assert len(aliases) == 2
    for recipe in aliases:
        selected, trace, provider = select(packet, fixed, recipe["recipe_id"])
        assert selected is None and trace["status"] == "error"
        assert all(r["recipe_id"] not in {a["recipe_id"] for a in aliases}
                   for r in provider.complete.call_args[0][0].payload["recipes"])


@pytest.mark.parametrize("policy", ["b0", "fixed", "fixed_readout", "fixed_clause_form", "random"])
def test_syntax_mode_preserves_fixed_and_random_comparators(policy):
    packet, fixed = fixture_packet("gen_059")
    before = deepcopy(packet)
    kwargs = {"provider": None, "fixed": fixed, "datasheet": True, "cap": 0.12, "clause_form": True}
    assert v2.select(policy, packet, clause_form_gate="syntax_v1", **kwargs) == v2.select(policy, packet, **kwargs)
    assert packet == before


@pytest.mark.parametrize("pick", [None, "comma"])
def test_gate_preserves_noop_and_valid_comma_without_predicting_their_outcomes(pick):
    packet, fixed = fixture_packet("gen_059")
    recipe_id = fixed["fixed_readout"] if pick == "comma" else None
    selected, trace, _ = select(packet, fixed, recipe_id)
    assert selected == recipe_id
    assert trace["status"] == ("selected" if recipe_id else "noop")
    assert "joint_success" not in trace


def test_empty_filtered_catalog_makes_no_provider_call():
    packet, fixed = fixture_packet("gen_059")
    packet["recipes"] = [r for r in packet["recipes"] if r["kind"] == "clause_form"]
    selected, trace, provider = select(packet, fixed, fixed["fixed_clause_form"])
    assert selected is None and trace["provider_calls"] == 0
    assert trace["controller_gate"]["offered_recipe_ids"] == []
    provider.complete.assert_not_called()


def test_gate_mode_requires_an_explicit_matching_packet_and_clause_form():
    packet, fixed = fixture_packet("gen_059", "legacy")
    with pytest.raises(ValueError, match="packet_gate_mismatch"):
        select(packet, fixed, None)
    packet, fixed = fixture_packet("gen_059")
    with pytest.raises(ValueError, match="packet_gate_mismatch"):
        select(packet, fixed, fixed["fixed_clause_form"], mode="legacy")
    with pytest.raises(ValueError, match="requires_clause_form"):
        v2.manifest(clause_form_gate="syntax_v1")
    with pytest.raises(ValueError, match="unknown_clause_form_gate"):
        v2.manifest(clause_form=True, clause_form_gate="typo")


def test_legacy_manifest_bytes_are_preserved_and_syntax_condition_is_versioned():
    # These hashes were recorded at PR50 main, before this change.
    assert hashlib.sha256(v2.manifest_bytes()).hexdigest() == "9df274ca19caf6219594262c4fd112cd3e5c05572161ac111cf8b32b96380bcc"
    cases, binding = load_selected_cases(SCREEN)
    case_set = {"cases": cases, "binding": binding}
    args = {"case_set": case_set, "datasheet": True, "cap": 0.12, "clause_form": True}
    assert hashlib.sha256(v2.manifest_bytes(**args)).hexdigest() == "787833c52eb75dbeac16fcfae7fc9a4945abbc7108d8aba24774c2aa5824951d"
    old, new = v2.manifest(**args), v2.manifest(**args, clause_form_gate="syntax_v1")
    changed = {"version", "controller_system_prompt_sha256", "controller_clause_form_gate"}
    assert {k: v for k, v in old.items() if k not in changed} == {k: v for k, v in new.items() if k not in changed}
    assert old["version"] != new["version"]
    assert old["controller_system_prompt_sha256"] != new["controller_system_prompt_sha256"]


def test_cli_prepares_a_new_condition_and_rejects_running_it_as_legacy(tmp_path):
    output = tmp_path / "syntax"
    args = ["--output-dir", str(output), "--clause-form", "--datasheet", "--cap", "0.12",
            "--screen-dir", str(SCREEN)]
    with patch.object(v2, "load_worker_model") as model, patch.object(v2, "create_controller_provider") as provider:
        assert v2.main(["--prepare", "--clause-form-gate", "syntax_v1", *args]) == 0
        frozen = (output / "manifest.json").read_bytes()
        assert json.loads(frozen)["controller_clause_form_gate"]["rule"] == "syntax_v1"
        with pytest.raises(ValueError, match="predeclared_manifest"):
            v2.main(["--run", "--manifest-sha256", hashlib.sha256(frozen).hexdigest(), *args])
        model.assert_not_called()
        provider.assert_not_called()
