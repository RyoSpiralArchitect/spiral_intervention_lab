"""Audit the preserved rescue_v3 receipts and syntax_v1 without loading a model.

Run with ``python -S scripts/audit_clause_form_applicability.py``. JSON goes to
stdout; all input files are read-only. Historical scores are copied, not
recomputed. The new rule sees only source text and the generated prefix.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import runpy


RUNS = {
    "structural_rescue_cap012_v3_20261107":
        "a0c537479e8249fcbfcd19a55ef3ce29f4c9610dce811b6a970b4ec04c9b39b7",
    "structural_rescue_clauseform_v3_20261108":
        "4ef1be4aecc7a2436b94aef4bdf131fd96101602088232a9c862b45c82d8fa0c",
    "structural_rescue_clauseform_ds_v3_20261108":
        "76fc0dd072500512e2add5ac678f0450b17ec35329c64963b396eb00246cd252",
}
PRIMARY = "structural_rescue_clauseform_ds_v3_20261108"
DISPLAY_POLICIES = ("b0", "fixed_readout", "fixed_clause_form", "controller")
OUTCOME_KEYS = ("output", "legacy_score", "joint_success", "constraints_ok",
                "constraint_violations", "role_status", "role_unknown_reasons",
                "surface_notes", "first_token_id", "selected_recipe_id")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def packet_identity(packet: dict) -> str:
    # Same canonicalization as runtime.response_probe.identity, without torch.
    encoded = json.dumps(packet, sort_keys=True, separators=(",", ":"), default=str)
    return "packet:" + sha(encoded.encode())[:24]


def audit(root: Path) -> dict:
    evidence = {}

    def read(relative: Path) -> dict:
        raw = (root / relative).read_bytes()
        evidence[relative.as_posix()] = sha(raw)
        return json.loads(raw)

    helper = Path("SpiralInterventionLab/runtime/clause_form_gate.py")
    evidence[helper.as_posix()] = sha((root / helper).read_bytes())
    assess = runpy.run_path(str(root / helper))["assess"]
    bundles = {}
    summaries = []
    for name, expected_hash in RUNS.items():
        directory = Path("results") / name
        manifest = read(directory / "manifest.json")
        results = read(directory / "results.json")
        execution = read(directory / "execution_plan.json")
        require(evidence[(directory / "manifest.json").as_posix()] == expected_hash,
                name + ": frozen manifest bytes changed")
        require(results["manifest_sha256"] == execution["manifest_sha256"] == expected_hash,
                name + ": receipt/manifest binding differs")
        require(results["status"] == "complete", name + ": incomplete run")
        ids = [case["case_id"] for case in manifest["cases"]]
        require(len(ids) == len(set(ids)) == 12, name + ": expected twelve unique cases")
        require(ids == manifest["case_set"]["selected_case_ids"]
                == results["case_ids_requested"] == execution["case_ids"]
                == [case["case_id"] for case in results["cases"]],
                name + ": case order or selection differs")
        packets = {}
        selected_starters = reached_starters = 0
        for case in results["cases"]:
            cid = case["case_id"]
            packet = read(directory / (cid + "_observation.json"))
            packets[cid] = packet
            require(packet["case_id"] == cid and packet["generated_tail"] == case["prefix"],
                    name + ": prefix/packet mismatch: " + cid)
            recipe_ids = [recipe["recipe_id"] for recipe in packet["recipes"]]
            require(len(recipe_ids) == len(set(recipe_ids)) == case["eligible_recipe_count"]
                    and recipe_ids == case["recipe_ids"], name + ": catalog mismatch: " + cid)
            by_id = {recipe["recipe_id"]: recipe for recipe in packet["recipes"]}
            require(all(value is None or value in by_id
                        for value in case["fixed_recipe_ids"].values()),
                    name + ": unavailable fixed recipe: " + cid)
            for policy, outcome in case["policies"].items():
                selected = outcome["selected_recipe_id"]
                recipe = by_id.get(selected)
                require(selected is None or recipe is not None,
                        name + ": selected recipe absent: " + cid + ":" + policy)
                require(outcome["recipe_kind"] == (recipe["kind"] if recipe else None),
                        name + ": selected kind differs: " + cid + ":" + policy)
                require(outcome["state_restored"]
                        and outcome["baseline_repeat_max_abs_logit_delta"] == 0,
                        name + ": restoration/drift receipt differs: " + cid)
                if recipe and recipe["kind"] in {"readout", "clause_form"}:
                    edit = outcome["command"]["edits"][0]
                    require(edit["source"]["expr"]["target_token_ids"]
                            == [recipe["sources"][0]["token_id"]]
                            and edit["target"]["surface_id"] == recipe["surface"]["surface_id"],
                            name + ": selected target differs from command: " + cid)
                if policy == "controller":
                    selection = outcome["selection"]
                    expected_response = ({"decision": "probe", "recipe_id": selected}
                                         if selected else {"decision": "noop"})
                    require(json.loads(selection["response"]) == expected_response
                            and selection["packet_sha256"] == packet_identity(packet)
                            and selection["system_prompt_sha256"]
                            == manifest["controller_system_prompt_sha256"],
                            name + ": controller selection binding differs: " + cid)
                    if recipe and recipe["kind"] == "clause_form":
                        selected_starters += 1
                        reached_starters += (outcome["first_token_id"]
                                             == recipe["sources"][0]["token_id"])
            for policy in case["fixed_recipe_ids"]:
                require(case["policies"][policy]["selected_recipe_id"]
                        == case["fixed_recipe_ids"][policy],
                        name + ": fixed policy binding differs: " + cid)
        success_ids = {
            policy: [case["case_id"] for case in results["cases"]
                     if case["policies"][policy]["joint_success"]]
            for policy in results["policies_requested"]
        }
        for policy, wins in success_ids.items():
            require(len(wins) == results["aggregate"]["rescue_v3"][policy]["joint_successes"],
                    name + ": aggregate differs: " + policy)
        summaries.append({"run": directory.as_posix(), "manifest_sha256": expected_hash,
                          "recorded_success_case_ids": success_ids,
                          "controller_starter_selections": selected_starters,
                          "controller_starter_targets_reached": reached_starters})
        bundles[name] = (manifest, results, packets)

    primary_manifest, primary_results, packets = bundles[PRIMARY]
    for name, (manifest, results, _) in bundles.items():
        require(manifest["cases"] == primary_manifest["cases"],
                name + ": fixed inputs differ from primary run")
        for prior, current in zip(results["cases"], primary_results["cases"]):
            require(prior["prefix"] == current["prefix"], name + ": prefix differs across runs")
            for policy in ("b0", "fixed_readout"):
                require(prior["policies"][policy]["output"] == current["policies"][policy]["output"],
                        name + ": baseline/comma output differs across runs")

    rows = []
    for source_case, case in zip(primary_manifest["cases"], primary_results["cases"]):
        packet = packets[case["case_id"]]
        # Deliberately no case_id, annotations, outcomes, or recipe IDs enter assess.
        assessment = assess(source_case["source"], packet["generated_tail"])
        readout = {row["token_id"]: row for row in packet["structure_readout"]}
        candidates = []
        for recipe in packet["recipes"]:
            if recipe["kind"] == "patch":
                continue
            token = recipe["sources"][0]
            candidates.append({"recipe_id": recipe["recipe_id"], "kind": recipe["kind"],
                               "piece": token["piece"], "token_id": token["token_id"],
                               "surface": recipe["surface"],
                               "relative_write_cap": recipe["relative_write_cap"],
                               "observed_gap_to_top1": readout.get(token["token_id"], {}).get("gap_to_top1")})
        rows.append({"case_id": case["case_id"], "source": source_case["source"],
                     "generated_tail": packet["generated_tail"],
                     "legacy_applies_to_this_case": bool(re.search(r"\bwith\b|\bwho\b", source_case["source"])),
                     "new_syntax_assessment": assessment,
                     "recipe_count": len(packet["recipes"]),
                     "patch_recipe_count": sum(r["kind"] == "patch" for r in packet["recipes"]),
                     "readout_candidates": candidates,
                     "recorded_outcomes": {policy: {key: case["policies"][policy][key]
                                                    for key in OUTCOME_KEYS}
                                           for policy in DISPLAY_POLICIES}})
    return {"version": "clause_form_applicability_audit_v1", "status": "verified",
            "scope": "historical_development_receipts_plus_unmeasured_syntax_assessments",
            "model_calls": 0, "historical_scores_recomputed": False,
            "checks": ["frozen_manifest_binding", "fixed_case_order", "packet_prefix_binding",
                       "catalog_and_selection_binding", "controller_packet_and_prompt_hashes",
                       "recorded_restoration_and_zero_drift", "aggregate_counts",
                       "same_inputs_prefixes_and_baseline_comma_outputs_across_runs"],
            "evidence_sha256": evidence, "historical_runs": summaries,
            "new_syntax_counts": dict(Counter(row["new_syntax_assessment"]["source_second_clause_form"]
                                               for row in rows)),
            "cases": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.repo_root.resolve()), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
