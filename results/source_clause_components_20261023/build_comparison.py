"""Recompute from retained vectors, then join numerics without pooling evidence."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from SpiralInterventionLab.runtime import clause_components
from SpiralInterventionLab.runtime.response_probe import tensor_identity


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_default_device("cpu")
    torch.set_num_threads(4)
    reports = {}
    result = {"kind": "source_component_numerical_comparison", "status": "complete",
              "backend_and_dtype_confounded": True, "controller_calls": 0,
              "production_apply_allowed": False, "task_success": "unassessed", "numerics": {}}
    for name in ("mps16", "cpu32"):
        path = HERE / name / "measurement" / "trajectory.json"
        if not path.exists():
            supervision = json.loads((HERE / name / "supervision.json").read_text())
            assert supervision["status"] == "blocked", supervision
            result["status"] = "partial_numerical_replication_blocked"
            result["numerics"][name] = {"status": "not_run", "reason": supervision["stop_reason"],
                "preflight_free_disk_gib": supervision["preflight_free_disk_gib"],
                "supervision_sha256": sha(HERE / name / "supervision.json")}
            continue
        report = json.loads(path.read_text())
        salvage = (report["status"] == "error" and report.get("error") ==
                   "ValueError: component_input_or_source_identity_mismatch")
        assert report["status"] == "complete" or salvage, (name, report["status"])
        vectors = []
        for row in report["records"]:
            vector_path = path.parent / row["vectors_artifact"]
            assert sha(vector_path) == row["vectors_sha256"]
            vectors.append(torch.load(vector_path, map_location="cpu", weights_only=True))
        corrections = []
        if salvage:
            # Retain raw files and the error receipt. Accept only this narrow,
            # independently checkable signed-zero reconstruction, never a generic
            # hash mismatch or a different nonzero source.
            vectors = deepcopy(vectors)
            for record in vectors:
                for i, row in enumerate(record["result"]["operator_rows"]):
                    source = record["inputs"][i]["source"]
                    if tensor_identity(source) == row["source_tensor_identity"]:
                        continue
                    replacement = record["inputs"][i]["target"] * 0.0
                    assert record["kind"] == "radial_shrink"
                    assert torch.count_nonzero(source).item() == 0
                    assert torch.equal(source, replacement)
                    assert tensor_identity(replacement) == row["source_tensor_identity"]
                    corrections.append({"label": record["label"], "surface_index": i,
                        "reason": "capture_used_zeros_like_but_compiled_source_is_target_times_zero",
                        "before_identity": tensor_identity(source), "after_identity": tensor_identity(replacement),
                        "changed_zero_sign_count": int((torch.signbit(source) != torch.signbit(replacement)).sum())})
                    record["inputs"][i]["source"] = replacement
            assert corrections, "the original failure was not a signed-zero source mismatch"
        recomputed = clause_components.summarize(vectors)
        if not salvage:
            assert recomputed == report["comparison"], f"retained-vector summary differs: {name}"
        else:
            qualified = {"status": "reanalysis_complete", "original_status": report["status"],
                "original_error": report["error"], "original_receipt_sha256": sha(path),
                "raw_artifacts_unchanged": True, "new_model_forwards": 0,
                "source_identity_corrections": corrections,
                "analyzer_sha256": sha(HERE.parents[1] / "SpiralInterventionLab/runtime/clause_components.py"),
                "comparison": recomputed}
            (HERE / name / "reanalysis.json").write_text(json.dumps(qualified, indent=2, allow_nan=False) + "\n")
        reports[name] = report
        result["numerics"][name] = {
            "receipt": str(path.relative_to(HERE)), "receipt_sha256": sha(path),
            "status": "qualified_reanalysis_complete" if salvage else "complete",
            "retained_vector_hashes_verified": True,
            "original_comparison_reproduced": True if not salvage else None,
            "signed_zero_reconstruction_verified": salvage,
            "observed_prefix": report["observed_prefix"], "model_execution": report["model_execution"],
            "diagnostic_budget_used": report["diagnostic_budget_used"],
            "declared_diagnostic_replay_count": 2 + sum(r["result"]["physical_replay_count"] for r in vectors),
            "diagnostic_interval_forward_counts": report["diagnostic_interval_forward_counts"],
            "forward_counts_total": report["forward_counts_total"],
            "surfaces": recomputed["surfaces"],
        }
    if len(reports) == 2:
        mps, cpu = reports["mps16"], reports["cpu32"]
        for key in ("component_plan", "source_sha256", "profile", "position", "observed_prefix", "code_sha256"):
            assert mps[key] == cpu[key], f"unmatched {key}"
        assert mps["cache_alignment"]["tokens_identity"] == cpu["cache_alignment"]["tokens_identity"]
        assert mps["reference_match"]["bindings"] == cpu["reference_match"]["bindings"]
    result["reference_bindings"] = reports["mps16"]["reference_match"]["bindings"]
    result["tokens_identity"] = reports["mps16"]["cache_alignment"]["tokens_identity"]
    result["comparison_scope"] = "same_prefix_sources_sites_caps_not_semantic_subspace_or_additive_mediation"
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name, value in result["numerics"].items():
        for site in value.get("surfaces", []):
            print(json.dumps({"numerics": name, "site": site["surface"]["surface_id"],
                "matched_dose": site["actual_dose_matches_cap_within_2pct"],
                "zero_null_repeat_floors": site["all_captured_null_repeat_and_baseline_floors_zero"],
                "rows": [{"condition": c["label"],
                          "norm": c["actual_written_delta_norms"][0],
                          "comma": next(r["probability_deltas"][0] for r in c["readout_rows"] if r["token_id"] == 11),
                          "while": next(r["probability_deltas"][0] for r in c["readout_rows"] if r["token_id"] == 1418),
                          "comma_minus_while_logit_contrast": c["first_minus_second_logit_contrast_deltas"][0],
                          "greedy": c["raw_greedy_token_ids"]} for c in site["conditions"]]}))


if __name__ == "__main__":
    main()
