"""Read-only reanalysis of both retained experiments; emit one new comparison."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / "results/source_clause_components_20261023"
sys.path.insert(0, str(ROOT))
from SpiralInterventionLab.runtime import clause_components
from SpiralInterventionLab.runtime.response_probe import tensor_identity


# Exact changes inspected before this replication, not an open-ended allowlist.
EXPECTED_CODE_CHANGES = {
    "SpiralInterventionLab/runtime/clause_trajectory.py": (
        "4e246cad0b313bfbbb94b109dbcf5b513bbd4e1d51a33759a671079affafca7b",
        "64b955084015d567f28e4b335d732829b5d7c1666484b07fb453efe295ce553b",
        "radial_capture_preserves_compiled_signed_zero",
    ),
    "SpiralInterventionLab/examples/replay_source_clause_trajectory.py": (
        "8f61136edf5dc5b6a53f107f81e47f197685c291b18c700c7236b4541d7dc717",
        "324dcfd0442ff7e728345a8a56cbc6cbccd17fa428a92d06ee8dcf00be6c05d2",
        "completed_replay_count_written_before_analysis",
    ),
    "SpiralInterventionLab/tests/test_clause_components.py": (
        "70188855629fd83bf8b0c97ef498c2a0185c8a0882336c3b6c7a8e0778b836cd",
        "a5dae9c879e412a8e959c95899f6087408b62da9733f63801e36d4e763cdc547",
        "signed_zero_capture_and_unexplained_identity_mismatch_regressions",
    ),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_numerics(path, *, qualified_path=None):
    report = json.loads(path.read_text())
    qualified = json.loads(qualified_path.read_text()) if qualified_path else None
    if qualified:
        assert report["status"] == "error"
        assert report["error"] == "ValueError: component_input_or_source_identity_mismatch"
        assert qualified["original_receipt_sha256"] == sha(path)
    else:
        assert report["status"] == "complete", report.get("error", report["status"])
    vectors, corrections = [], []
    for row in report["records"]:
        vector_path = path.parent / row["vectors_artifact"]
        assert sha(vector_path) == row["vectors_sha256"], vector_path
        record = torch.load(vector_path, map_location="cpu", weights_only=True)
        if qualified:
            record = deepcopy(record)
            for i, operator_row in enumerate(record["result"]["operator_rows"]):
                source = record["inputs"][i]["source"]
                if tensor_identity(source) == operator_row["source_tensor_identity"]:
                    continue
                replacement = record["inputs"][i]["target"] * 0.0
                assert record["kind"] == "radial_shrink"
                assert torch.count_nonzero(source).item() == 0
                assert torch.equal(source, replacement)
                assert tensor_identity(replacement) == operator_row["source_tensor_identity"]
                corrections.append({
                    "label": record["label"], "surface_index": i,
                    "reason": "capture_used_zeros_like_but_compiled_source_is_target_times_zero",
                    "before_identity": tensor_identity(source), "after_identity": tensor_identity(replacement),
                    "changed_zero_sign_count": int((torch.signbit(source) != torch.signbit(replacement)).sum()),
                })
                record["inputs"][i]["source"] = replacement
        vectors.append(record)
    if qualified:
        assert corrections == qualified["source_identity_corrections"]
        assert corrections, "qualification requires an exact signed-zero reconstruction"
    comparison = clause_components.summarize(vectors)
    assert comparison == (qualified or report)["comparison"], "retained-vector summary changed"
    replay_count = 2 + sum(r["result"]["physical_replay_count"] for r in vectors)
    assert replay_count == 30
    assert report["diagnostic_budget_used"] == 9
    assert report["controller_calls"] == report["rollout_apply_count"] == 0
    assert report["production_apply_allowed"] is False
    if not qualified:
        assert report["declared_diagnostic_replay_count"] == replay_count
    summary = {
        "receipt": str(path.relative_to(ROOT)), "receipt_sha256": sha(path),
        "original_status": report["status"],
        "status": "qualified_reanalysis_complete" if qualified else "complete",
        "retained_vector_hashes_verified": True, "retained_comparison_reproduced": True,
        "signed_zero_reconstruction_verified": bool(qualified),
        "source_identity_corrections": corrections,
        "observed_prefix": report["observed_prefix"], "model_execution": report["model_execution"],
        "diagnostic_budget_used": report["diagnostic_budget_used"],
        "declared_diagnostic_replay_count": replay_count,
        "diagnostic_interval_forward_counts": report["diagnostic_interval_forward_counts"],
        "forward_counts_total": report["forward_counts_total"], "surfaces": comparison["surfaces"],
    }
    if qualified_path:
        summary["qualified_receipt"] = str(qualified_path.relative_to(ROOT))
        summary["qualified_receipt_sha256"] = sha(qualified_path)
    return report, summary


def main():
    output = HERE / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    torch.set_default_device("cpu")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    mps, mps_summary = load_numerics(
        PRIOR / "mps16/measurement/trajectory.json", qualified_path=PRIOR / "mps16/reanalysis.json")
    cpu, cpu_summary = load_numerics(HERE / "cpu32/measurement/trajectory.json")
    for key in ("component_plan", "source_sha256", "profile", "position", "observed_prefix",
                "base_commit", "checkpoint_sha256_verified", "versions", "torch_threads"):
        assert mps[key] == cpu[key], f"unmatched {key}"
    assert cpu["model_execution"] == {
        "parameter_devices": ["cpu"], "parameter_dtypes": ["torch.float32"], "n_layers": 28,
    }
    assert mps["cache_alignment"]["tokens_identity"] == cpu["cache_alignment"]["tokens_identity"]
    assert mps["reference_match"]["bindings"] == cpu["reference_match"]["bindings"]
    assert mps["code_sha256"].keys() == cpu["code_sha256"].keys()
    changed = {k: (mps["code_sha256"][k], cpu["code_sha256"][k])
               for k in mps["code_sha256"] if mps["code_sha256"][k] != cpu["code_sha256"][k]}
    assert changed == {k: v[:2] for k, v in EXPECTED_CODE_CHANGES.items()}, changed
    code_changes = [{"path": k, "before_sha256": v[0], "after_sha256": v[1], "reason": v[2]}
                    for k, v in EXPECTED_CODE_CHANGES.items()]
    result = {
        "kind": "source_component_numerical_replication", "status": "complete",
        "new_model_forwards_for_reanalysis": 0, "original_artifacts_modified": False,
        "backend_and_dtype_confounded": True, "matched_prefix_sources_sites_caps": True,
        "code_identity": "matched_except_exact_predeclared_capture_accounting_and_test_fixes",
        "code_changes": code_changes, "controller_calls": 0, "production_apply_allowed": False,
        "task_success": "unassessed", "semantic_ownership": "not_established",
        "additive_mediation": "not_established",
        "tokens_identity": cpu["cache_alignment"]["tokens_identity"],
        "reference_bindings": cpu["reference_match"]["bindings"],
        "numerics": {"mps16": mps_summary, "cpu32": cpu_summary}, "contrasts": [],
    }
    for name, summary in result["numerics"].items():
        for surface in summary["surfaces"]:
            conditions = {c["label"]: c for c in surface["conditions"]}
            positive = conditions["difference_positive"]["first_minus_second_logit_contrast_deltas"]
            negative = conditions["difference_negative"]["first_minus_second_logit_contrast_deltas"]
            result["contrasts"].append({
                "numerics": name, "surface_id": surface["surface"]["surface_id"],
                "positive_difference_contrast_deltas": positive,
                "negative_difference_contrast_deltas": negative,
                "relative_contrast_sign_flip_observed": min(positive) > 0 and max(negative) < 0,
                "conditions": [{
                    "label": c["label"],
                    "comma_probability_deltas": next(r["probability_deltas"] for r in c["readout_rows"]
                                                     if r["token_id"] == 11),
                    "while_probability_deltas": next(r["probability_deltas"] for r in c["readout_rows"]
                                                     if r["token_id"] == 1418),
                    "comma_minus_while_logit_contrast_deltas": c["first_minus_second_logit_contrast_deltas"],
                    "actual_written_delta_norms": c["actual_written_delta_norms"],
                    "raw_greedy_token_ids": c["raw_greedy_token_ids"],
                } for c in surface["conditions"]],
            })
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "contrasts": result["contrasts"]}), flush=True)


if __name__ == "__main__":
    main()
