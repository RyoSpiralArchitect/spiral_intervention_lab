"""Hash-checked retained-vector comparison across two different natural prefixes."""
import hashlib
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PRIOR = ROOT / "results/source_clause_components_cpu32_20261024/cpu32/measurement/trajectory.json"
CURRENT = HERE / "cpu32/measurement/trajectory.json"
sys.path.insert(0, str(ROOT))
from SpiralInterventionLab.runtime import clause_choice, clause_components, clause_trajectory


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    report = json.loads(path.read_text())
    assert report["status"] == "complete", report.get("error", report["status"])
    records = []
    for row in report["records"]:
        vector_path = path.parent / row["vectors_artifact"]
        assert sha(vector_path) == row["vectors_sha256"], vector_path
        record = torch.load(vector_path, map_location="cpu", weights_only=True)
        assert clause_trajectory.compact_record(record) == {
            k: v for k, v in row.items() if k not in {"vectors_artifact", "vectors_sha256"}}
        records.append(record)
    assert clause_components.summarize(records) == report["comparison"]
    pieces = {r["token_id"]: r["piece"] for s in report["comparison"]["surfaces"]
              for c in s["conditions"] for r in c["readout_rows"]}
    if "choice_comparison" in report:
        saved = report["choice_comparison"]
        pieces[saved["fixed_baseline_winner_token_id"]] = saved["fixed_baseline_winner_piece"]
        for surface in saved["surfaces"]:
            for condition in surface["conditions"]:
                pieces.update(zip(condition["edited_winner_ids"], condition["edited_winner_pieces"]))
    choice = clause_choice.summarize(records, component_labels=clause_components.COMPONENTS,
                                    decode_token=pieces.__getitem__)
    if "choice_comparison" in report:
        assert choice == report["choice_comparison"]
    assert report["controller_calls"] == report["rollout_apply_count"] == 0
    assert report["production_apply_allowed"] is False
    assert report["declared_diagnostic_replay_count"] == 30
    assert report["diagnostic_budget_used"] == 9
    return report, {
        "receipt": str(path.relative_to(ROOT)), "receipt_sha256": sha(path),
        "vector_hashes_and_compact_records_verified": True, "component_summary_reproduced": True,
        "position": report["position"], "prefix": report["observed_prefix"],
        "tokens_identity": report["cache_alignment"]["tokens_identity"],
        "forward_counts_total": report["forward_counts_total"],
        "diagnostic_interval_forward_counts": report["diagnostic_interval_forward_counts"],
        "choice_comparison": choice,
        "component_conditions": report["comparison"]["surfaces"],
    }


def main():
    torch.set_default_device("cpu")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    output = HERE / "comparison.json"
    if output.exists():
        raise FileExistsError(output)
    prior, before = load(PRIOR)
    current, after = load(CURRENT)
    for key in ("source_sha256", "component_plan", "profile", "model_execution", "torch_threads",
                "checkpoint_sha256_verified", "versions", "base_commit", "device", "dtype"):
        assert prior[key] == current[key], f"unmatched {key}"
    assert prior["position"] == 8 and current["position"] == 9
    assert prior["cache_alignment"]["tokens_identity"] == current["verified_anchor_tokens_identity"]
    assert current["verified_boundary_tokens_identity"] == current["cache_alignment"]["tokens_identity"]
    assert prior["reference_match"]["bindings"] == current["reference_match"]["bindings"]
    assert prior["observed_prefix"] != current["observed_prefix"]
    old_code, new_code = prior["code_sha256"], current["code_sha256"]
    changes = [{"path": k, "before_sha256": old_code.get(k), "after_sha256": new_code.get(k)}
               for k in sorted(old_code.keys() | new_code.keys()) if old_code.get(k) != new_code.get(k)]
    assert {c["path"] for c in changes} == {
        "SpiralInterventionLab/runtime/clause_choice.py",
        "SpiralInterventionLab/examples/replay_source_clause_trajectory.py",
        "SpiralInterventionLab/tests/test_clause_choice.py",
        "SpiralInterventionLab/tests/test_clause_trajectory.py",
    }, changes
    for k, value in new_code.items():
        assert sha(ROOT / k) == value, f"code drift since run: {k}"
    result = {
        "kind": "source_component_adjacent_boundary_choice_audit", "status": "complete",
        "prefixes_differ": True, "same_prefix_replication": False,
        "checkpoint_sources_sites_caps_numerics_matched": True,
        "new_model_forwards_for_reanalysis": 0, "original_artifacts_modified": False,
        "code_changes": changes, "controller_calls": 0, "production_apply_allowed": False,
        "task_success": "unassessed", "semantic_ownership": "not_established",
        "generalization": "not_established_two_boundaries_one_trajectory",
        "reference_bindings": current["reference_match"]["bindings"],
        "boundaries": {"position8": before, "position9": after},
    }
    with output.open("x") as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": "complete", "output": str(output)}))


if __name__ == "__main__":
    main()
