"""Verify only retained completed conditions; never promote an interrupted run."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

torch.set_default_device("cpu")
torch.set_num_threads(4)
from SpiralInterventionLab.runtime import clause_choice, clause_radial, clause_trajectory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    path = directory / "cpu32/measurement/trajectory.json"
    receipt = json.loads(path.read_text())
    supervision = json.loads((directory / "cpu32/supervision.json").read_text())
    if supervision["status"] != "stopped_or_failed":
        raise ValueError("only_a_stopped_attempt_can_be_partially_verified")
    labels = [r["label"] for r in receipt["records"]]
    planned = clause_radial.plan([11, 1418])["condition_order"]
    if not 4 <= len(labels) < len(planned) or labels != planned[:len(labels)]:
        raise ValueError("subset_must_be_a_strict_predeclared_prefix_with_all_four_base_conditions")
    for relative, expected in receipt["code_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"analysis_code_drift:{relative}")
    records = []
    for compact in receipt["records"]:
        artifact = path.parent / compact["vectors_artifact"]
        if digest(artifact) != compact["vectors_sha256"]:
            raise ValueError("raw_vector_hash_mismatch")
        raw = torch.load(artifact, map_location="cpu", weights_only=True)
        if clause_trajectory.compact_record(raw) != {k: v for k, v in compact.items() if k not in {"vectors_artifact", "vectors_sha256"}}:
            raise ValueError("compact_and_raw_receipt_disagree")
        records.append(raw)
    components = tuple(r["label"] for r in records if r["kind"] == "component")
    trajectory = clause_trajectory.summarize(records, component_labels=components)
    choice = clause_choice.summarize(records, component_labels=components)
    previous = json.loads((ROOT / "results/source_direction_controls_cpu32_20261024/cpu32/measurement/trajectory.json").read_text())
    shared = {
        "same_prefix": receipt["observed_prefix"] == previous["observed_prefix"],
        "same_tokens": receipt["cache_alignment"]["tokens_identity"] == previous["cache_alignment"]["tokens_identity"],
        "same_checkpoint": receipt["checkpoint_sha256_verified"] == previous["checkpoint_sha256_verified"],
        "shared_condition_count": min(len(labels), 7),
        "same_shared_inputs": [r["inputs"] for r in receipt["records"][:7]] == [r["inputs"] for r in previous["records"][:min(len(labels), 7)]],
        "same_shared_captured_vectors": [[f["vectors"] for f in r["frames"]] for r in receipt["records"][:7]]
            == [[f["vectors"] for f in r["frames"]] for r in previous["records"][:min(len(labels), 7)]],
    }
    shapes = []
    for index in range(len(records[0]["inputs"])):
        left, right = [r["inputs"][index] for r in records[:2]]
        if not torch.equal(left["target"], right["target"]):
            raise ValueError("source_targets_differ")
        common = torch.stack([left["source"], right["source"]]).mean(0)
        shapes.append({**clause_radial.geometry(common, left["target"]),
                       "scope": "source_input_geometry_only_not_a_component_effect_measurement"})
    result = {"status": "verified_completed_subset_only", "experiment_complete": False,
        "stop_reason": supervision.get("stop_reason"), "elapsed_seconds": supervision["elapsed_seconds"],
        "supervision_sha256": digest(directory / "cpu32/supervision.json"), "receipt_sha256": digest(path),
        "completed_conditions": [r["label"] for r in records],
        "missing_conditions": planned[len(labels):],
        "previous_cpu32_identity": shared, "trajectory": trajectory, "choice": choice,
        "common_input_geometry": shapes,
        "completed_condition_replay_count": sum(r["result"]["physical_replay_count"] for r in records),
        "captured_completed_condition_forward_count": sum(len(r["frames"]) for r in records),
        "total_forward_count": None,
        "total_forward_count_status": "unknown_inflight_condition_not_persisted_before_process_termination",
        "radial_vs_orthogonal_effect_conclusion": "unmeasured_do_not_infer_from_geometry",
        "controller_calls": 0, "production_apply_allowed": False}
    output = directory / "partial_verification.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"trajectory", "choice"}}, indent=2))


if __name__ == "__main__":
    main()
