"""Explicit post-hoc partition of the first capture; never overwrite its error receipt.

The verified call path is preflight canary, baseline x2, packet canary + edit,
packet canary + edit. Accept this partition ONLY if every baseline/preparation
vector is exactly equal, every edited repeat is exactly equal, and the independent
post-hook blend check passes. Otherwise another labelled run is required.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch

from SpiralInterventionLab.runtime import clause_trajectory


directory = Path(__file__).resolve().parent
source_path = directory / "measurement" / "trajectory.json"
output = directory / "trajectory_reanalysis.json"
if output.exists():
    raise FileExistsError(output)
original = json.loads(source_path.read_text())
if (original["status"] != "error" or original["error"] != "ValueError: trajectory_forward_count_or_tokens_mismatch"
        or len(original["records"]) != 4):
    raise ValueError("not the specifically identified initial capture")

records = []
for receipt in original["records"]:
    artifact = directory / "measurement" / receipt["vectors_artifact"]
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != receipt["vectors_sha256"]:
        raise ValueError("raw vector artifact changed")
    record = torch.load(artifact, map_location="cpu", weights_only=True)
    frames = record["frames"]
    if record["result"]["status"] != "ok" or len(frames) != 7 or len(record["result"]["operator_rows"]) != 1:
        raise ValueError("partial or different execution shape")
    for group in ([0, 1, 2, 3, 5], [4, 6]):
        anchor = frames[group[0]]
        for index in group:
            if (anchor["tokens_identity"] != frames[index]["tokens_identity"]
                    or list(anchor["vectors"]) != list(frames[index]["vectors"])
                    or any(not torch.equal(value, frames[index]["vectors"][key])
                           for key, value in anchor["vectors"].items())):
                raise ValueError("posthoc grouping is ambiguous; do not repair noisy data")
    for frame in frames:
        frame["origin"] = {"kind": "posthoc_preparation", "attribution": "static_call_path_with_exact_equivalence_check"}
    edit_id = record["result"]["operator_rows"][0]["edit"]["id"]
    for replay_index, index in enumerate([1, 2, 4, 6]):
        frames[index]["origin"] = {"kind": "replay", "replay_index": replay_index,
                                    "edit_ids": [] if replay_index < 2 else [edit_id],
                                    "attribution": "posthoc_exact_vector_equivalence_not_live_tag"}
    records.append(record)

report = deepcopy(original)
report.update(status="reanalysis_complete", original_status=original["status"], original_error=original["error"],
              original_receipt_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
              analysis_method="posthoc_static_call_partition_verified_by_exact_all_stage_vector_equivalence",
              analysis_caveat="capture-time call-site labels absent; counts outside the four conditions were not traced",
              analysis_source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in (Path(__file__), Path(clause_trajectory.__file__))},
              comparison=clause_trajectory.summarize(records),
              declared_diagnostic_replay_count=18)
report.pop("error")
report["records"] = [{**clause_trajectory.compact_record(record), "vectors_artifact": receipt["vectors_artifact"],
                      "vectors_sha256": receipt["vectors_sha256"]}
                     for record, receipt in zip(records, original["records"], strict=True)]
output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(json.dumps({"status": report["status"], "comparison": report["comparison"]["status"],
                  "source_geometry": report["comparison"]["surfaces"][0]["source_geometry"]}))
