"""Rebuild interview input from immutable logs, without a provider or model run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from SpiralInterventionLab.runtime.debrief import event_anchored_log_digest


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild(log_dir: Path, output_dir: Path) -> dict:
    input_path = log_dir / "post_run_debrief_input.json"
    original = input_path.read_bytes()
    packet = json.loads(original)
    before = packet["log_digest"]
    source_paths = sorted(log_dir.rglob("*.jsonl"))
    source_hashes = {str(path.relative_to(log_dir)): file_hash(path)
                     for path in source_paths}
    after = event_anchored_log_digest(log_dir)
    packet["log_digest"] = after
    packet["offline_rebuild"] = {"source_input_sha256": hashlib.sha256(original).hexdigest(),
        "provider_called": False, "scope": "input_reconstruction_not_a_new_interview_or_run"}
    old_ids = set(before["coverage_manifest"]["included_event_ids"])
    new_ids = set(after["coverage_manifest"]["included_event_ids"])
    comparison = {
        "artifact_kind": "offline_debrief_input_comparison", "provider_calls": 0, "worker_runs": 0,
        "source_log_dir": str(log_dir.resolve()), "source_jsonl_sha256": source_hashes,
        "old_digest_characters": len(json.dumps(before)), "new_digest_characters": len(json.dumps(after)),
        "character_budget": after["coverage_manifest"]["character_budget"],
        "added_event_ids": sorted(new_ids - old_ids), "removed_event_ids": sorted(old_ids - new_ids),
        "event_anchor_kinds": {row["event_id"]: row["anchor_kinds"] for row in after["event_anchors"]},
        "diagnostic_accounting_summary": after["diagnostic_accounting_summary"],
        "final_constraint_report": after["final_constraint_report"],
        "source_input_unchanged": input_path.read_bytes() == original,
        "source_logs_unchanged": all(file_hash(log_dir / name) == value
                                     for name, value in source_hashes.items()),
        "interpretation": "No counterfactual rollout or interview response was generated; recorded spend is not refunded.",
    }
    if not comparison["source_input_unchanged"] or not comparison["source_logs_unchanged"]:
        raise RuntimeError("source changed while rebuilding the interview input")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("rebuilt_post_run_debrief_input.json", packet), ("comparison.json", comparison)):
        (output_dir / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing artifacts are never replaced.")
    args = parser.parse_args()
    result = rebuild(args.log_dir, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "provider_calls": 0,
        "new_digest_characters": result["new_digest_characters"],
        "source_logs_unchanged": result["source_logs_unchanged"]}))


if __name__ == "__main__":
    main()
