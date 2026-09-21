"""Scripted review-accounting audit, not another model/controller trajectory."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from SpiralInterventionLab.runtime.loop import _extract_diagnostic_requests
from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source_jsonl.read_bytes()
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    worker = object.__new__(HookedTransformerWorkerRuntime)
    worker.max_diagnostic_calls_per_run = 12
    worker.diagnostic_result_window = 12
    worker._diagnostic_results = [e for e in events if e["event"] == "controller_diagnostic_result" and e["step"] < 4]
    worker._diagnostic_review_cache = {}
    worker._evidence_inspection_count = 0
    worker._pending_diagnostic_events = []
    worker._last_packet = {}
    worker._collect_active_edits = lambda: []
    rows = []
    for step in range(4, 9):
        command = next(e["command"] for e in events if e["event"] == "controller_command" and e["step"] == step)
        recorded = next(e for e in events if e["event"] == "controller_diagnostic_result" and e["step"] == step)
        prefix = next(e["generated_tail"] for e in events if e["event"] == "controller_observation" and e["step"] == step + 1)
        worker._steps = step + 1
        # Byte markers only exercise changed-prefix cache signaling; they are
        # not model token IDs or a claimed reconstruction of model state.
        worker._segments = [SimpleNamespace(kind="output", token_ids=list(prefix.encode()))]
        with patch.object(worker, "_execute_controller_diagnostic_request", return_value=deepcopy(recorded)) as execute:
            result = worker.request_controller_diagnostics(_extract_diagnostic_requests(command, {}))[0]
            rows.append({"controller_step": step, "diagnostic": result["diagnostic"],
                "status": result["status"], "review_executor_calls": execute.call_count,
                "budget_after": result["budget_after"],
                "prefix_changed_since_review": result.get("prefix_changed_since_review"),
                "production_apply_allowed": result.get("production_apply_allowed")})
    report = {"scope": "command-meta-only scripted accounting with recorded results substituted",
        "source_jsonl_sha256": hashlib.sha256(raw).hexdigest(), "rows": rows,
        "avoided_reviews": sum(r["review_executor_calls"] == 0 for r in rows),
        "controller_calls": 0, "model_calls": 0, "controller_behavior_improvement_demonstrated": False,
        "production_apply_allowed": False}
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
