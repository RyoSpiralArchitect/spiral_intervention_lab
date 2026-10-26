"""Read-only pilot report: paired task outcomes, branch response and actual work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def snapshot_member(path: str) -> str:
    if "SpiralInterventionLab/" in path:
        path = "SpiralInterventionLab/" + path.split("SpiralInterventionLab/", 1)[1]
    if (path != "pyproject.toml" and not path.startswith("SpiralInterventionLab/")) or ".." in Path(path).parts:
        raise ValueError("unsupported_snapshot_member")
    return path


def summarize(report: dict) -> dict:
    policies = report["policies_requested"]
    summaries = []
    calibration = []
    for case in report["cases"]:
        baseline = case["policies"].get("b0", {})
        for policy, row in case["policies"].items():
            if row.get("status") != "ok":
                continue
            telemetry = row.get("runtime_telemetry", [])
            summaries.append({"case_id": case["case_id"], "split": case["split"], "policy": policy,
                "comparison_eligible": case.get("comparison_eligible"),
                "selection_status": row.get("selection", {}).get("status"),
                "legacy_score": row["legacy_score"], "joint_success": row["joint_success"],
                "role_status": row["role_status"], "output": row["output"],
                "output_changed_vs_b0": row["output"] != baseline.get("output"),
                "branch_changed": row.get("natural_branch_changed"),
                "actual_write_norm": telemetry[0].get("blend_delta_norm") if telemetry else None,
                "relative_write_norm": telemetry[0].get("relative_write_norm") if telemetry else None,
                "actual_forwards": sum(c["attempted"] for c in row.get("forwards", {}).values()),
                "state_restored": row.get("state_restored"),
                "baseline_repeat_max_abs_logit_delta": row.get("baseline_repeat_max_abs_logit_delta"),
                "provider_usage": row.get("selection", {}).get("usage", {})})
        for row in case["calibration_matrix"]:
            if row.get("status") != "ok":
                continue
            readout = row["readout"]
            calibration.append({"case_id": case["case_id"], "component": row["component"],
                "raw_greedy_branch_changed": readout["raw_greedy_branch_changed"],
                "joint_success": row["joint_success"], "role_status": row["role_status"],
                "best_gap_closure_fraction": max((r["gap_closure_fraction"] for r in readout["rows"]
                                                  if r["gap_closure_fraction"] is not None), default=None),
                "output_changed_vs_b0": row["output"] != baseline.get("output")})
    aggregates = json.loads(json.dumps(report["aggregate"]))
    for policies_for_split in aggregates.values():
        for counts in policies_for_split.values():
            n, edits = counts["completed_pairs"], counts["applied_diagnostic_edits"]
            gain = counts["wins_vs_b0"] - counts["losses_vs_b0"]
            counts["joint_success_delta_vs_b0"] = gain / n if n else None
            counts["net_joint_success_gain_per_edit"] = gain / edits if edits else None
    usage = {}
    for case in report["cases"]:
        for row in case["policies"].values():
            for key, value in row.get("selection", {}).get("usage", {}).items():
                if isinstance(value, (int, float)):
                    usage[key] = usage.get(key, 0) + value
    return {"status": report["status"], "manifest_sha256": report["manifest_sha256"],
            "aggregate": aggregates, "conditions": summaries, "calibration_matrix": calibration,
            "policy_actual_forward_totals": {p: sum(sum(v["attempted"] for v in
                c["policies"].get(p, {}).get("forwards", {}).values()) for c in report["cases"]) for p in policies},
            "shared_setup_forwards": sum(sum(v["attempted"] for v in
                c.get("shared_setup_forwards", c.get("setup_forwards", {})).values()) for c in report["cases"]),
            "shared_fallback_forwards": sum(sum(v["attempted"] for v in
                c.get("shared_fallback_forwards", c.get("fallback_result", {}).get("forwards", {})).values())
                for c in report["cases"]),
            "total_provider_calls": sum(r.get("selection", {}).get("provider_calls", 0)
                for c in report["cases"] for r in c["policies"].values()),
            "provider_token_totals": usage,
            "calibration_matrix_actual_forwards": sum(sum(v["attempted"] for v in row.get("forwards", {}).values())
                for c in report["cases"] for row in c["calibration_matrix"]),
            "production_apply_allowed": False,
            "limitations": ["one worker and one predeclared boundary per case",
                "six related confirmation variants, not independent population samples",
                "role check is conservative surface analysis, not semantic ground truth",
                "one-shot recipe selection, not a multi-step adaptive controller",
                "restored diagnostic branches, not certified production applies",
                "equal ceilings with separately reported actual consumption"]}


def markdown(summary: dict) -> str:
    lines = ["# Structural Rescue Pilot", "", f"Run status: `{summary['status']}`.", "",
             f"Frozen manifest SHA-256: `{summary['manifest_sha256']}`.", "",
             "## Paired Task Outcomes", "",
             "| Split | Policy | Pairs | Eligible | Joint Success | Wins / Losses vs B0 | Output Changes | Selection Errors |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split, policies in summary["aggregate"].items():
        for policy, row in policies.items():
            eligible = row.get("eligible_pairs", "not recorded")
            lines.append(f"| {split} | {policy} | {row['completed_pairs']} | {eligible} | {row['joint_successes']} | "
                         f"{row['wins_vs_b0']} / {row['losses_vs_b0']} | {row['output_changes']} | {row['selection_errors']} |")
    lines += ["", "Joint success requires lexical constraints, a single sentence and a passing role report.",
              "Provider errors are excluded from paired efficacy estimates; noop is retained as a decision.",
              "Unavailable-boundary cases remain in overall policy outcomes, not eligible intervention comparisons.",
              "", "## Calibration Actuation", "",
              "| Case | Component | Raw Winner Changed | Full Output Changed | Joint Success | Best Gap Closure |",
              "| --- | --- | --- | --- | --- | ---: |"]
    for row in summary["calibration_matrix"]:
        gap = row["best_gap_closure_fraction"]
        lines.append(f"| {row['case_id']} | {row['component']} | {row['raw_greedy_branch_changed']} | "
                     f"{row['output_changed_vs_b0']} | {row['joint_success']} | {gap:.6f} |" if gap is not None else
                     f"| {row['case_id']} | {row['component']} | {row['raw_greedy_branch_changed']} | "
                     f"{row['output_changed_vs_b0']} | {row['joint_success']} | n/a |")
    lines += ["", "Best gap closure is the maximum among observed SOURCE structural tokens, not an ownership claim.",
              "", "## Cost", "",
              f"Provider calls: {summary['total_provider_calls']}. Shared setup forwards: {summary['shared_setup_forwards']}.",
              f"Shared no-boundary fallback forwards: {summary['shared_fallback_forwards']}.",
              f"Calibration-matrix forwards: {summary['calibration_matrix_actual_forwards']}.",
              f"Provider token totals: `{json.dumps(summary['provider_token_totals'], sort_keys=True)}`.", "",
              "| Policy | Actual Forward Attempts |", "| --- | ---: |"]
    lines += [f"| {p} | {n} |" for p, n in summary["policy_actual_forward_totals"].items()]
    lines += ["", "Per-policy ceilings: 24 diagnostic forwards plus 64 continuation forwards. "
              "Shared natural-prefix reconstruction/alignment/observation and calibration matrix are separately charged.",
              "", "## Scope", ""]
    lines += ["- " + text + "." for text in summary["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    source = args.run_dir / "results.json"
    report = json.loads(source.read_text())
    manifest_hash = hashlib.sha256((args.run_dir / "manifest.json").read_bytes()).hexdigest()
    execution = json.loads((args.run_dir / "execution_plan.json").read_text())
    if manifest_hash != report["manifest_sha256"] or execution["manifest_sha256"] != manifest_hash:
        raise ValueError("frozen_manifest_hash_mismatch")
    with tarfile.open(args.run_dir / "source_snapshot.tgz", "r:gz") as snapshot:
        for path, expected in execution["code_sha256"].items():
            relative = snapshot_member(path)
            item = snapshot.extractfile(relative)
            if item is None or hashlib.sha256(item.read()).hexdigest() != expected:
                raise ValueError("executed_source_snapshot_mismatch:" + relative)
    summary = summarize(report)
    summary["manifest_and_executed_source_verified"] = True
    summary["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    summary["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.run_dir / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.run_dir / "report.md").write_text(markdown(summary))
    print(json.dumps({"status": summary["status"], "report": str(args.run_dir / "report.md")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
