"""Compare legacy and operation-card interfaces at one reconstructed prefix.

Optional dispatch executes each distinct controller-selected diagnostic once in
an independently reset context. It never applies a rollout edit or continues
generation past that prefix. The interface ablation changes both presentation
and prompt, not just option visibility.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from ..bridge.controller_clients import ProviderControllerClient
from ..controllers.factory import create_controller_provider
from ..runtime import candidate_handoff, diagnostic_budget, operation_cards, prefix_control
from ..runtime.loop import _extract_diagnostic_requests
from ..runtime.response_probe import state_identity
from .replay_candidate_handoff_choice import _sha
from .replay_candidate_seed_discovery_choice import reconstruct_early_worker, restore_early_context
from .run_iteration_pair import PROFILES


def dispatch_selected(worker: Any, requests: list[dict[str, Any]], *, packet: dict[str, Any]) -> dict[str, Any]:
    """Count real simulation calls without inferring them from summary row counts."""
    if len(requests) != 1:
        return {"status": "not_dispatched", "reason": "requires_one_diagnostic_request"}
    before = state_identity(worker)
    budget_before = diagnostic_budget.left(worker)
    original = worker._simulate_decode
    calls = 0
    def counted(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)
    worker._simulate_decode = counted
    start = perf_counter()
    try:
        results = worker.request_controller_diagnostics(requests, source="controller", packet=packet)
    finally:
        worker._simulate_decode = original
    return {"status": "executed" if results else "no_result", "results": results,
            "elapsed_seconds": perf_counter() - start,
            "simulate_decode_call_count": calls,
            "diagnostic_slots_charged": budget_before - diagnostic_budget.left(worker),
            "diagnostic_budget_left": diagnostic_budget.left(worker),
            "frozen_card_count": len(worker._frozen_diagnostic_candidates),
            "handoff_state": candidate_handoff.report(worker)["state"],
            "context_unchanged": before == state_identity(worker),
            "rollout_edit_count": len(worker._collect_active_edits()),
            "production_apply_allowed": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--prefix-steps", type=int, default=1)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--controller-model", default="gpt-5.6-luna")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dispatch-selected", action="store_true")
    args = parser.parse_args(argv)
    if args.prefix_steps < 1:
        parser.error("prefix-steps must be positive")
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required for live controller comparison")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source = args.source_jsonl.expanduser().resolve(strict=True)
    model_path = args.worker_model_path.expanduser().resolve(strict=True)
    worker, prompt, early = reconstruct_early_worker(source_path=source, model_path=model_path,
        profile=args.profile, prefix=args.prefix, prefix_steps=args.prefix_steps,
        objective=args.objective, device=args.device)
    worker.candidate_handoff_mode = "soft"
    packet = prefix_control.annotate(worker.build_controller_packet(), rounds_left=2, terminal=worker.done())
    context = state_identity(worker)
    menu, offered = operation_cards.build_menu(packet)
    if not menu["cards"]:
        raise ValueError("no operations offered at the reconstructed prefix")
    report: dict[str, Any] = {
        "kind": "fixed_prefix_controller_operation_interface_comparison", "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "dry_run" if args.dry_run else "in_progress", "profile": args.profile,
        "worker_model_path": str(model_path), "device": args.device,
        "model_config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
        "controller_model": args.controller_model, "seed": PROFILES[args.profile]["seed"],
        "source_jsonl_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "predecessor_result_sha256": [_sha(row) for row in early],
        "prefix": args.prefix, "packet_sha256": _sha(packet), "context_id": context,
        "comparison_scope": "interface_and_prompt_ablation_same_evidence_and_offered_operations",
        "operation_menu": menu, "offered_requests": list(offered.values()),
        "diagnostic_budget_left": diagnostic_budget.left(worker),
        "cycles": [], "executions": {}, "production_apply_allowed": False,
    }
    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
    save()
    if not args.dry_run:
        provider = create_controller_provider("openai", model=args.controller_model)
        clients = {view: ProviderControllerClient(provider, prompt_asset="controller_v01_compact.txt",
            packet_view="compact", action_view=view, max_attempts=2) for view in ("legacy", "cards")}
        for order in (("legacy", "cards"), ("cards", "legacy")):
            cycle: dict[str, Any] = {"order": order, "conditions": {}}
            report["cycles"].append(cycle)
            for view in order:
                command = clients[view].invoke(packet)
                requests = _extract_diagnostic_requests(command, packet)
                cycle["conditions"][view] = {"command": asdict(command), "requests": requests,
                    "request_identity": _sha(requests), "trace": clients[view].latest_trace()}
                if _sha(packet) != report["packet_sha256"] or state_identity(worker) != context:
                    raise ValueError("controller comparison mutated the frozen packet or worker context")
                save()
        if args.dispatch_selected:
            for cycle in report["cycles"]:
                for choice in cycle["conditions"].values():
                    key = choice["request_identity"]
                    if key in report["executions"]:
                        continue
                    restore_early_context(worker, prompt=prompt, prefix=args.prefix,
                        prefix_steps=args.prefix_steps, early_reviews=early)
                    execution_packet = prefix_control.annotate(worker.build_controller_packet(),
                        rounds_left=2, terminal=worker.done())
                    if state_identity(worker) != context:
                        raise ValueError("diagnostic dispatch did not restore the comparison context")
                    if operation_cards.build_menu(execution_packet)[1] != offered:
                        raise ValueError("diagnostic dispatch offers drifted from controller comparison")
                    report["executions"][key] = dispatch_selected(worker, choice["requests"], packet=execution_packet)
                    if not report["executions"][key].get("context_unchanged", True):
                        report["status"] = "context_drift"
                        save()
                        raise ValueError("selected diagnostic changed the frozen generation context")
                    save()
        report["status"] = "complete"
        save()
    print(json.dumps({"output": str(output), "status": report["status"],
        "choices": [{v: row["requests"] for v, row in cycle["conditions"].items()}
                    for cycle in report["cycles"]],
        "executions": {k: {field: row.get(field) for field in (
            "status", "simulate_decode_call_count", "frozen_card_count", "diagnostic_slots_charged",
            "context_unchanged", "rollout_edit_count")} for k, row in report["executions"].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
