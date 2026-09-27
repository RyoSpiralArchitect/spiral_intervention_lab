"""Compare controller choices with and without an early blueprint review offer.

This fixed-prefix shadow replay never executes the selected diagnostic, applies
an edit, or advances generation after the recorded prefix.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..bridge.controller_clients import ProviderControllerClient, _compact_controller_payload
from ..controllers.factory import create_controller_provider
from ..runtime.loop import _extract_diagnostic_requests
from .digit_transform_e2e import (
    build_hooked_transformer_worker_runtime,
    create_readout_analyzer,
    create_task_env,
    load_worker_model,
)
from .replay_candidate_handoff_choice import _choice, _diff_paths, _sha
from .replay_candidate_seed_discovery import load_early_reviews
from .run_iteration_pair import PROFILES


EARLY_DIAGNOSTIC = "activation_patch_candidate_review"


def assert_early_offer_only_diff(off: Mapping[str, Any], soft: Mapping[str, Any],
                                 *, compact: bool) -> list[str]:
    paths = _diff_paths(off, soft)
    allowed = (
        "strategy_hints.candidate_handoff",
        "strategy_hints.candidate_seed_discovery_status",
        "strategy_hints.candidate_seed_discovery_blueprint_count",
        "strategy_hints.available_next_diagnostics",
    )
    if compact:
        allowed += ("source_packet_sha256", "strategy_hints.__omitted_hint_key_count")
    unexpected = [path for path in paths if not any(path == key or path.startswith(key + ".")
                  for key in allowed)]
    if unexpected:
        raise ValueError(f"off/soft packet drift outside early offer: {unexpected[:12]}")
    if not any(path.startswith("strategy_hints.candidate_seed_discovery_status") for path in paths):
        raise ValueError("soft packet did not expose early seed discovery status")
    return paths


def offered_reviews(packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    hints = packet.get("strategy_hints") or {}
    return [dict(item) for item in hints.get("available_next_diagnostics", ())
            if isinstance(item, Mapping)
            and isinstance(item.get("request"), Mapping)
            and item["request"].get("diagnostic") == EARLY_DIAGNOSTIC
            and item["request"].get("operator_recipe_expansion_mode") == EARLY_DIAGNOSTIC]


def classify_choice(choice: Mapping[str, Any], *, offered_objectives: set[str],
                    canonical_request: Mapping[str, Any] | None) -> str:
    requests = choice.get("diagnostic_requests") or ()
    if any(isinstance(request, Mapping) and request.get("diagnostic") == EARLY_DIAGNOSTIC
           and request.get("objective_bundle_key") in offered_objectives for request in requests):
        return "early_activation_patch_review"
    if canonical_request and any(isinstance(request, Mapping)
                                 and request.get("diagnostic") == canonical_request.get("diagnostic")
                                 and (not canonical_request.get("objective_bundle_key")
                                      or request.get("objective_bundle_key") == canonical_request.get("objective_bundle_key"))
                                 for request in requests):
        return "canonical_frontier"
    if requests:
        return "other_diagnostic"
    return "no_diagnostic_request"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--prefix-steps", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--controller-model", default="gpt-5.6-luna")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.prefix_steps < 1:
        parser.error("prefix-steps must be positive")
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required unless --dry-run is set")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    model_path = args.worker_model_path.expanduser().resolve(strict=True)
    source_prompt, early_reviews = load_early_reviews(
        source_path, prefix=args.prefix, objective=args.objective)
    spec = PROFILES[args.profile]

    import torch

    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(0.7)
    task = create_task_env(spec["task"])
    prompt = task.reset(spec["seed"])
    if prompt != source_prompt:
        raise ValueError("task fixture differs from the recorded episode")
    model = load_worker_model(spec["worker_model"], model_path=model_path, device=args.device,
                              dtype=spec["dtype"], hf_offline=True, mps_mode="conservative")
    worker = build_hooked_transformer_worker_runtime(
        model, task, seed=spec["seed"], activation_surface_profile="activation_patch_expanded",
        max_diagnostic_calls_per_run=12, diagnostic_result_window=12,
        candidate_handoff_mode="off", readout_sidecar_analyzer=create_readout_analyzer("sae_scaffold"),
        readout_analyzer_rerank_mode="apply")
    worker.reset(prompt)
    for _ in range(args.prefix_steps):
        worker.step()
    if worker.final_text() != args.prefix:
        raise ValueError(f"generated prefix drifted: {worker.final_text()!r}")
    worker._diagnostic_results = [dict(row) for row in early_reviews]
    worker._diagnostic_calls_used = sum(int(row.get("diagnostic_budget_charged") is True)
                                        for row in early_reviews)
    if worker._diagnostic_calls_used != 2:
        raise ValueError("expected exactly two charged predecessor diagnostics")

    off_packet = worker.build_controller_packet()
    worker.candidate_handoff_mode = "soft"
    soft_packet = worker.build_controller_packet()
    raw_diff = assert_early_offer_only_diff(off_packet, soft_packet, compact=False)
    off_compact = _compact_controller_payload(off_packet)
    soft_compact = _compact_controller_payload(soft_packet)
    compact_diff = assert_early_offer_only_diff(off_compact, soft_compact, compact=True)
    offers = offered_reviews(soft_compact)
    if not offers or args.objective not in {
        offer["request"].get("objective_bundle_key") for offer in offers
    }:
        raise ValueError("requested objective has no visible executable review offer")
    if offered_reviews(off_compact):
        raise ValueError("off packet unexpectedly contains an early review offer")
    soft_hints = soft_packet["strategy_hints"]
    off_hints = off_packet["strategy_hints"]
    canonical = soft_hints.get("diagnostic_frontier_canonical_request")
    if canonical is None and soft_hints.get("diagnostic_frontier_request"):
        canonical = {"diagnostic": soft_hints["diagnostic_frontier_request"]}
    off_canonical = off_hints.get("diagnostic_frontier_canonical_request")
    if off_canonical is None and off_hints.get("diagnostic_frontier_request"):
        off_canonical = {"diagnostic": off_hints["diagnostic_frontier_request"]}
    if canonical != off_canonical:
        raise ValueError("canonical diagnostic frontier changed across conditions")

    report: dict[str, Any] = {
        "kind": "early_blueprint_fixed_prefix_controller_choice_shadow",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile, "controller_model": args.controller_model,
        "source_jsonl": str(source_path),
        "source_jsonl_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "worker_model_path": str(model_path),
        "model_config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prefix": args.prefix, "prefix_steps": args.prefix_steps,
        "seed": spec["seed"], "worker_dtype": spec["dtype"], "device": args.device,
        "predecessor_diagnostics": [row.get("diagnostic") for row in early_reviews],
        "predecessor_result_sha256": [_sha(row) for row in early_reviews],
        "diagnostics_used": worker._diagnostic_calls_used,
        "diagnostic_budget_left": soft_packet["telemetry"]["diagnostic_call_budget_left"],
        "canonical_frontier_request": canonical,
        "soft_offered_reviews": offers,
        "packet_identity": {"off_sha256": _sha(off_compact), "soft_sha256": _sha(soft_compact),
                            "raw_diff_paths": raw_diff, "compact_diff_paths": compact_diff},
        "comparison_scope": "controller_choice_only_no_diagnostic_dispatch_no_apply_no_continuation",
        "production_apply_allowed_by_offer": False,
        "controller_call_orders": [["off", "soft"], ["soft", "off"]],
        "status": "dry_run" if args.dry_run else "in_progress",
        "cycles": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if not args.dry_run:
        provider = create_controller_provider("openai", model=args.controller_model)
        controller = ProviderControllerClient(provider, prompt_asset="controller_v01_compact.txt",
                                              packet_view="compact", max_attempts=2)
        packets = {"off": off_packet, "soft": soft_packet}
        offered_objectives = {str(offer["request"]["objective_bundle_key"]) for offer in offers}
        for order in report["controller_call_orders"]:
            cycle: dict[str, Any] = {"order": order, "conditions": {}}
            report["cycles"].append(cycle)
            for label in order:
                command = controller.invoke(packets[label])
                choice = _choice(command, packets[label], controller.latest_trace() or {})
                choice["choice_class"] = classify_choice(
                    choice, offered_objectives=offered_objectives, canonical_request=canonical)
                cycle["conditions"][label] = choice
                output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
                                  encoding="utf-8")
        report["status"] = "complete"
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
                          encoding="utf-8")
    print(json.dumps({"output": str(output), "status": report["status"],
                      "prefix": args.prefix, "offers": len(offers),
                      "raw_diff_paths": raw_diff, "compact_diff_paths": compact_diff,
                      "cycles": [{label: choice["choice_class"] for label, choice in cycle["conditions"].items()}
                                 for cycle in report["cycles"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
