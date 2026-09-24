"""Compare controller choices at one fixed prefix with a recorded seed available early.

This is a counterfactual, controller-only shadow replay. It never dispatches a
diagnostic, compiles an edit, or advances generation after the chosen prefix.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..bridge.controller_clients import ProviderControllerClient, _compact_controller_payload
from ..controllers.factory import create_controller_provider
from ..runtime import candidate_handoff
from ..runtime.loop import _build_controller_selection_report, _extract_diagnostic_requests
from .digit_transform_e2e import (
    build_hooked_transformer_worker_runtime,
    create_readout_analyzer,
    create_task_env,
    load_worker_model,
)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def _diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return [path for key in sorted(set(left) | set(right))
                for path in _diff_paths(left.get(key), right.get(key),
                                        f"{prefix}.{key}" if prefix else str(key))
                ]
    return [] if left == right else [prefix]


def _assert_handoff_only_diff(off: Mapping[str, Any], soft: Mapping[str, Any],
                              *, compact: bool) -> list[str]:
    paths = _diff_paths(off, soft)
    allowed = ("strategy_hints.candidate_handoff",
               "strategy_hints.available_next_diagnostics")
    if compact:
        allowed += ("source_packet_sha256", "strategy_hints.__omitted_hint_key_count")
    unexpected = [path for path in paths if not any(path == key or path.startswith(key + ".")
                  for key in allowed)]
    if unexpected:
        raise ValueError(f"off/soft packet drift outside handoff: {unexpected[:12]}")
    if not any(path.startswith("strategy_hints.candidate_handoff") for path in paths):
        raise ValueError("soft packet did not expose candidate_handoff")
    return paths


def _load_recorded_seed(path: Path, recipe_id: str, prefix: str) -> tuple[str, dict[str, Any]]:
    prompt: str | None = None
    matches: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") == "episode_start":
                prompt = str(event["prompt"])
            if event.get("event") != "controller_diagnostic_result":
                continue
            for row in event.get("evidence_rows", ()):
                if isinstance(row, Mapping) and row.get("operator_recipe_id") == recipe_id:
                    matches.append(dict(row))
    if prompt is None or len(matches) != 1:
        raise ValueError(f"expected one episode prompt and one recorded seed, got {len(matches)} seeds")
    row = matches[0]
    binding = row.get("target_piece_binding_manifest")
    if not isinstance(binding, Mapping) or binding.get("answer_prefix_tail") != prefix:
        raise ValueError("recorded seed was measured at a different answer prefix")
    if row.get("activation_patch_site") not in {"resid_pre", "resid_post", "mlp_out"}:
        raise ValueError("recorded seed is not an activation-patch preflight row")
    return prompt, row


def _choice(command: Any, packet: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    requests = _extract_diagnostic_requests(command, packet)
    selection = _build_controller_selection_report(packet, command)
    parsed = asdict(command)
    return {
        "command": parsed,
        "diagnostic_requests": requests,
        "selection": {key: selection.get(key) for key in (
            "candidate_handoff_state", "candidate_handoff_choice",
            "candidate_handoff_defer_reason", "candidate_handoff_defer_reason_source",
            "controller_selection_source", "controller_selected_bundle_key")},
        "provider_trace": trace,
        "shadow_only": True,
        "diagnostic_dispatched": False,
        "edit_compiled": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--recipe-id", required=True)
    parser.add_argument("--prefix", default=" In the case of")
    parser.add_argument("--prefix-steps", type=int, default=4)
    parser.add_argument("--diagnostics-used", type=int, default=7)
    parser.add_argument("--max-diagnostics", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--controller-model", default="gpt-5.6-luna")
    parser.add_argument("--order", choices=("off,soft", "soft,off"), default="off,soft")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.diagnostics_used < args.max_diagnostics:
        parser.error("diagnostics-used must leave at least one diagnostic slot")
    if args.prefix_steps < 1:
        parser.error("prefix-steps must be positive")
    if not args.dry_run and not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is required unless --dry-run is set")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    model_path = args.worker_model_path.expanduser().resolve(strict=True)
    recorded_prompt, seed_row = _load_recorded_seed(source_path, args.recipe_id, args.prefix)

    import torch

    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(0.7)
    task = create_task_env("constrained_rewrite")
    prompt = task.reset(args.seed)
    if prompt != recorded_prompt:
        raise ValueError("task fixture differs from the recorded episode")
    model = load_worker_model("gpt2", model_path=model_path, device=args.device,
                              dtype="float32", hf_offline=True, mps_mode="conservative")
    worker = build_hooked_transformer_worker_runtime(
        model, task, seed=args.seed, activation_surface_profile="activation_patch_expanded",
        max_diagnostic_calls_per_run=args.max_diagnostics, diagnostic_result_window=args.max_diagnostics,
        candidate_handoff_mode="off", readout_sidecar_analyzer=create_readout_analyzer("sae_scaffold"),
        readout_analyzer_rerank_mode="apply")
    worker.reset(prompt)
    for _ in range(args.prefix_steps):
        worker.step()
    if worker.final_text() != args.prefix:
        raise ValueError(f"generated prefix drifted: {worker.final_text()!r}")

    # The row was observed at this prefix in the source run, but its earlier
    # availability and the remaining five slots are deliberately counterfactual.
    worker._diagnostic_calls_used = args.diagnostics_used
    candidate_handoff.capture_seed_rows(worker, {"evidence_rows": [seed_row]})
    if not worker._candidate_handoff_seed_rows:
        raise ValueError("recorded row did not pass seed preflight")
    off_packet = worker.build_controller_packet()
    worker.candidate_handoff_mode = "soft"
    soft_packet = worker.build_controller_packet()
    raw_diff = _assert_handoff_only_diff(off_packet, soft_packet, compact=False)
    off_compact = _compact_controller_payload(off_packet)
    soft_compact = _compact_controller_payload(soft_packet)
    compact_diff = _assert_handoff_only_diff(off_compact, soft_compact, compact=True)
    handoff = soft_packet["strategy_hints"]["candidate_handoff"]
    if handoff["state"] != "measurable" or len(handoff["measurement_offers"]) != 1:
        raise ValueError(f"expected exactly one measurable offer: {handoff}")
    if handoff["measurement_offers"][0]["seed_recipe_id"] != args.recipe_id:
        raise ValueError("offer does not reference the recorded recipe")

    result: dict[str, Any] = {
        "kind": "counterfactual_fixed_prefix_controller_choice_shadow",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_jsonl_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_jsonl": str(source_path), "worker_model_path": str(model_path),
        "seed": args.seed, "prefix": args.prefix, "prefix_steps": args.prefix_steps,
        "controller_model": args.controller_model, "controller_call_order": args.order.split(","),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "model_config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
        "diagnostics_used_counterfactual": args.diagnostics_used,
        "diagnostic_budget_left": args.max_diagnostics - args.diagnostics_used,
        "recorded_seed": {"row_sha256": _sha(seed_row), "operator_recipe_id": args.recipe_id,
                          "objective_bundle_key": seed_row.get("objective_bundle_key"),
                          "target_piece": seed_row.get("target_piece"),
                          "target_piece_binding_id": seed_row.get("target_piece_binding_id")},
        "handoff": handoff,
        "packet_identity": {"off_sha256": _sha(off_compact), "soft_sha256": _sha(soft_compact),
                            "raw_diff_paths": raw_diff, "compact_diff_paths": compact_diff},
        "comparison_scope": "controller_choice_only_no_physical_replay_no_apply_no_generation_outcome",
        "status": "dry_run" if args.dry_run else "in_progress",
        "conditions": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if not args.dry_run:
        provider = create_controller_provider("openai", model=args.controller_model)
        controller = ProviderControllerClient(provider, prompt_asset="controller_v01_compact.txt",
                                              packet_view="compact", max_attempts=2)
        packets = {"off": off_packet, "soft": soft_packet}
        for label in args.order.split(","):
            packet = packets[label]
            command = controller.invoke(packet)
            result["conditions"][label] = _choice(command, packet, controller.latest_trace() or {})
            output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        result["status"] = "complete"
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "dry_run": args.dry_run,
                      "prefix": args.prefix, "offer_state": handoff["state"],
                      "raw_diff_paths": raw_diff, "compact_diff_paths": compact_diff,
                      "choices": {key: value["selection"] for key, value in result["conditions"].items()}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
