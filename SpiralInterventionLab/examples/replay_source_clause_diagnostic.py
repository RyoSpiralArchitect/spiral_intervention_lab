"""Exercise source-clause operation cards at recorded no-edit prefixes.

This harness, not a controller model, requests each observation. Optional normal-
cap operator measurements force no token, import no rescue labels, and grant no
production permission.
A matching prefix does not reconstruct the historical diagnostic ledger.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from ..bridge.controller_clients import _payload_for_provider
from ..runtime import clause_actuation, clause_specificity, clause_structure, diagnostic_budget, operation_cards, prefix_control
from ..runtime.response_probe import state_identity
from .digit_transform_e2e import build_hooked_transformer_worker_runtime, create_task_env, load_worker_model
from .replay_orphan_blueprint_ceiling import verify_checkpoint


def source_prefixes(path: Path) -> tuple[str, dict[int, str]]:
    prompt, prefixes, finished = None, {}, False
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "episode_start":
            if prompt is not None:
                raise ValueError("source JSONL must contain exactly one episode")
            prompt = event["prompt"]
        elif event.get("event") == "controller_observation":
            step, prefix = event.get("step"), event.get("generated_tail")
            if isinstance(step, int) and not isinstance(step, bool) and isinstance(prefix, str):
                if step in prefixes and prefixes[step] != prefix:
                    raise ValueError("recorded prefix differs within one generation step")
                prefixes[step] = prefix
        elif event.get("event") == "episode_end":
            finished = True
    if prompt is None or not finished:
        raise ValueError("source episode is incomplete")
    return prompt, prefixes


def measure_current_prefix(worker: Any, *, export_controller_packets: bool = False) -> dict[str, Any]:
    before = state_identity(worker)
    step = worker._steps
    packet = prefix_control.annotate(worker.build_controller_packet(), rounds_left=2, terminal=worker.done())
    menu, _ = operation_cards.build_menu(packet)
    card = next((card for card in menu["cards"] if card["diagnostic"] == clause_structure.NAME and card["available"]), None)
    if card is None:
        raise ValueError("no executable source-clause observation offered")
    command = {"decision": "noop", "meta": {"operation_id": card["operation_id"]}}
    request = operation_cards.resolve(packet, command)
    before_packet = _payload_for_provider(packet, packet_view="full") if export_controller_packets else None
    results = worker.request_controller_diagnostics(request, source="diagnostic_harness", packet=packet)
    if len(results) != 1:
        raise ValueError("expected one diagnostic result")
    same_state = before == state_identity(worker) and step == worker._steps
    if not same_state:
        raise ValueError("structure observation changed the prefix state")
    measurement = {
        "generated_tokens": len(worker._output_token_ids()), "prefix": worker.final_text(),
        "selection_origin": "diagnostic_harness_not_controller_choice",
        "operation_card": card, "resolved_request": request, "result": results[0],
        "context_unchanged": same_state, "diagnostic_budget_left": diagnostic_budget.left(worker),
        "rollout_edit_count": len(worker._collect_active_edits()), "production_apply_allowed": False,
    }
    if export_controller_packets:
        from .compare_source_clause_choices import visibility_packets, stable_hash

        after_packet = prefix_control.annotate(
            worker.build_controller_packet(), rounds_left=2, terminal=worker.done())
        packets = visibility_packets(
            before_packet, _payload_for_provider(after_packet, packet_view="full"))
        if before != state_identity(worker) or step != worker._steps:
            raise ValueError("packet export changed the prefix state")
        measurement["controller_comparison"] = {
            "scope": "frozen_menu_visibility_then_harness_supplied_observation",
            "packets": packets,
            "packet_sha256": {name: stable_hash(value) for name, value in packets.items()},
            "observation_origin": "diagnostic_harness_not_controller_choice",
        }
    return measurement


def measure_source_operators(worker: Any, *, record: Any = None, reserve_slots: int = 0) -> list[dict[str, Any]]:
    """One pass in source order, not adaptively selected by rescue or readout wins."""
    expected = {c["structure_operator_id"] for c in clause_actuation.offer(worker)["candidates"] if c["available"]}
    if diagnostic_budget.left(worker) < len(expected) + reserve_slots:
        raise ValueError("insufficient budget for the full source comparison and reserved controls")
    receipts = []
    attempted = set()
    for _ in range(clause_structure.MAX_CANDIDATES):
        packet = prefix_control.annotate(worker.build_controller_packet(), rounds_left=2, terminal=worker.done())
        menu, _ = operation_cards.build_menu(packet)
        card = next((r for r in menu["cards"] if r["diagnostic"] == clause_actuation.NAME and r["available"]), None)
        if card is None:
            break
        request = operation_cards.resolve(packet, {"decision": "noop", "meta": {"operation_id": card["operation_id"]}})
        if request["structure_operator_id"] in attempted:
            raise ValueError("structure operator was reoffered after attempt")
        attempted.add(request["structure_operator_id"])
        results = worker.request_controller_diagnostics(request, source="diagnostic_harness", packet=packet)
        if len(results) != 1:
            raise ValueError("expected one operator diagnostic receipt")
        receipts.append({"operation_card": card, "resolved_request": request,
            "selection_origin": "source_order_diagnostic_harness_not_controller_choice", "result": results[0]})
        if record is not None:
            record(receipts[-1])
        if results[0]["status"] != "ok":
            break
    if all(r["result"]["status"] == "ok" for r in receipts) and attempted != expected:
        raise ValueError("incomplete source comparison; not all declared candidates were measured")
    return receipts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--positions", required=True, help="One to four recorded token counts, e.g. 7,8,9")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.7)
    parser.add_argument("--export-controller-packets", action="store_true",
                        help="Save frozen off/on/observed inputs at one prefix; makes no provider calls")
    parser.add_argument("--measure-source-operators", action="store_true",
                        help="At one prefix, measure source-bound activation patches within normal caps; no rollout apply")
    parser.add_argument("--source-specificity-controls", action="store_true",
                        help="With source operators, add self-copy and norm-matched radial-shrink controls; offline only")
    args = parser.parse_args(argv)
    try:
        positions = sorted({int(value) for value in args.positions.split(",")})
    except ValueError:
        parser.error("--positions must be comma-separated integers")
    if not 1 <= len(positions) <= 4 or min(positions) < 1:
        parser.error("--positions must contain one to four positive token counts")
    if args.export_controller_packets and len(positions) != 1:
        parser.error("controller packet export requires one prefix, without earlier probe history")
    if args.measure_source_operators and (len(positions) != 1 or args.export_controller_packets):
        parser.error("operator measurements require one prefix and a separate receipt from controller visibility comparison")
    if args.source_specificity_controls and not args.measure_source_operators:
        parser.error("--source-specificity-controls requires --measure-source-operators")
    if not 0 < args.mps_memory_fraction <= 1:
        parser.error("--mps-memory-fraction must be in (0, 1]")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    manifest_path = args.source_manifest.expanduser().resolve(strict=True)
    model_path = args.worker_model_path.expanduser().resolve(strict=True)
    prompt, prefixes = source_prefixes(source_path)
    if not all(position in prefixes for position in positions):
        raise ValueError("requested positions must all exist in the recorded trajectory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = manifest["profile"]
    checkpoint = verify_checkpoint(model_path, manifest.get("checkpoint_sha256") or {})
    task = create_task_env(profile["task"])
    if task.reset(int(profile["seed"])) != prompt:
        raise ValueError("task fixture differs from recorded prompt")

    import torch

    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    model = load_worker_model(
        profile["worker_model"], model_path=model_path, device=args.device,
        dtype=profile.get("dtype", "float16"), hf_offline=True, mps_mode="conservative")
    worker = build_hooked_transformer_worker_runtime(
        model, task, seed=int(profile["seed"]), clause_structure_diagnostics=True,
        clause_structure_operator_diagnostics=args.measure_source_operators,
        max_diagnostic_calls_per_run=12)
    worker.reset(prompt)
    report: dict[str, Any] = {
        "kind": "source_clause_diagnostic_replay", "schema_version": 1, "status": "in_progress",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_jsonl_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "checkpoint_sha256_verified": checkpoint, "worker_model": profile["worker_model"],
        "task": profile["task"], "seed": profile["seed"], "device": args.device,
        "dtype": profile.get("dtype"), "positions": positions, "measurements": [],
        "controller_calls": 0, "scope": "normal_cap_one_token_operator_response_not_clause_rescue"
            if args.measure_source_operators else "current_prefix_readout_not_operator_or_controller_efficacy",
        "caps": "unchanged_runtime_defaults", "production_apply_allowed": False,
        "reconstruction": "natural_no_edit_prefix_without_historical_diagnostic_ledger",
        "export_controller_packets": args.export_controller_packets,
        "measure_source_operators": args.measure_source_operators,
        "source_specificity_controls": args.source_specificity_controls,
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    save()
    try:
        for position in positions:
            while len(worker._output_token_ids()) < position:
                if worker.done():
                    raise ValueError("worker terminated before requested prefix")
                worker.step()
            if worker.final_text() != prefixes[position]:
                raise ValueError(f"prefix differs at {position}; refusing non-matched measurement")
            if args.source_specificity_controls:
                report["cache_alignment"] = clause_specificity.align_cache(worker)
                save()
            measurement = measure_current_prefix(worker, export_controller_packets=args.export_controller_packets)
            report["measurements"].append(measurement)
            save()
            print(json.dumps({"generated_tokens": position, "status": measurement["result"]["status"],
                "rows": measurement["result"].get("structure_readout_rows"),
                "context_unchanged": measurement["context_unchanged"]}, ensure_ascii=False), flush=True)
            if measurement["result"]["status"] != "ok":
                raise ValueError("source-clause observation failed; retained receipt")
            if args.measure_source_operators:
                measurement["operator_measurements"] = []
                def record_operator(receipt: dict[str, Any]) -> None:
                    measurement["operator_measurements"].append(receipt)
                    save()
                    print(json.dumps({"operator_target": receipt["operation_card"]["target_piece"],
                        "status": receipt["result"]["status"],
                        "physical_replays": receipt["result"]["physical_replay_count"],
                        "effects": [r["effect_role"] for r in receipt["result"]["operator_rows"]]}), flush=True)
                measure_source_operators(worker, record=record_operator,
                                         reserve_slots=2 if args.source_specificity_controls else 0)
                measurement["operator_offer_after"] = clause_actuation.offer(worker)
                measurement["diagnostic_budget_left_after_operators"] = diagnostic_budget.left(worker)
                save()
                rows = measurement["operator_measurements"]
                if not rows or any(r["result"]["status"] != "ok" for r in rows):
                    raise ValueError("operator measurement unavailable or failed; retained receipt")
                if args.source_specificity_controls:
                    measurement["specificity_controls"] = []
                    for kind in clause_specificity.CONTROL_KINDS:
                        control = clause_specificity.measure_control(worker, kind)
                        measurement["specificity_controls"].append(control)
                        save()
                        print(json.dumps({"specificity_control": kind, "status": control["status"],
                            "physical_replays": control["physical_replay_count"]}), flush=True)
                        if control["status"] != "ok":
                            raise ValueError("specificity control failed; retained receipt")
                    measurement["specificity_comparison"] = clause_specificity.compare(
                        rows, measurement["specificity_controls"])
                    measurement["diagnostic_budget_left_after_controls"] = diagnostic_budget.left(worker)
                    if measurement["specificity_comparison"]["status"] != "ok":
                        raise ValueError("specificity controls not comparable; retained receipt")
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="error", error=str(exc))
        raise
    finally:
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
