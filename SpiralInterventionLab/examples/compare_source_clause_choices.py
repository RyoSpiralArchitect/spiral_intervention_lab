"""Bounded, non-dispatching controller choices from a frozen source-clause replay.

Off/on differ only in the source-clause offer and its explanatory hint.
Observed is a separate harness-supplied-evidence condition, not an interactive
follow-up to a controller request. No local model is loaded by this command.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from ..bridge.controller_clients import ProviderControllerClient, _payload_for_provider
from ..runtime import clause_structure, operation_cards
from ..runtime.loop import _extract_diagnostic_requests


NAME = clause_structure.NAME
SCHEDULE = ((1, "off"), (1, "offered"), (2, "offered"), (2, "off"),
            (1, "observed"), (2, "observed"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def without_source_offer(packet: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(packet)
    hints = result.get("strategy_hints") or {}
    hints.pop("source_clause_structure", None)
    for field in ("available_next_diagnostics", "blocked_next_diagnostics"):
        if field in hints:
            hints[field] = [row for row in hints[field]
                            if row.get("diagnostic") != NAME
                            and (row.get("request") or {}).get("diagnostic") != NAME]
    return result


def visibility_packets(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    packets = {"off": without_source_offer(before), "offered": deepcopy(before),
               "observed": deepcopy(after)}
    for condition in ("off", "offered"):
        serialized = json.dumps(packets[condition], ensure_ascii=False)
        if "structure_readout_rows" in serialized:
            raise ValueError("pre-observation condition contains source-clause measurements")
    off_menu, off_requests = operation_cards.build_menu(packets["off"])
    on_menu, on_requests = operation_cards.build_menu(packets["offered"])
    source_cards = [card for card in on_menu["cards"] if card["diagnostic"] == NAME]
    if len(source_cards) != 1 or not source_cards[0]["available"]:
        raise ValueError("expected one executable source-clause offer")
    if list(off_requests.values()) != [
        req for req in on_requests.values() if req.get("diagnostic") != NAME
    ]:
        raise ValueError("old diagnostic requests changed")
    # IDs bind the whole packet, so visibility changes legitimately rotate them.
    # All other old-card content, including order, cost and availability, must match.
    def old_cards(menu: dict[str, Any]) -> list[dict[str, Any]]:
        return [{k: v for k, v in card.items() if k != "operation_id"}
                for card in menu["cards"] if card["diagnostic"] != NAME]
    if old_cards(off_menu) != old_cards(on_menu):
        raise ValueError("old operation cards changed")
    after_menu, _ = operation_cards.build_menu(packets["observed"])
    if any(card["diagnostic"] == NAME and card["available"] for card in after_menu["cards"]):
        raise ValueError("observed condition still offers the consumed source-clause probe")
    return packets


def load_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = json.loads(path.read_text(encoding="utf-8"))
    if replay.get("kind") != "source_clause_diagnostic_replay" or replay.get("status") != "complete":
        raise ValueError("requires a complete source-clause replay")
    measurements = replay.get("measurements") or []
    if len(measurements) != 1:
        raise ValueError("comparison requires one independently measured prefix")
    row = measurements[0]
    if row.get("context_unchanged") is not True or row.get("rollout_edit_count") != 0:
        raise ValueError("measurement is not a restored no-edit observation")
    result = row.get("result") or {}
    if result.get("status") != "ok" or result.get("state_restored") is not True:
        raise ValueError("source-clause result is not valid")
    comparison = row["controller_comparison"]
    packets = comparison["packets"]
    if set(packets) != {"off", "offered", "observed"}:
        raise ValueError("unexpected comparison conditions")
    for condition, packet in packets.items():
        if stable_hash(packet) != comparison["packet_sha256"].get(condition):
            raise ValueError("snapshot hash mismatch")
    expected = visibility_packets(packets["offered"], packets["observed"])
    if packets != expected:
        raise ValueError("off/on condition contains unrelated changes")
    return row, packets


def prepare_report(path: Path, model: str) -> dict[str, Any]:
    row, packets = load_snapshot(path)
    inputs = {}
    for condition, packet in packets.items():
        payload = operation_cards.project_payload(
            _payload_for_provider(packet, packet_view="compact"), packet)
        client = ProviderControllerClient(None, action_view="cards", packet_view="compact", max_attempts=1)
        inputs[condition] = {
            "packet_sha256": stable_hash(packet), "provider_payload": payload,
            "provider_payload_sha256": stable_hash(payload),
            "system_prompt_sha256": stable_hash(client.system_prompt),
            "provider_payload_chars": len(json.dumps(payload, ensure_ascii=False)),
        }
    return {
        "kind": "source_clause_controller_visibility_comparison", "schema_version": 1,
        "status": "prepared", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "generated_tokens": row["generated_tokens"], "prefix": row["prefix"],
        "controller_model": model, "inputs": inputs,
        "scope": "interface_choice_not_operator_or_task_efficacy",
        "observed_condition": "harness_supplied_evidence_not_controller_requested_followup",
        "operation_ids": "context_bound_ids_rotate_across_conditions; old_request_semantics_fixed",
        "sampling": "two_choices_per_condition_not_a_statistical_effect_estimate",
        "schedule": [{"repeat": repeat, "condition": condition} for repeat, condition in SCHEDULE],
        "provider_call_limit": len(SCHEDULE), "provider_calls_attempted": 0,
        "choices": [], "diagnostics_dispatched": 0, "rollout_edits_applied": 0,
        "production_apply_executed": False,
    }


def run_choices(report: dict[str, Any], packets: dict[str, Any], provider: Any, save: Any) -> None:
    report["status"] = "in_progress"
    save()
    for repeat, condition in SCHEDULE:
        packet = packets[condition]
        expected_hash = report["inputs"][condition]["packet_sha256"]
        client = ProviderControllerClient(provider, action_view="cards", packet_view="compact",
                                          max_attempts=1)
        choice: dict[str, Any] = {"repeat": repeat, "condition": condition, "status": "started"}
        report["choices"].append(choice)
        report["provider_calls_attempted"] += 1
        save()
        try:
            command = client.invoke(packet)
            requests = _extract_diagnostic_requests(command, packet)
            choice.update(status="valid", command=asdict(command), requests=requests,
                          source_clause_selected=any(req.get("diagnostic") == NAME for req in requests),
                          command_would_apply=command.decision == "apply")
        except ValueError as exc:
            choice.update(status="invalid_command", error_type=type(exc).__name__)
        except Exception as exc:
            choice.update(status="provider_error", error_type=type(exc).__name__,
                          status_code=getattr(exc, "status_code", None))
            report["status"] = "provider_error"
            raise
        finally:
            choice["trace"] = client.latest_trace()
            choice["packet_unchanged"] = stable_hash(packet) == expected_hash
            if not choice["packet_unchanged"]:
                report["status"] = "packet_drift"
            save()
        if not choice["packet_unchanged"]:
            raise ValueError("controller invocation changed the frozen packet")
    report["status"] = "complete"
    save()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controller-model", default="gpt-5.6-luna")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    source = args.source_receipt.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    report = prepare_report(source, args.controller_model)
    _, packets = load_snapshot(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                          encoding="utf-8")

    save()
    if args.dry_run:
        report["status"] = "dry_run"
        save()
    else:
        from openai import OpenAI
        from ..controllers.providers import OpenAIControllerProvider

        provider = OpenAIControllerProvider(
            model=args.controller_model, client=OpenAI(max_retries=0, timeout=180))
        run_choices(report, packets, provider, save)
    print(json.dumps({"output": str(output), "status": report["status"],
                     "provider_calls_attempted": report["provider_calls_attempted"],
                     "choices": [{k: row.get(k) for k in
                                  ("repeat", "condition", "status", "requests", "source_clause_selected")}
                                 for row in report["choices"]]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
