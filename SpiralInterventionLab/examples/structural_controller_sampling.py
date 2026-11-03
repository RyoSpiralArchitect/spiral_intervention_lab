"""Provider-only controller choices on the packets of a first-token table run.

Three arms see each case. ``controller`` gets the v2 prompt template at the run
cap and the v2 packet; ``controller_mechanics`` adds a paragraph of actuator
mechanics that hold by construction; ``controller_datasheet`` also gets packet
v3, whose recipes carry a calibration datasheet. Each arm is asked k times per
case. Every response is persisted before the next call, and a rerun replays the
recorded responses instead of asking again. No model forward runs here: each
choice is scored later by lookup in the table.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from ..controllers.base import ControllerProviderRequest
from ..controllers.factory import create_controller_provider
from ..runtime import reach_datasheet
from ..runtime.response_probe import identity
from . import run_hygiene


VERSION = "structural_controller_sampling_v1"
ARMS = ("controller", "controller_mechanics", "controller_datasheet")
MAX_OUTPUT_TOKENS = 1600
RETRY_SECONDS = 5

MECHANICS = """
How the edit acts. It changes the next token only. If the next token stays the same, the whole
continuation is identical to noop, so a push that does not change the next token costs nothing.
Pushing a target that is already the next token cannot change it. A target that is sentence-final
punctuation ends the answer, because the worker stops at its first sentence-final punctuation.
In calibration, every push that changed the next token made its own target the next token.
"""

DATASHEET = """
Each recipe also carries a datasheet measured by this harness on calibration prompts. For a readout
push, closure_logits (n, min, median, max) is how much the same push at this cap reduced its target's
gap below the top token on calibration prompts whose top token was of the same class (top1_class:
source_token when the top token is itself an eligible source token, otherwise other). reach compares
it with target_gap_to_top1: likely (gap below min), borderline (between min and max), unlikely
(at or above max), already_top, or uncalibrated (fewer than 5 calibration pushes). ends_output marks a
target that ends the answer. For a blend, first_token_changes of n calibration blends changed the
next token at all, at the cap stated.
"""


def plain_prompt(cap: float) -> str:
    """The v2 controller prompt, byte for byte at cap 0.08."""
    return f"""Select at most one bounded internal intervention recipe at this current worker prefix.
Return strict JSON only: {{"decision":"noop"}} or {{"decision":"probe","recipe_id":"<offered ID>"}}.
You receive generated text, next-token readout, and a catalog of source-bound recipes. Each recipe is
one edit for one token: a readout push toward one source punctuation or connective token, or a blend
toward the states of two source structural tokens. Every write is clipped to {cap:g} of the
edited residual norm, and names and required terms are never targets.
Choose a recipe only if it plausibly helps the worker preserve independent clauses. You may choose noop.
Do not output an answer, a token to force, new expressions, or any other fields.
This selects one restored diagnostic continuation, not production permission.
Role annotations, reference continuations and candidate outcomes are unavailable.
"""


def prompt(arm: str, cap: float) -> str:
    return plain_prompt(cap) + {"controller": "", "controller_mechanics": MECHANICS,
                                "controller_datasheet": MECHANICS + DATASHEET}[arm]


def opaque_id(case_id: str) -> str:
    return hashlib.sha256(case_id.encode()).hexdigest()[:12]


def arm_packet(arm: str, packet: dict, case: dict, sheet: dict | None, *, opaque: bool) -> dict:
    result = deepcopy(packet)
    if opaque:
        result["case_id"] = opaque_id(case["case_id"])
    if arm == "controller_datasheet":
        klass = reach_datasheet.top1_class(case)
        result["version"] = "structural_rescue_choice_v3_datasheet"
        for offered in result["recipes"]:
            offered["datasheet"] = reach_datasheet.recipe_entry(offered, case["structure_readout"], sheet,
                                                                case_top1_class=klass)
        result["datasheet"] = {"version": sheet["version"], "cap": sheet["cap"], "min_n": sheet["min_n"],
                               "source": "calibration_prompts", "blend_cap_measured": sheet["patch"]["cap_measured"]}
    return result


def parse(text: str, offered: set[str]) -> tuple[str, str | None]:
    command = json.loads(text)
    if command == {"decision": "noop"}:
        return "noop", None
    if set(command) != {"decision", "recipe_id"} or command["decision"] != "probe" or command["recipe_id"] not in offered:
        raise ValueError("controller_choice_outside_offered_schema")
    return "selected", command["recipe_id"]


def transient(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    return status == 429 or (isinstance(status, int) and status >= 500) or name in {
        "APITimeoutError", "APIConnectionError", "Timeout", "ReadTimeout", "ConnectionError"}


def ask(provider: Any, system: str, packet: dict) -> dict:
    """One choice; one retry when no response arrived. A still-failing call is recorded as an error (ITT: noop)."""
    offered = {r["recipe_id"] for r in packet["recipes"]}
    attempts = []
    for attempt in range(2):
        started = time.monotonic()
        try:
            response = provider.complete(ControllerProviderRequest(system_prompt=system, payload=packet,
                                                                   max_output_tokens=MAX_OUTPUT_TOKENS,
                                                                   expect_json=True))
        except Exception as exc:
            # Provider errors may carry credential fragments; keep safe metadata only.
            attempts.append({"error_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None),
                             "latency_seconds": round(time.monotonic() - started, 3)})
            if attempt == 0 and transient(exc):
                time.sleep(RETRY_SECONDS)
                continue
            return {"status": "error", "recipe_id": None, "attempts": attempts}
        record = {"response": response.text, "usage": dict(response.usage), "metadata": dict(response.metadata),
                  "resolved_model": getattr(response.raw, "model", None),
                  "latency_seconds": round(time.monotonic() - started, 3), "attempts": attempts}
        try:
            status, recipe_id = parse(response.text, offered)
        except Exception as exc:
            return {**record, "status": "invalid", "recipe_id": None, "error_type": type(exc).__name__}
        return {**record, "status": status, "recipe_id": recipe_id}
    return {"status": "error", "recipe_id": None, "attempts": attempts}


def table_cases(table_dir: Path) -> tuple[dict, list[dict], dict]:
    manifest_bytes = (table_dir / "manifest.json").read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    cases, parts = [], {}
    for part in sorted(p for p in table_dir.iterdir() if p.is_dir() and p.name.startswith("part_")):
        raw = (part / "results.json").read_bytes()
        report = json.loads(raw)
        if report["manifest_sha256"] != digest or report["status"] != "complete":
            continue
        parts[part.name] = hashlib.sha256(raw).hexdigest()
        cases.extend(report["cases"])
    cases.sort(key=lambda c: c["pool_index"])
    return json.loads(manifest_bytes), cases, {"table_manifest_sha256": digest, "table_parts": parts}


def sheets_for(path: Path, cap: float) -> tuple[dict, str]:
    """One sheet for every case, or a map from case ID to its leave-one-out sheet."""
    raw = path.read_bytes()
    data = json.loads(raw)
    sheets = {"*": data} if "version" in data else data
    for sheet in sheets.values():
        reach_datasheet.verify(sheet, cap=cap)
    return sheets, hashlib.sha256(raw).hexdigest()


def manifest(*, table: dict, cap: float, k: int, arms: list[str], provider: str, model: str, opaque: bool,
             datasheet_sha256: str) -> dict[str, Any]:
    return {"version": VERSION, "table": table, "cap": cap, "samples_per_arm": k, "arms": arms,
            "provider": provider, "model": model, "opaque_case_ids": opaque, "datasheet_file_sha256": datasheet_sha256,
            "system_prompts": {arm: prompt(arm, cap) for arm in arms},
            "system_prompt_sha256": {arm: hashlib.sha256(prompt(arm, cap).encode()).hexdigest() for arm in arms},
            "request_settings": {"max_output_tokens": MAX_OUTPUT_TOKENS, "json": True,
                                 "openai_gpt5": {"reasoning_effort": "low", "verbosity": "low", "temperature": None}},
            "retry_rule": "one_retry_only_when_no_response_arrived_timeout_connection_429_5xx",
            "missing_data_rule": "intention_to_treat_error_or_invalid_choice_counts_as_noop",
            "replay_rule": "rerun_reuses_recorded_responses_keyed_by_case_arm_sample_packet_prompt",
            "production_apply_allowed": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--table-dir", type=Path, required=True)
    parser.add_argument("--datasheets", type=Path, required=True)
    parser.add_argument("--cap", type=float, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--opaque-case-ids", action="store_true")
    parser.add_argument("--controller-provider", choices=("openai",), default="openai")
    parser.add_argument("--controller-model", default="gpt-5.6-luna")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--replay-from", type=Path, help="an earlier results.json of this manifest")
    args = parser.parse_args(argv)
    arms = args.arms.split(",")
    if set(arms) - set(ARMS) or len(set(arms)) != len(arms) or not 1 <= args.samples <= 10:
        raise ValueError("unknown or duplicate arm, or samples outside 1..10")
    table_manifest, cases, table = table_cases(args.table_dir.expanduser().resolve())
    if args.cap not in table_manifest["packet_caps"]:
        raise ValueError("the table stored no packets at this cap")
    sheets, sheet_sha = sheets_for(args.datasheets, args.cap)
    frozen = (json.dumps(manifest(table=table, cap=args.cap, k=args.samples, arms=arms,
                                  provider=args.controller_provider, model=args.controller_model,
                                  opaque=args.opaque_case_ids, datasheet_sha256=sheet_sha),
                         indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(frozen).hexdigest()
    output = args.output_dir.expanduser().resolve()
    if args.prepare:
        output.mkdir(parents=True, exist_ok=False)
        (output / "manifest.json").write_bytes(frozen)
        print(json.dumps({"manifest_sha256": digest, "output_dir": str(output), "cases": len(cases)}))
        return 0
    if args.manifest_sha256 != digest or (output / "manifest.json").read_bytes() != frozen:
        raise ValueError("predeclared_manifest_hash_required_and_must_match")
    result_path = output / "results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    replay = {}
    if args.replay_from:
        for row in json.loads(args.replay_from.read_text())["choices"]:
            if row["status"] in ("selected", "noop", "invalid"):
                replay[(row["case_id"], row["arm"], row["sample"], row["packet_sha256"],
                        row["system_prompt_sha256"])] = row
    from openai import OpenAI
    provider = create_controller_provider(args.controller_provider, model=args.controller_model,
                                          client=OpenAI(timeout=90, max_retries=0))
    report = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": digest, "provenance": run_hygiene.provenance(), "choices": []}

    def persist():
        temp = result_path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temp.replace(result_path)

    persist()
    resolved = set()
    try:
        for case in cases:
            packet = case.get("packets", {}).get(f"{args.cap:.2f}")
            if case.get("status") != "ok" or packet is None:
                continue
            sheet = sheets.get(case["case_id"], sheets.get("*"))
            for sample in range(args.samples):
                for arm in arms:
                    shown = arm_packet(arm, packet, case, sheet, opaque=args.opaque_case_ids)
                    system = prompt(arm, args.cap)
                    key = (case["case_id"], arm, sample, identity("packet:", shown),
                           hashlib.sha256(system.encode()).hexdigest())
                    row = {"case_id": case["case_id"], "arm": arm, "sample": sample, "packet_sha256": key[3],
                           "system_prompt_sha256": key[4]}
                    if key in replay:
                        row.update({k: v for k, v in replay[key].items() if k not in row}, replayed=True)
                    else:
                        row.update(ask(provider, system, shown), replayed=False)
                    if row.get("resolved_model"):
                        resolved.add(row["resolved_model"])
                        if len(resolved) > 1:
                            raise RuntimeError("resolved_controller_model_changed_within_run")
                    report["choices"].append(row)
                    persist()
                    print(json.dumps({k: row.get(k) for k in ("case_id", "arm", "sample", "status", "recipe_id")}),
                          flush=True)
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc)[:300])
        persist()
        raise
    report["resolved_models"] = sorted(resolved)
    persist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
