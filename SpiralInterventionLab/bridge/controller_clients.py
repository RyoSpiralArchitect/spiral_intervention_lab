from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping
from typing import Sequence

from ..controllers.base import ControllerProvider, ControllerProviderRequest
from ..runtime.edit_budget import estimate_edit_cost
from ..runtime.schema import ControllerCommand, parse_controller_command


def load_prompt_asset(asset_name: str) -> str:
    path = Path(__file__).resolve().parents[1] / "prompts" / asset_name
    return path.read_text(encoding="utf-8")


def _payload_for_provider(packet: Any) -> Any:
    if is_dataclass(packet):
        return asdict(packet)
    return packet


def _stable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(value)


def _stable_hash(value: Any) -> str:
    text = _stable_text(value)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truncate_text(text: Any, limit: int = 240) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    try:
        return {str(key): _json_ready(item) for key, item in vars(value).items()}
    except Exception:
        return str(value)


def _list_of_ids(items: Any, key: str) -> list[str]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return []
    values: list[str] = []
    for item in items:
        if isinstance(item, Mapping):
            value = item.get(key)
            if value is not None:
                values.append(str(value))
    return values


def _compact_controller_memory(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    summary: dict[str, Any] = {}
    for key in (
        "hypothesis",
        "micro_rationale",
        "expected_effect",
        "observed_outcome",
        "why_failed_or_helped",
        "next_change",
        "stop_condition",
        "decision",
        "recorded_step",
    ):
        if key in value and value.get(key) not in (None, ""):
            summary[key] = value.get(key)
    confidence = value.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        summary["confidence"] = float(confidence)
    return summary or None


def _controller_memory_summaries(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return []
    summaries: list[dict[str, Any]] = []
    for item in items:
        summary = _compact_controller_memory(item)
        if summary is not None:
            summaries.append(summary)
    return summaries


def _compact_observer_check(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    summary: dict[str, Any] = {}
    for key in (
        "check_type",
        "trigger",
        "verdict",
        "score",
        "raw_score",
        "coverage_signal",
        "coverage_weight",
        "delta_vs_last_check",
        "recorded_step",
    ):
        if key in value and value.get(key) not in (None, ""):
            summary[key] = value.get(key)
    return summary or None


def _observer_check_summaries(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return []
    summaries: list[dict[str, Any]] = []
    for item in items:
        summary = _compact_observer_check(item)
        if summary is not None:
            summaries.append(summary)
    return summaries


def _compact_observer_check_request(value: Any) -> dict[str, Any] | None:
    if value is True:
        return {"kind": "semantic_progress"}
    if not isinstance(value, Mapping):
        return None
    summary: dict[str, Any] = {}
    for key in ("kind", "reason", "trigger"):
        if key in value and value.get(key) not in (None, ""):
            summary[key] = value.get(key)
    return summary or None


def _recent_effect_summaries(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return []
    summaries: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        delta = item.get("delta", {})
        summary = {
            "edit_id": item.get("edit_id"),
            "surface_id": item.get("surface_id"),
            "verdict": item.get("verdict"),
        }
        if isinstance(delta, Mapping):
            summary["delta"] = {
                key: delta.get(key)
                for key in (
                    "entropy",
                    "top1_margin",
                    "repetition_score",
                    "partial_score",
                    "semantic_progress_score",
                    "required_term_span_progress",
                )
                if key in delta
            }
        summaries.append(summary)
    return summaries


def _observation_summary(payload: Any) -> dict[str, Any]:
    summary = {
        "packet_sha256": _stable_hash(payload),
        "payload_kind": type(payload).__name__,
    }
    if not isinstance(payload, Mapping):
        summary["payload_preview"] = _truncate_text(payload)
        return summary

    task_view = payload.get("task_view") if isinstance(payload.get("task_view"), Mapping) else {}
    worker_view = payload.get("worker_view") if isinstance(payload.get("worker_view"), Mapping) else {}
    summary.update(
        {
            "run_id": payload.get("run_id"),
            "episode_id": payload.get("episode_id"),
            "worker_id": payload.get("worker_id"),
            "task_id": task_view.get("task_id"),
            "step": payload.get("step"),
            "task_mode": task_view.get("mode"),
            "prompt_hash": task_view.get("prompt_hash"),
            "worker_status": worker_view.get("status"),
            "generated_tail": worker_view.get("generated_tail"),
            "telemetry": dict(payload.get("telemetry", {})) if isinstance(payload.get("telemetry"), Mapping) else {},
            "task_feedback": dict(payload.get("task_feedback", {})) if isinstance(payload.get("task_feedback"), Mapping) else {},
            "budget": dict(payload.get("budget", {})) if isinstance(payload.get("budget"), Mapping) else {},
            "surface_ids": _list_of_ids(payload.get("surface_catalog"), "surface_id"),
            "trace_ids": _list_of_ids(payload.get("trace_bank"), "trace_id"),
            "active_edit_ids": _list_of_ids(payload.get("active_edits"), "edit_id"),
            "recent_effects": _recent_effect_summaries(payload.get("recent_effects")),
            "recent_effect_summary": dict(payload.get("recent_effect_summary", {}))
            if isinstance(payload.get("recent_effect_summary"), Mapping)
            else {},
            "latest_observer_check": _compact_observer_check(payload.get("latest_observer_check")),
            "recent_observer_checks": _observer_check_summaries(payload.get("recent_observer_checks")),
            "controller_memory": _controller_memory_summaries(payload.get("controller_memory")),
        }
    )
    return summary


def _target_label(target: Any) -> str | None:
    if target is None:
        return None
    surface_id = getattr(target, "surface_id", None)
    if surface_id:
        return str(surface_id)
    kind = getattr(target, "kind", None)
    layer = getattr(target, "layer", None)
    site = getattr(target, "site", None)
    module = getattr(target, "module", None)
    suffix = module or site
    if kind and layer is not None and suffix:
        return f"{kind}:{suffix}:l{layer}"
    if kind and layer is not None:
        return f"{kind}:l{layer}"
    return None


def _decision_summary(command: ControllerCommand) -> dict[str, Any]:
    hypotheses = []
    confidences = []
    edit_ids: list[str] = []
    targets: list[str] = []
    ops: list[str] = []
    step_sizes: list[float] = []
    total_alpha = 0.0
    total_edit_cost = 0.0

    meta = getattr(command, "meta", None)
    controller_memory = None
    micro_rationale = None
    observer_check_request = None
    if isinstance(meta, Mapping):
        if meta.get("hypothesis"):
            hypotheses.append(str(meta["hypothesis"]))
        if meta.get("confidence") is not None:
            confidences.append(float(meta["confidence"]))
        controller_memory = meta.get("controller_memory")
        if meta.get("micro_rationale") is not None:
            micro_rationale = str(meta["micro_rationale"])
        observer_check_request = meta.get("observer_check_request")
    elif meta is not None:
        if getattr(meta, "hypothesis", None):
            hypotheses.append(str(meta.hypothesis))
        if getattr(meta, "confidence", None) is not None:
            confidences.append(float(meta.confidence))

    for edit in getattr(command, "edits", ()) or ():
        edit_ids.append(edit.id)
        target_label = _target_label(edit.target)
        if target_label is not None:
            targets.append(target_label)
        op_kind = edit.op.kind
        ops.append(op_kind)
        alpha = float(getattr(edit.op, "alpha", 0.0))
        total_alpha += abs(alpha)
        step_size = edit.budget.step_size
        if step_size is not None:
            step_sizes.append(float(step_size))
        total_edit_cost += estimate_edit_cost(
            op_kind=op_kind,
            alpha=alpha,
            ttl_steps=edit.budget.ttl_steps,
            step_size=step_size,
            rank_cap=edit.budget.rank_cap,
        )

    summary = {
        "decision": command.decision,
        "hypothesis": hypotheses[0] if hypotheses else None,
        "confidence": confidences[0] if confidences else None,
        "edit_ids": edit_ids,
        "rollback_ids": list(command.rollback_ids or ()),
        "targets": targets,
        "ops": ops,
        "step_sizes": step_sizes,
        "total_alpha": total_alpha,
        "total_edit_cost": total_edit_cost,
    }
    if micro_rationale:
        summary["micro_rationale"] = micro_rationale
    compact_observer_request = _compact_observer_check_request(observer_check_request)
    if compact_observer_request is not None:
        summary["observer_check_request"] = compact_observer_request
    compact_memory = _compact_controller_memory(controller_memory)
    if compact_memory is not None:
        summary["controller_memory"] = compact_memory
    return summary


def _extract_json_object(text: str) -> str:
    stripped = str(text).strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("provider response does not contain a JSON object")
    return stripped[start : end + 1]


def _normalize_controller_payload(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_controller_payload(item) for item in value]
    if not isinstance(value, dict):
        return value

    normalized = {key: _normalize_controller_payload(item) for key, item in value.items()}

    if "decision" in normalized and "version" not in normalized:
        normalized["version"] = "0.1"

    if "edit_id" in normalized and "id" not in normalized:
        normalized["id"] = normalized.pop("edit_id")

    command_like = any(key in normalized for key in ("decision", "edits", "rollback_ids"))
    meta = normalized.get("meta")
    if command_like:
        if meta is None:
            meta = {}
        if isinstance(meta, dict):
            if "controller_memory" in normalized and "controller_memory" not in meta:
                meta["controller_memory"] = normalized.pop("controller_memory")
            if "reflection_log" in normalized and "controller_memory" not in meta:
                meta["controller_memory"] = normalized.pop("reflection_log")
            if "reflection_log" in meta and "controller_memory" not in meta:
                meta["controller_memory"] = meta.pop("reflection_log")
            if "request_observer_check" in normalized and "observer_check_request" not in meta:
                meta["observer_check_request"] = normalized.pop("request_observer_check")
            if "observer_check_request" in normalized and "observer_check_request" not in meta:
                meta["observer_check_request"] = normalized.pop("observer_check_request")
            if "request_observer_check" in meta and "observer_check_request" not in meta:
                meta["observer_check_request"] = meta.pop("request_observer_check")
            if "rationale" in normalized and "micro_rationale" not in meta:
                meta["micro_rationale"] = normalized.pop("rationale")
            if "micro_rationale" in normalized and "micro_rationale" not in meta:
                meta["micro_rationale"] = normalized.pop("micro_rationale")
            if meta:
                normalized["meta"] = meta

    if "surface_id" in normalized and "target" not in normalized and any(
        key in normalized for key in ("id", "source", "op", "budget", "ttl_steps")
    ):
        normalized["target"] = {"surface_id": normalized.pop("surface_id")}

    if isinstance(normalized.get("target"), str):
        normalized["target"] = {"surface_id": normalized["target"]}
    elif isinstance(normalized.get("target"), dict) and set(normalized["target"].keys()) == {"target"}:
        inner_target = normalized["target"].get("target")
        if isinstance(inner_target, dict):
            normalized["target"] = inner_target

    fn = normalized.get("fn")
    if fn in {"add", "sub", "mean"} and "args" not in normalized:
        if "a" in normalized and "b" in normalized:
            normalized["args"] = [normalized.pop("a"), normalized.pop("b")]
        elif "left" in normalized and "right" in normalized:
            normalized["args"] = [normalized.pop("left"), normalized.pop("right")]
    elif fn in {"project_parallel", "project_orthogonal"}:
        args = normalized.get("args")
        if isinstance(args, list) and len(args) >= 2:
            normalized.setdefault("arg", args[0])
            normalized.setdefault("basis", args[1])
        if "arg" not in normalized:
            for alias in ("input", "x"):
                if alias in normalized:
                    normalized["arg"] = normalized.pop(alias)
                    break
        if "basis" not in normalized:
            for alias in ("against", "reference", "direction", "onto"):
                if alias in normalized:
                    normalized["basis"] = normalized.pop(alias)
                    break

    if "ttl_steps" in normalized and "budget" not in normalized and "op" in normalized:
        normalized["budget"] = {
            "ttl_steps": normalized.pop("ttl_steps"),
            "revertible": True,
        }

    if "scope" in normalized and "tensor" in normalized:
        scope = normalized.get("scope")
        if scope in {"best_success", "last_success", "paired_baseline"}:
            normalized["scope"] = "trace"
            normalized.setdefault("trace_id", scope)
        elif scope == "running_stats":
            normalized["scope"] = "stats"

    if normalized.get("fn") == "scale" and "by" not in normalized:
        for alias in ("alpha", "factor", "weight"):
            if alias in normalized:
                normalized["by"] = normalized.pop(alias)
                break

    token = normalized.get("token")
    if isinstance(token, dict):
        mode = token.get("mode")
        if mode in {"prev", "previous"}:
            token["mode"] = "index"
            token.setdefault("value", -2)
        elif mode == "current":
            token["mode"] = "last"

    return normalized


class ProviderControllerClient:
    def __init__(
        self,
        provider: ControllerProvider,
        *,
        system_prompt: str | None = None,
        prompt_asset: str = "controller_v01.txt",
        max_output_tokens: int = 800,
        temperature: float = 0.0,
        max_attempts: int = 2,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt or load_prompt_asset(prompt_asset)
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self.max_attempts = max(1, int(max_attempts))
        self._last_trace: dict[str, Any] | None = None

    def latest_trace(self) -> dict[str, Any] | None:
        return self._last_trace

    def invoke(self, packet: Any) -> ControllerCommand:
        payload = _payload_for_provider(packet)
        trace: dict[str, Any] = {
            "provider": self.provider.provider_name,
            "model": self.provider.model_name,
            "max_attempts": self.max_attempts,
            "system_prompt_sha256": _stable_hash(self.system_prompt),
            "observation": _observation_summary(payload),
            "attempts": [],
            "decision": None,
            "success": False,
        }
        self._last_trace = trace
        retry_note: str | None = None
        last_error: Exception | None = None
        for attempt_index in range(1, self.max_attempts + 1):
            request = ControllerProviderRequest(
                system_prompt=self.system_prompt,
                payload=payload,
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
                expect_json=True,
                retry_note=retry_note,
            )
            started = perf_counter()
            response = self.provider.complete(request)
            latency_ms = (perf_counter() - started) * 1000.0
            attempt_trace: dict[str, Any] = {
                "attempt": attempt_index,
                "latency_ms": round(latency_ms, 3),
                "request": {
                    "expect_json": True,
                    "max_output_tokens": self.max_output_tokens,
                    "temperature": self.temperature,
                    "retry_note": retry_note,
                },
                "provider": response.provider,
                "model": response.model,
                "usage": _json_ready(dict(response.usage)),
                "response_metadata": _json_ready(dict(getattr(response, "metadata", {}) or {})),
                "response_sha256": _stable_hash(response.text),
                "response_text": str(response.text),
            }
            try:
                parsed = _normalize_controller_payload(json.loads(_extract_json_object(response.text)))
                command = parse_controller_command(parsed)
                attempt_trace["parse_ok"] = True
                trace["attempts"].append(attempt_trace)
                trace["decision"] = _decision_summary(command)
                trace["success"] = True
                trace["attempt_count"] = attempt_index
                self._last_trace = trace
                return command
            except Exception as exc:
                attempt_trace["parse_ok"] = False
                attempt_trace["parse_error"] = str(exc)
                trace["attempts"].append(attempt_trace)
                self._last_trace = trace
                last_error = exc
                retry_note = f"Previous reply was invalid. Return only one compact JSON object matching ControllerCommand. Error: {exc}"
        trace["error"] = str(last_error) if last_error is not None else "unknown provider error"
        self._last_trace = trace
        raise ValueError(f"provider '{self.provider.provider_name}' failed to return valid ControllerCommand JSON") from last_error


class ProviderPromptHintController:
    def __init__(
        self,
        provider: ControllerProvider,
        *,
        system_prompt: str | None = None,
        prompt_asset: str = "prompt_hint_v01.txt",
        max_output_tokens: int = 160,
        temperature: float = 0.0,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt or load_prompt_asset(prompt_asset)
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self._last_trace: dict[str, Any] | None = None

    def latest_trace(self) -> dict[str, Any] | None:
        return self._last_trace

    def invoke(self, packet: Any) -> str | None:
        payload = _payload_for_provider(packet)
        request = ControllerProviderRequest(
            system_prompt=self.system_prompt,
            payload=payload,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            expect_json=False,
        )
        started = perf_counter()
        response = self.provider.complete(request)
        latency_ms = (perf_counter() - started) * 1000.0
        text = str(response.text).strip()
        if not text:
            self._last_trace = {
                "provider": response.provider,
                "model": response.model,
                "system_prompt_sha256": _stable_hash(self.system_prompt),
                "observation": _observation_summary(payload),
                "attempts": [
                    {
                        "attempt": 1,
                        "latency_ms": round(latency_ms, 3),
                        "request": {
                            "expect_json": False,
                            "max_output_tokens": self.max_output_tokens,
                            "temperature": self.temperature,
                            "retry_note": None,
                        },
                        "provider": response.provider,
                        "model": response.model,
                        "usage": _json_ready(dict(response.usage)),
                        "response_metadata": _json_ready(dict(getattr(response, "metadata", {}) or {})),
                        "response_sha256": _stable_hash(response.text),
                        "response_text": str(response.text),
                        "parse_ok": True,
                    }
                ],
                "decision": {"advice": None, "advice_length": 0},
                "success": True,
                "attempt_count": 1,
            }
            return None
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        self._last_trace = {
            "provider": response.provider,
            "model": response.model,
            "system_prompt_sha256": _stable_hash(self.system_prompt),
            "observation": _observation_summary(payload),
            "attempts": [
                {
                    "attempt": 1,
                    "latency_ms": round(latency_ms, 3),
                    "request": {
                        "expect_json": False,
                        "max_output_tokens": self.max_output_tokens,
                        "temperature": self.temperature,
                        "retry_note": None,
                    },
                    "provider": response.provider,
                    "model": response.model,
                    "usage": _json_ready(dict(response.usage)),
                    "response_metadata": _json_ready(dict(getattr(response, "metadata", {}) or {})),
                    "response_sha256": _stable_hash(response.text),
                    "response_text": str(response.text),
                    "parse_ok": True,
                }
            ],
            "decision": {"advice": text, "advice_length": len(text)},
            "success": True,
            "attempt_count": 1,
        }
        return text
