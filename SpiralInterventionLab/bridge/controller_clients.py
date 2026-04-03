from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ..controllers.base import ControllerProvider, ControllerProviderRequest
from ..runtime.schema import ControllerCommand, parse_controller_command


def load_prompt_asset(asset_name: str) -> str:
    path = Path(__file__).resolve().parents[1] / "prompts" / asset_name
    return path.read_text(encoding="utf-8")


def _payload_for_provider(packet: Any) -> Any:
    if is_dataclass(packet):
        return asdict(packet)
    return packet


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

    if "ttl_steps" in normalized and "budget" not in normalized and "op" in normalized:
        normalized["budget"] = {
            "ttl_steps": normalized.pop("ttl_steps"),
            "revertible": True,
        }

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

    def invoke(self, packet: Any) -> ControllerCommand:
        payload = _payload_for_provider(packet)
        retry_note: str | None = None
        last_error: Exception | None = None
        for _attempt in range(self.max_attempts):
            response = self.provider.complete(
                ControllerProviderRequest(
                    system_prompt=self.system_prompt,
                    payload=payload,
                    max_output_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    expect_json=True,
                    retry_note=retry_note,
                )
            )
            try:
                parsed = _normalize_controller_payload(json.loads(_extract_json_object(response.text)))
                return parse_controller_command(parsed)
            except Exception as exc:
                last_error = exc
                retry_note = f"Previous reply was invalid. Return only one compact JSON object matching ControllerCommand. Error: {exc}"
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

    def invoke(self, packet: Any) -> str | None:
        response = self.provider.complete(
            ControllerProviderRequest(
                system_prompt=self.system_prompt,
                payload=_payload_for_provider(packet),
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
                expect_json=False,
            )
        )
        text = str(response.text).strip()
        if not text:
            return None
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text
