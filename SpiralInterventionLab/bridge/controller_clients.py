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
                parsed = json.loads(_extract_json_object(response.text))
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
