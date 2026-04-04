from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Mapping, Sequence


def _serialize_payload(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if is_dataclass(payload):
        payload = asdict(payload)
    if isinstance(payload, (Mapping, list, tuple)):
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray)):
        return json.dumps(list(payload), ensure_ascii=False, sort_keys=True)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class ControllerProviderRequest:
    system_prompt: str
    payload: Any
    max_output_tokens: int = 800
    temperature: float = 0.0
    expect_json: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    retry_note: str | None = None

    def payload_text(self) -> str:
        return _serialize_payload(self.payload)

    def effective_system_prompt(self) -> str:
        if not self.retry_note:
            return self.system_prompt
        return f"{self.system_prompt}\n\nRepair note: {self.retry_note}"


@dataclass(frozen=True)
class ControllerProviderResponse:
    text: str
    provider: str
    model: str
    raw: Any = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ControllerProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def complete(self, request: ControllerProviderRequest) -> ControllerProviderResponse:
        ...
