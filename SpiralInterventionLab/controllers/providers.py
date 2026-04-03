from __future__ import annotations

from typing import Any, Mapping

from .base import ControllerProvider, ControllerProviderRequest, ControllerProviderResponse

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at import time
    OpenAI = None  # type: ignore[assignment]

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - optional dependency at import time
    Anthropic = None  # type: ignore[assignment]

try:
    from mistralai import Mistral
except Exception:  # pragma: no cover - optional dependency at import time
    Mistral = None  # type: ignore[assignment]

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency at import time
    genai = None  # type: ignore[assignment]


def _usage_dict(raw: Any) -> Mapping[str, Any]:
    usage = getattr(raw, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, Mapping):
        return dict(usage)
    try:
        return dict(vars(usage))
    except Exception:
        return {}


def _anthropic_text(raw: Any) -> str:
    blocks = getattr(raw, "content", None) or []
    texts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if text:
            texts.append(str(text))
    return "".join(texts).strip()


def _mistral_text(raw: Any) -> str:
    choices = getattr(raw, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
                continue
            text = getattr(item, "text", None)
            if text:
                texts.append(str(text))
        return "".join(texts).strip()
    return str(content).strip()


def _google_text(raw: Any) -> str:
    text = getattr(raw, "text", None)
    if text:
        return str(text).strip()
    candidates = getattr(raw, "candidates", None) or []
    chunks: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(str(part_text))
    return "".join(chunks).strip()


class OpenAIControllerProvider(ControllerProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
        base_url: str | None = None,
    ) -> None:
        if client is None:
            if OpenAI is None:
                raise ImportError("openai is not installed; install the 'controllers' extra to use OpenAIControllerProvider")
            client = OpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self._model_name = model

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, request: ControllerProviderRequest) -> ControllerProviderResponse:
        text_config: dict[str, Any] = {"verbosity": "low"}
        if request.expect_json:
            text_config["format"] = {"type": "json_object"}
        payload_text = request.payload_text()
        if request.expect_json:
            payload_text = f"JSON packet:\n{payload_text}"
        raw = self.client.responses.create(
            model=self.model_name,
            instructions=request.effective_system_prompt(),
            input=payload_text,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            metadata=dict(request.metadata),
            text=text_config,
        )
        text = getattr(raw, "output_text", None) or ""
        return ControllerProviderResponse(
            text=str(text).strip(),
            provider=self.provider_name,
            model=self.model_name,
            raw=raw,
            usage=_usage_dict(raw),
        )


class AnthropicControllerProvider(ControllerProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if Anthropic is None:
                raise ImportError("anthropic is not installed; install the 'controllers' extra to use AnthropicControllerProvider")
            client = Anthropic(api_key=api_key)
        self.client = client
        self._model_name = model

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, request: ControllerProviderRequest) -> ControllerProviderResponse:
        raw = self.client.messages.create(
            model=self.model_name,
            max_tokens=request.max_output_tokens,
            temperature=request.temperature,
            system=request.effective_system_prompt(),
            messages=[{"role": "user", "content": request.payload_text()}],
        )
        return ControllerProviderResponse(
            text=_anthropic_text(raw),
            provider=self.provider_name,
            model=self.model_name,
            raw=raw,
            usage=_usage_dict(raw),
        )


class MistralControllerProvider(ControllerProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if Mistral is None:
                raise ImportError("mistralai is not installed; install the 'controllers' extra to use MistralControllerProvider")
            client = Mistral(api_key=api_key)
        self.client = client
        self._model_name = model

    @property
    def provider_name(self) -> str:
        return "mistral"

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, request: ControllerProviderRequest) -> ControllerProviderResponse:
        raw = self.client.chat.complete(
            model=self.model_name,
            messages=[
                {"role": "system", "content": request.effective_system_prompt()},
                {"role": "user", "content": request.payload_text()},
            ],
            temperature=request.temperature,
            max_tokens=request.max_output_tokens,
        )
        return ControllerProviderResponse(
            text=_mistral_text(raw),
            provider=self.provider_name,
            model=self.model_name,
            raw=raw,
            usage=_usage_dict(raw),
        )


class GoogleGenAIControllerProvider(ControllerProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            if genai is None:
                raise ImportError("google-genai is not installed; install the 'controllers' extra to use GoogleGenAIControllerProvider")
            client = genai.Client(api_key=api_key)
        self.client = client
        self._model_name = model

    @property
    def provider_name(self) -> str:
        return "google"

    @property
    def model_name(self) -> str:
        return self._model_name

    def complete(self, request: ControllerProviderRequest) -> ControllerProviderResponse:
        config = genai.types.GenerateContentConfig(
            systemInstruction=request.effective_system_prompt(),
            temperature=request.temperature,
            maxOutputTokens=request.max_output_tokens,
            responseMimeType="application/json" if request.expect_json else "text/plain",
        )
        raw = self.client.models.generate_content(
            model=self.model_name,
            contents=request.payload_text(),
            config=config,
        )
        return ControllerProviderResponse(
            text=_google_text(raw),
            provider=self.provider_name,
            model=self.model_name,
            raw=raw,
            usage=_usage_dict(raw),
        )
