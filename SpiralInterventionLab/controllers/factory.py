from __future__ import annotations

import os
from typing import Any

from .providers import (
    AnthropicControllerProvider,
    GoogleGenAIControllerProvider,
    MistralControllerProvider,
    OpenAIControllerProvider,
)

PROVIDER_API_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "google": "GOOGLE_API_KEY",
}


def normalize_provider_name(name: str) -> str:
    value = str(name).strip().lower().replace("-", "_")
    aliases = {
        "claude": "anthropic",
        "gemini": "google",
        "google_genai": "google",
    }
    return aliases.get(value, value)


def provider_api_env_var(name: str) -> str:
    normalized = normalize_provider_name(name)
    try:
        return PROVIDER_API_ENV_VARS[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown controller provider '{name}'") from exc


def create_controller_provider(
    provider_name: str,
    *,
    model: str,
    api_key: str | None = None,
    **kwargs: Any,
):
    normalized = normalize_provider_name(provider_name)
    resolved_api_key = api_key or os.getenv(provider_api_env_var(normalized))

    if normalized == "openai":
        return OpenAIControllerProvider(model=model, api_key=resolved_api_key, **kwargs)
    if normalized == "anthropic":
        return AnthropicControllerProvider(model=model, api_key=resolved_api_key, **kwargs)
    if normalized == "mistral":
        return MistralControllerProvider(model=model, api_key=resolved_api_key, **kwargs)
    if normalized == "google":
        return GoogleGenAIControllerProvider(model=model, api_key=resolved_api_key, **kwargs)
    raise ValueError(f"unknown controller provider '{provider_name}'")
