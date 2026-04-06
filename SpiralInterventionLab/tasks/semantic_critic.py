from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F


class SemanticCritic(Protocol):
    mode: str
    model_name: str

    def score(self, *, reference_text: str, candidate_text: str) -> float:
        ...


@contextmanager
def _temporarily_enable_hf_downloads():
    offline_keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HUGGINGFACE_HUB_OFFLINE")
    previous_values = {key: os.environ.get(key) for key in offline_keys}
    hub_constants = None
    previous_hub_offline = None
    transformers_hub = None
    previous_is_offline_mode = None
    try:
        for key in offline_keys:
            os.environ[key] = "0"
        try:
            import huggingface_hub.constants as hub_constants

            previous_hub_offline = getattr(hub_constants, "HF_HUB_OFFLINE", None)
            hub_constants.HF_HUB_OFFLINE = False
        except Exception:
            hub_constants = None
        try:
            import transformers.utils.hub as transformers_hub

            previous_is_offline_mode = getattr(transformers_hub, "is_offline_mode", None)
            transformers_hub.is_offline_mode = lambda: False
        except Exception:
            transformers_hub = None
        yield
    finally:
        if hub_constants is not None and previous_hub_offline is not None:
            hub_constants.HF_HUB_OFFLINE = previous_hub_offline
        if transformers_hub is not None and previous_is_offline_mode is not None:
            transformers_hub.is_offline_mode = previous_is_offline_mode
        for key, value in previous_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass
class MiniLMSemanticCritic:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "cpu"
    normalize_to_unit_interval: bool = True
    _model: object | None = field(default=None, init=False, repr=False)
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _reference_cache: dict[str, torch.Tensor] = field(default_factory=dict, init=False, repr=False)

    mode: str = field(default="minilm", init=False)

    def score(self, *, reference_text: str, candidate_text: str) -> float:
        reference = " ".join(str(reference_text).split()).strip()
        candidate = " ".join(str(candidate_text).split()).strip()
        if not reference or not candidate:
            return 0.0
        reference_embedding = self._cached_embedding(reference)
        candidate_embedding = self._encode_text(candidate)
        similarity = torch.nn.functional.cosine_similarity(reference_embedding, candidate_embedding, dim=0)
        value = float(similarity.item())
        if self.normalize_to_unit_interval:
            value = 0.5 * (value + 1.0)
        return max(0.0, min(1.0, value))

    def _cached_embedding(self, text: str) -> torch.Tensor:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = self._reference_cache.get(key)
        if cached is not None:
            return cached
        encoded = self._encode_text(text)
        self._reference_cache[key] = encoded
        return encoded

    def _encode_text(self, text: str) -> torch.Tensor:
        tokenizer, model = self._load_components()
        encoded = tokenizer(
            [text],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        device = torch.device(self.device)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        token_embeddings = outputs.last_hidden_state
        attention_mask = (
            encoded["attention_mask"]
            .to(token_embeddings.device)
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1e-9)
        normalized = F.normalize(pooled, p=2, dim=1)
        return normalized[0].detach().cpu().to(dtype=torch.float32)

    def _load_components(self):
        if self._model is None or self._tokenizer is None:
            from transformers import AutoModel, AutoTokenizer

            with _temporarily_enable_hf_downloads():
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=False,
                )
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    local_files_only=False,
                )
            self._model.to(torch.device(self.device))
            self._model.eval()
        return self._tokenizer, self._model
