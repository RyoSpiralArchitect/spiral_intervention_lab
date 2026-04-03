from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from .base import AutoregressiveBackend, BackendCapabilities, BackendStepResult

try:
    from mlx_lm import load as mlx_load
    from mlx_lm import stream_generate
except Exception:  # pragma: no cover - optional dependency at import time
    mlx_load = None  # type: ignore[assignment]
    stream_generate = None  # type: ignore[assignment]


class MLXLMBackend(AutoregressiveBackend):
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens_per_step: int = 1,
    ) -> None:
        if stream_generate is None:
            raise ImportError("mlx_lm is not installed; install the 'mlx' extra to use MLXLMBackend")
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens_per_step = int(max_tokens_per_step)
        self.capabilities = BackendCapabilities(
            backend_name="mlx_lm",
            device="mlx",
            supports_logits=True,
            supports_hidden_state=False,
            supports_kv_cache=False,
            supports_prompt_hints=True,
            supports_mlx=True,
        )

        self._context_text = ""
        self._output_token_ids: list[int] = []
        self._output_fragments: list[str] = []
        self._last_logits: torch.Tensor | None = None
        self._finish_reason: str | None = None

    @classmethod
    def from_pretrained(cls, repo_or_path: str, **kwargs: Any) -> "MLXLMBackend":
        if mlx_load is None:
            raise ImportError("mlx_lm is not installed; install the 'mlx' extra to use MLXLMBackend")
        model, tokenizer = mlx_load(repo_or_path)
        return cls(model=model, tokenizer=tokenizer, **kwargs)

    def reset(self, prompt: str) -> None:
        self._context_text = str(prompt)
        self._output_token_ids = []
        self._output_fragments = []
        self._last_logits = None
        self._finish_reason = None

    def step(self) -> BackendStepResult:
        response = next(
            stream_generate(
                self.model,
                self.tokenizer,
                prompt=self._context_text,
                max_tokens=self.max_tokens_per_step,
                temp=self.temperature,
                top_p=self.top_p,
            )
        )
        token_id = int(response.token)
        token_text = str(response.text)
        self._context_text += token_text
        self._output_token_ids.append(token_id)
        self._output_fragments.append(token_text)
        self._finish_reason = getattr(response, "finish_reason", None)
        self._last_logits = self._coerce_logprobs(getattr(response, "logprobs", None))
        return BackendStepResult(token_id=token_id, token_text=token_text, logits=self._last_logits)

    def append_prompt_hint(self, hint: str) -> bool:
        text = str(hint)
        if not text:
            return False
        self._context_text += text
        self._last_logits = None
        return True

    def current_tokens(self) -> torch.Tensor:
        token_ids = self._encode(self._context_text)
        return torch.tensor([token_ids], dtype=torch.long)

    def output_token_ids(self) -> list[int]:
        return list(self._output_token_ids)

    def decode_tokens(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().reshape(-1).tolist()
        return str(self.tokenizer.decode(list(token_ids)))

    def final_text(self) -> str:
        return "".join(self._output_fragments)

    def last_logits_tensor(self) -> torch.Tensor | None:
        return None if self._last_logits is None else self._last_logits.detach().clone()

    def backend_done(self) -> bool:
        return self._finish_reason is not None

    def _encode(self, text: str) -> list[int]:
        encoded = self.tokenizer.encode(text)
        if isinstance(encoded, torch.Tensor):
            return encoded.detach().reshape(-1).to(dtype=torch.long).tolist()
        return [int(token_id) for token_id in encoded]

    def _coerce_logprobs(self, logprobs: Any) -> torch.Tensor | None:
        if logprobs is None:
            return None
        try:
            array = np.asarray(logprobs)
        except Exception:
            return None
        if array.ndim == 0:
            return None
        return torch.from_numpy(array.astype(np.float32)).reshape(-1)
