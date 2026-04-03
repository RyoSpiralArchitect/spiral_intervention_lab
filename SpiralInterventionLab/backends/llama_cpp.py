from __future__ import annotations

from typing import Any, Sequence

import torch

from .base import AutoregressiveBackend, BackendCapabilities, BackendStepResult

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - optional dependency at import time
    Llama = None  # type: ignore[assignment]


class LlamaCppBackend(AutoregressiveBackend):
    def __init__(
        self,
        *,
        model: Any,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        if Llama is None:
            raise ImportError("llama_cpp is not installed; install the 'llama' extra to use LlamaCppBackend")
        self.model = model
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.capabilities = BackendCapabilities(
            backend_name="llama_cpp",
            device="cpu",
            supports_logits=False,
            supports_hidden_state=False,
            supports_kv_cache=False,
            supports_prompt_hints=True,
        )
        self._context_text = ""
        self._output_token_ids: list[int] = []
        self._output_fragments: list[str] = []
        self._last_logits: torch.Tensor | None = None
        self._finish_reason: str | None = None

    @classmethod
    def from_model_path(cls, model_path: str, **kwargs: Any) -> "LlamaCppBackend":
        if Llama is None:
            raise ImportError("llama_cpp is not installed; install the 'llama' extra to use LlamaCppBackend")
        model = Llama(model_path=model_path, logits_all=True)
        return cls(model=model, **kwargs)

    def reset(self, prompt: str) -> None:
        self._context_text = str(prompt)
        self._output_token_ids = []
        self._output_fragments = []
        self._last_logits = None
        self._finish_reason = None

    def step(self) -> BackendStepResult:
        completion = self.model.create_completion(
            prompt=self._context_text,
            max_tokens=1,
            temperature=self.temperature,
            top_p=self.top_p,
            echo=False,
        )
        choice = completion["choices"][0]
        token_text = str(choice.get("text", ""))
        self._context_text += token_text
        self._output_fragments.append(token_text)
        token_ids = self.model.tokenize(token_text.encode("utf-8"), add_bos=False, special=False)
        token_id = int(token_ids[0]) if token_ids else -1
        if token_id >= 0:
            self._output_token_ids.append(token_id)
        self._finish_reason = choice.get("finish_reason")
        return BackendStepResult(token_id=token_id, token_text=token_text, logits=None)

    def append_prompt_hint(self, hint: str) -> bool:
        text = str(hint)
        if not text:
            return False
        self._context_text += text
        return True

    def current_tokens(self) -> torch.Tensor:
        token_ids = self.model.tokenize(self._context_text.encode("utf-8"), add_bos=False, special=False)
        return torch.tensor([list(token_ids)], dtype=torch.long)

    def output_token_ids(self) -> list[int]:
        return list(self._output_token_ids)

    def decode_tokens(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().reshape(-1).tolist()
        return self.model.detokenize(list(token_ids)).decode("utf-8", errors="replace")

    def final_text(self) -> str:
        return "".join(self._output_fragments)

    def last_logits_tensor(self) -> torch.Tensor | None:
        return None if self._last_logits is None else self._last_logits.detach().clone()

    def backend_done(self) -> bool:
        return self._finish_reason is not None
