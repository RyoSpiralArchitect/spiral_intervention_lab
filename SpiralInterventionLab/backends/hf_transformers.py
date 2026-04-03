from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ..runtime.codecs import TextCodec
from .base import AutoregressiveBackend, BackendCapabilities, BackendStepResult

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency at import time
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


def _preferred_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass(frozen=True)
class _TokenizerCodec:
    tokenizer: Any

    def encode(self, text: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().reshape(-1).tolist()
        return str(self.tokenizer.decode(list(token_ids), clean_up_tokenization_spaces=False))


class HFTransformersBackend(AutoregressiveBackend):
    def __init__(
        self,
        *,
        model: Any,
        codec: TextCodec | None = None,
        tokenizer: Any | None = None,
        device: str | torch.device | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        eos_token_id: int | None = None,
        output_hidden_states: bool = True,
    ) -> None:
        if codec is None and tokenizer is None:
            tokenizer = getattr(model, "tokenizer", None)
        if codec is None and tokenizer is None:
            raise ValueError("HFTransformersBackend requires either codec or tokenizer")
        self.model = model
        self.codec = codec or _TokenizerCodec(tokenizer=tokenizer)
        self.tokenizer = tokenizer
        self.device = torch.device(device) if device is not None else _preferred_torch_device()
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.eos_token_id = eos_token_id
        self.output_hidden_states = bool(output_hidden_states)
        self.capabilities = BackendCapabilities(
            backend_name="hf_transformers",
            device=str(self.device),
            supports_logits=True,
            supports_hidden_state=bool(output_hidden_states),
            supports_kv_cache=True,
            supports_prompt_hints=True,
            supports_mps=(self.device.type == "mps"),
        )

        self.model.to(self.device)
        self.model.eval()

        self._context_token_ids: list[int] = []
        self._output_token_ids: list[int] = []
        self._past_key_values: Any = None
        self._last_logits: torch.Tensor | None = None
        self._last_hidden_state: torch.Tensor | None = None
        self._eos_reached = False

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        tokenizer_name_or_path: str | None = None,
        device: str | torch.device | None = None,
        torch_dtype: Any | None = None,
        trust_remote_code: bool = False,
        attn_implementation: str | None = "sdpa",
        **kwargs: Any,
    ) -> "HFTransformersBackend":
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("transformers is not installed; install the 'hf' extra to use HFTransformersBackend")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path, trust_remote_code=trust_remote_code)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        return cls(model=model, tokenizer=tokenizer, device=device, eos_token_id=eos_token_id)

    def reset(self, prompt: str) -> None:
        self._context_token_ids = self._encode(prompt)
        if not self._context_token_ids:
            raise ValueError("prompt must encode to at least one token")
        self._output_token_ids = []
        self._past_key_values = None
        self._last_logits = None
        self._last_hidden_state = None
        self._eos_reached = False

    def step(self) -> BackendStepResult:
        input_ids = self._step_input_ids()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=self._past_key_values,
                use_cache=True,
                output_hidden_states=self.output_hidden_states,
                return_dict=True,
            )
        next_logits = outputs.logits[:, -1, :].detach()
        next_token = self._select_next_token(next_logits[0])
        self._context_token_ids.append(next_token)
        self._output_token_ids.append(next_token)
        self._past_key_values = getattr(outputs, "past_key_values", None)
        self._last_logits = next_logits[0].detach().clone()
        hidden_states = getattr(outputs, "hidden_states", None)
        self._last_hidden_state = None if not hidden_states else hidden_states[-1][0, -1].detach().clone()
        if self.eos_token_id is not None and next_token == int(self.eos_token_id):
            self._eos_reached = True
        return BackendStepResult(
            token_id=next_token,
            token_text=self.decode_tokens([next_token]),
            logits=self._last_logits,
            hidden_state=self._last_hidden_state,
        )

    def append_prompt_hint(self, hint: str) -> bool:
        hint_token_ids = self._encode(hint)
        if not hint_token_ids:
            return False
        self._context_token_ids.extend(hint_token_ids)
        self._past_key_values = None
        self._last_logits = None
        self._last_hidden_state = None
        return True

    def current_tokens(self) -> torch.Tensor:
        return torch.tensor([self._context_token_ids], dtype=torch.long)

    def output_token_ids(self) -> list[int]:
        return list(self._output_token_ids)

    def decode_tokens(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        return self.codec.decode(token_ids)

    def final_text(self) -> str:
        return self.decode_tokens(self._output_token_ids)

    def last_logits_tensor(self) -> torch.Tensor | None:
        return None if self._last_logits is None else self._last_logits.detach().clone()

    def backend_done(self) -> bool:
        return self._eos_reached

    def _encode(self, text: str) -> list[int]:
        return self.codec.encode(text).detach().reshape(-1).to(dtype=torch.long).tolist()

    def _step_input_ids(self) -> torch.Tensor:
        if self._past_key_values is None:
            token_ids = self._context_token_ids
        else:
            token_ids = [self._context_token_ids[-1]]
        return torch.tensor([token_ids], dtype=torch.long, device=self.device)

    def _select_next_token(self, logits: torch.Tensor) -> int:
        if self.temperature <= 0.0:
            return int(torch.argmax(logits).item())
        scaled = logits / max(self.temperature, 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        if self.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            keep_mask = cumulative <= self.top_p
            keep_mask[0] = True
            filtered = torch.zeros_like(probs)
            filtered[sorted_indices[keep_mask]] = probs[sorted_indices[keep_mask]]
            probs = filtered / filtered.sum().clamp_min(1e-8)
        return int(torch.multinomial(probs, num_samples=1).item())
