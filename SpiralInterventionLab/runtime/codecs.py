from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch


class TextCodec(Protocol):
    def encode(self, text: str) -> torch.Tensor:
        ...

    def decode(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        ...


@dataclass(frozen=True)
class CharacterCodec:
    alphabet: str
    unknown_token_id: int = 0

    def __post_init__(self) -> None:
        if not self.alphabet:
            raise ValueError("alphabet must not be empty")
        mapping = {char: idx for idx, char in enumerate(self.alphabet)}
        object.__setattr__(self, "_encode_map", mapping)

    def encode(self, text: str) -> torch.Tensor:
        token_ids = [self._encode_map.get(char, self.unknown_token_id) for char in text]
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            flat = token_ids.detach().reshape(-1).tolist()
        else:
            flat = list(token_ids)
        return "".join(self.alphabet[int(token_id) % len(self.alphabet)] for token_id in flat)


@dataclass(frozen=True)
class ModelTokenizerCodec:
    model: Any
    prepend_bos: bool = False

    def __post_init__(self) -> None:
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("model does not expose a tokenizer; pass an explicit codec instead")

    def encode(self, text: str) -> torch.Tensor:
        tokens = self.model.to_tokens(text, prepend_bos=self.prepend_bos)
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError(f"expected model.to_tokens to return shape [1, seq], got {tuple(tokens.shape)}")
        return tokens[0].detach().clone().to(dtype=torch.long)

    def decode(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            tensor = token_ids.detach().clone().to(dtype=torch.long)
        else:
            tensor = torch.tensor(list(token_ids), dtype=torch.long)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return str(self.model.to_string(tensor))


def resolve_text_codec(model: Any, codec: TextCodec | None = None) -> TextCodec:
    if codec is not None:
        return codec
    return ModelTokenizerCodec(model=model)
