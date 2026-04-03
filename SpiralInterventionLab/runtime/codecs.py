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
            flat = token_ids.detach().reshape(-1).to(dtype=torch.long).tolist()
        else:
            flat = [int(token_id) for token_id in token_ids]
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            return str(tokenizer.decode(flat, clean_up_tokenization_spaces=False))

        tensor = torch.tensor(flat, dtype=torch.long).unsqueeze(0)
        decoded = self.model.to_string(tensor)
        if isinstance(decoded, (list, tuple)):
            if len(decoded) == 1:
                return str(decoded[0])
            return "".join(str(item) for item in decoded)
        return str(decoded)


def resolve_text_codec(model: Any, codec: TextCodec | None = None) -> TextCodec:
    if codec is not None:
        return codec
    return ModelTokenizerCodec(model=model)
