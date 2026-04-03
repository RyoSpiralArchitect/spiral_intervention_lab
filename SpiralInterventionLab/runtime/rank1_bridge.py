from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _normalize_if_nonzero(x: torch.Tensor) -> torch.Tensor:
    norm = x.norm()
    if float(norm) <= 1e-8:
        return x
    return x / norm


def _resample_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.numel() == target_len:
        return x
    view = x.reshape(1, 1, -1)
    mode = "linear" if x.numel() > 1 and target_len > 1 else "nearest"
    resized = F.interpolate(view, size=target_len, mode=mode, align_corners=False if mode == "linear" else None)
    return resized.reshape(-1)


@dataclass(frozen=True)
class Rank1Geometry:
    target_shape: tuple[int, ...]
    rows: int
    cols: int
    matrix: torch.Tensor
    surface_id: str | None = None
    site: str | None = None


class Rank1VectorBridge:
    def adapt(self, raw: torch.Tensor, *, side: str, geometry: Rank1Geometry) -> torch.Tensor:
        raise NotImplementedError


class HybridRank1VectorBridge(Rank1VectorBridge):
    """
    Adapter layer for rank-1 patches.

    Strategy order:
    1. Exact size match: pass through.
    2. Parameter-aware lift:
       - row side: matrix @ raw when raw matches cols
       - col side: matrix.T @ raw when raw matches rows
    3. Deterministic 1D resample fallback.
    """

    def adapt(self, raw: torch.Tensor, *, side: str, geometry: Rank1Geometry) -> torch.Tensor:
        flat = raw.reshape(-1).to(device=geometry.matrix.device, dtype=geometry.matrix.dtype)
        target_len = geometry.rows if side == "row" else geometry.cols
        other_len = geometry.cols if side == "row" else geometry.rows

        if flat.numel() == target_len:
            return flat

        if flat.numel() == other_len:
            projected = geometry.matrix @ flat if side == "row" else geometry.matrix.transpose(0, 1) @ flat
            return _normalize_if_nonzero(projected)

        return _normalize_if_nonzero(_resample_1d(flat, target_len))
