from __future__ import annotations

import math
import torch
import torch.nn as nn

__all__ = ["PosEmb"]


class PosEmb(nn.Module):
    """
    Lightweight positional embedding for continuous coordinates.

    Input:  coords [..., d] in (typically) [-1, 1]
    Output: emb   [..., D]
    """

    def __init__(self, embedding_dim: int, *, coord_dim: int, freq: float = 1.0):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.coord_dim = int(coord_dim)
        self.freq = float(freq)

        if self.coord_dim <= 0:
            raise ValueError(f"coord_dim must be > 0, got {self.coord_dim}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {self.embedding_dim}")

        # IMPORTANT: do NOT lazily create parameters in forward().
        # This repo constructs optimizers immediately after model construction.
        # Lazy parameters would not be included in the optimizer (and are unsafe with DDP).
        self._emb_layer = nn.Linear(self.coord_dim, self.embedding_dim // 2, bias=False)
        nn.init.normal_(self._emb_layer.weight, mean=0.0, std=self.freq)

        in_features = self.coord_dim + self.embedding_dim
        self._out_layer = nn.Linear(in_features, self.embedding_dim, bias=True)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if int(coords.shape[-1]) != int(self.coord_dim):
            raise ValueError(f"coords[...,d] has d={int(coords.shape[-1])} but PosEmb expects coord_dim={self.coord_dim}")

        projected = self._emb_layer(math.pi * (coords + 1.0))
        concat = torch.cat([coords, projected, projected + (math.pi / 2.0)], dim=-1)
        x = torch.sin(concat)
        return self._out_layer(x)

