from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register
from diffusion.models.pos_emb import PosEmb

__all__ = ["HiPStructuredClassifier"]


class _Mlp(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float):
        super().__init__()
        hidden = int(d_model * float(mlp_ratio))
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc2(F.gelu(self.fc1(x)))
        return self.drop(x)


class _IntraGroupBlock(nn.Module):
    """Local attention within each group: [B,G,K,D] -> [B,G,K,D]."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = _Mlp(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, G, K, D = x.shape
        xf = x.reshape(B * G, K, D)
        xf = xf + self.attn(self.norm1(xf), self.norm1(xf), self.norm1(xf), need_weights=False)[0]
        xf = xf + self.mlp(self.norm2(xf))
        return xf.reshape(B, G, K, D)


class _InterGroupBlock(nn.Module):
    """Global exchange between groups: pool K, self-attend over G, broadcast back to tokens."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, int(num_heads), dropout=float(dropout), batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.mlp = _Mlp(d_model, mlp_ratio=mlp_ratio, dropout=dropout)
        # Broadcast starts as near-identity (zero-init) so early training is stable.
        self.broadcast = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.broadcast.weight)
        nn.init.zeros_(self.broadcast.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reps = x.mean(dim=2)  # [B,G,D]
        reps = reps + self.attn(self.norm1(reps), self.norm1(reps), self.norm1(reps), need_weights=False)[0]
        reps = reps + self.mlp(self.norm2(reps))
        return x + self.broadcast(reps).unsqueeze(2)


def _extract_group_centers(p: torch.Tensor, *, G: int, K: int, coord_dim: int) -> torch.Tensor:
    """
    Accept p as:
      - [G,d] or [B,G,d] (already group centers)
      - [G*K,d] or [B,G*K,d] (token positions repeated per group) -> take every K-th
    Return: [B,G,d]
    """
    if p.ndim == 2:
        L = int(p.shape[0])
        if L == int(G * K):
            p = p[:: int(K)].contiguous()
        elif L != int(G):
            raise ValueError(f"Expected p [G*K={G*K},d] or [G={G},d], got [L={L},d]")
        if int(p.shape[-1]) != int(coord_dim):
            raise ValueError(f"p has coord_dim={int(p.shape[-1])}, expected {coord_dim}")
        return p.unsqueeze(0)
    if p.ndim == 3:
        L = int(p.shape[1])
        if L == int(G * K):
            p = p[:, :: int(K), :].contiguous()
        elif L != int(G):
            raise ValueError(f"Expected p [B,G*K={G*K},d] or [B,G={G},d], got [B,{L},d]")
        if int(p.shape[-1]) != int(coord_dim):
            raise ValueError(f"p has coord_dim={int(p.shape[-1])}, expected {coord_dim}")
        return p
    raise ValueError(f"p must be 2D or 3D, got {p.ndim}D")


@register("hip_structured_classifier")
class HiPStructuredClassifier(nn.Module):
    """Structure-aware classifier for HiP grouped token latents ``[B,G,K,C]``.

    Alternates intra-group (local) and inter-group (global) attention blocks; optionally
    breaks within-group permutation symmetry via token-id embeddings, geometry-conditioned
    slot positions, and FiLM conditioning on ``(slot_offset, log(group_scale))``.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        num_classes: int,
        num_groups: int,
        tokens_per_group: int,
        coord_dim: int = 2,
        d_model: int = 256,
        depth: int = 8,  # total blocks, alternating intra/inter
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pos_freq: float = 1.0,
        use_within_group_token_id_emb: bool = True,
        use_within_group_slot_pos: bool = True,
        use_slot_scale_film: bool = True,
        label_smoothing: float = 0.0,
        final_dropout: float = 0.0,
        pool: str = "cls",  # cls | mean (over groups)
        latent_pool: str = "none",  # none | mean_tokens | mean_channels
    ):
        super().__init__()
        self.latent_pool = str(latent_pool).lower().strip()
        self.num_classes = int(num_classes)
        self.G = int(num_groups)
        self.K_raw = int(tokens_per_group)  # original K from manifest (for reshape)
        self.coord_dim = int(coord_dim)
        # Effective K/token_dim depend on latent_pool.
        if self.latent_pool == "mean_tokens":
            self.K = 1
            self.token_dim = int(token_dim)
        elif self.latent_pool == "mean_channels":
            self.K = int(tokens_per_group)
            self.token_dim = 1
        else:
            self.K = int(tokens_per_group)
            self.token_dim = int(token_dim)
        self.d_model = int(d_model)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout = float(dropout)
        self.pos_freq = float(pos_freq)
        self.use_within_group_token_id_emb = bool(use_within_group_token_id_emb)
        self.use_within_group_slot_pos = bool(use_within_group_slot_pos)
        self.use_slot_scale_film = bool(use_slot_scale_film)
        self.label_smoothing = float(label_smoothing)
        self.final_dropout = float(final_dropout)
        self.pool = str(pool).lower().strip()
        if self.pool not in ("cls", "mean"):
            raise ValueError("pool must be 'cls' or 'mean'")
        if self.label_smoothing < 0.0 or self.label_smoothing > 1.0:
            raise ValueError("label_smoothing must be in [0,1]")

        self.in_proj = nn.Linear(self.token_dim, self.d_model)

        self.pos_group = PosEmb(self.d_model, coord_dim=self.coord_dim, freq=self.pos_freq)
        self.pos_token = PosEmb(self.d_model, coord_dim=self.coord_dim, freq=self.pos_freq)

        self.token_id_emb = None
        if self.use_within_group_token_id_emb:
            self.token_id_emb = nn.Embedding(self.K, self.d_model)
            nn.init.normal_(self.token_id_emb.weight, std=0.02)

        self.slot_offsets = None
        self.slot_film = None
        if self.use_within_group_slot_pos:
            # slot offsets in normalized group coords (~[-1,1]).
            self.slot_offsets = nn.Parameter(torch.zeros((int(self.K), int(self.coord_dim)), dtype=torch.float32))
            with torch.no_grad():
                if int(self.coord_dim) == 2 and int(self.K) == 4:
                    init = torch.tensor([[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]], dtype=torch.float32)
                    self.slot_offsets.copy_(init)
                elif int(self.K) == 1:
                    self.slot_offsets.zero_()
                elif int(self.coord_dim) == 2:
                    ang = torch.linspace(0.0, 2.0 * math.pi, steps=int(self.K) + 1, dtype=torch.float32)[:-1]
                    init = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1) * 0.5
                    self.slot_offsets.copy_(init)
                else:
                    self.slot_offsets.normal_(mean=0.0, std=0.25)

            if self.use_slot_scale_film:
                in_dim = int(2 * self.coord_dim)
                self.slot_film = nn.Sequential(
                    nn.Linear(in_dim, int(self.d_model)),
                    nn.SiLU(),
                    nn.Linear(int(self.d_model), int(2 * self.d_model)),
                )
                nn.init.zeros_(self.slot_film[-1].weight)
                nn.init.zeros_(self.slot_film[-1].bias)

        blocks = []
        for i in range(int(self.depth)):
            if i % 2 == 0:
                blocks.append(_IntraGroupBlock(self.d_model, self.num_heads, self.mlp_ratio, self.dropout))
            else:
                blocks.append(_InterGroupBlock(self.d_model, self.num_heads, self.mlp_ratio, self.dropout))
        self.blocks = nn.ModuleList(blocks)

        self.cls = nn.Parameter(torch.zeros(1, 1, self.d_model)) if self.pool == "cls" else None
        if self.cls is not None:
            nn.init.trunc_normal_(self.cls, std=0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=int(self.d_model * self.mlp_ratio),
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Small final mixer over group reps.
        self.group_mixer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.norm = nn.LayerNorm(self.d_model)
        self.final_drop = nn.Dropout(self.final_dropout) if self.final_dropout > 0 else nn.Identity()
        self.head = nn.Linear(self.d_model, self.num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c = batch.get("c", None)
        if c is None:
            raise KeyError("batch missing 'c'")

        # Accept [B,G,K,C] (preferred) or [B,L,C] (fallback).
        if c.ndim == 3:
            B, L, Cc = c.shape
            if int(L) != int(self.G * self.K_raw):
                raise ValueError(f"Expected L=G*K_raw={self.G*self.K_raw}, got L={L}")
            c = c.view(B, int(self.G), int(self.K_raw), Cc)
        elif c.ndim == 4:
            B, G, K, Cc = c.shape
            if int(G) != int(self.G) or int(K) != int(self.K_raw):
                raise ValueError(f"Expected c [B,{self.G},{self.K_raw},C], got {tuple(c.shape)}")
        else:
            raise ValueError(f"c must be [B,L,C] or [B,G,K,C], got {tuple(c.shape)}")

        if self.latent_pool == "mean_tokens":
            c = c.mean(dim=2, keepdim=True)  # [B,G,1,C]
        elif self.latent_pool == "mean_channels":
            c = c.mean(dim=3, keepdim=True)  # [B,G,K,1]

        p = batch.get("p", None)
        if p is None:
            raise KeyError("batch missing 'p'")
        p_group = _extract_group_centers(p, G=self.G, K=self.K, coord_dim=self.coord_dim)  # [B,G,d]
        if int(p_group.shape[0]) == 1 and int(c.shape[0]) != 1:
            p_group = p_group.expand(int(c.shape[0]), -1, -1)

        x = self.in_proj(c.float())  # [B,G,K,D]

        # Group positional embedding (shared across tokens in a group).
        x = x + self.pos_group(p_group).unsqueeze(2)

        if self.token_id_emb is not None:
            tok = torch.arange(int(self.K), device=x.device, dtype=torch.long)
            x = x + self.token_id_emb(tok).view(1, 1, int(self.K), int(self.d_model))

        if self.use_within_group_slot_pos:
            gs = batch.get("group_scales", None)
            if gs is None:
                raise KeyError("use_within_group_slot_pos=true but batch missing 'group_scales'")
            if gs.ndim == 2:
                gs = gs.unsqueeze(0).expand(x.shape[0], -1, -1)
            if gs.ndim != 3 or int(gs.shape[1]) != int(self.G) or int(gs.shape[-1]) != int(self.coord_dim):
                raise ValueError(f"group_scales must be [B,G,{self.coord_dim}] or [G,{self.coord_dim}], got {tuple(gs.shape)}")
            gs = gs.to(dtype=torch.float32).clamp_min(1e-6)

            slot = self.slot_offsets.to(dtype=torch.float32)  # [K,d]
            p_eff = p_group.unsqueeze(2) + slot.view(1, 1, int(self.K), int(self.coord_dim)) * gs.unsqueeze(2)  # [B,G,K,d]
            x = x + self.pos_token(p_eff)

            if self.slot_film is not None:
                log_s = torch.log(gs)  # [B,G,d]
                sinp = torch.cat(
                    [
                        slot.view(1, 1, int(self.K), int(self.coord_dim)).expand(x.shape[0], int(self.G), -1, -1),
                        log_s.unsqueeze(2).expand(x.shape[0], int(self.G), int(self.K), -1),
                    ],
                    dim=-1,
                )  # [B,G,K,2d]
                gb = self.slot_film(sinp)  # [B,G,K,2D]
                gamma, beta = gb.chunk(2, dim=-1)
                x = x * (1.0 + gamma) + beta

        for blk in self.blocks:
            x = blk(x)

        reps = x.mean(dim=2)  # [B,G,D]
        reps = reps + self.pos_group(p_group)  # reinforce group identity for final mixer
        if self.cls is not None:
            cls = self.cls.expand(reps.shape[0], -1, -1)
            reps = torch.cat([cls, reps], dim=1)
        reps = self.group_mixer(reps)
        pooled = reps[:, 0] if self.cls is not None else reps.mean(dim=1)
        logits = self.head(self.norm(pooled))

        y = batch.get("y", None)
        if y is None:
            return {"logits": logits}
        if y.ndim != 1:
            y = y.view(-1)
        logits = self.final_drop(logits)
        loss = F.cross_entropy(logits, y, label_smoothing=float(self.label_smoothing))
        acc = (logits.argmax(dim=-1) == y).float().mean()
        return {"loss": loss, "acc": acc}

