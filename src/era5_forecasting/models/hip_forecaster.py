"""HiP-structured forecaster for temporal prediction on grouped latents.

Predicts the residual delta_z = z_{t+1} - z_t.
"""

from __future__ import annotations

import math
from math import sqrt
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

import models
from models import register


def regroup(inputs: torch.Tensor, num_output_groups: int) -> torch.Tensor:
    """Re-group [B, G, N, C] to [B, G', N', C] where G*N = G'*N'."""
    B, G_in, N_in, C = inputs.shape
    N_out = G_in * N_in // num_output_groups
    return inputs.reshape(B, num_output_groups, N_out, C)


class SqReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x) ** 2


class FeedForward(nn.Module):
    def __init__(self, dim: int, widening_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = dim * widening_factor
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), SqReLU(),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GroupedAttention(nn.Module):
    """Multi-head attention within groups. Operates on [B, G, N, C]."""

    def __init__(self, dim: int, heads: int = 4, qk_dim: Optional[int] = None,
                 v_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        qk_dim = qk_dim or dim
        v_dim = v_dim or qk_dim
        self.heads = heads
        self.qk_head_dim = qk_dim // heads
        self.v_head_dim = v_dim // heads
        self.scale = self.qk_head_dim ** -0.5
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, v_dim, bias=False)
        self.to_out = nn.Linear(v_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        ctx = context if context is not None else x
        h = self.heads
        q = rearrange(self.to_q(x), "b g n (h d) -> b g h n d", h=h, d=self.qk_head_dim)
        k = rearrange(self.to_k(ctx), "b g m (h d) -> b g h m d", h=h, d=self.qk_head_dim)
        v = rearrange(self.to_v(ctx), "b g m (h d) -> b g h m d", h=h, d=self.v_head_dim)
        sim = torch.einsum("b g h n d, b g h m d -> b g h n m", q, k) * self.scale
        attn = self.dropout(sim.softmax(dim=-1))
        out = torch.einsum("b g h n m, b g h m d -> b g h n d", attn, v)
        return self.to_out(rearrange(out, "b g h n d -> b g n (h d)"))


class SelfAttentionBlock(nn.Module):
    """Within-group self-attention + FFN with pre-norm (no adaLN — no timestep)."""

    def __init__(self, dim: int, heads: int = 4, widening_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = GroupedAttention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, widening_factor=widening_factor, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention with learnable queries for regrouping (HiP-style)."""

    def __init__(self, input_dim: int, output_dim: int, num_output_groups: int,
                 output_tokens_per_group: int, heads: int = 1,
                 widening_factor: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_output_groups = num_output_groups
        self.output_tokens_per_group = output_tokens_per_group

        self.latent_queries = nn.Parameter(
            torch.empty(num_output_groups * output_tokens_per_group, output_dim)
        )
        nn.init.trunc_normal_(self.latent_queries, std=1.0 / sqrt(output_dim))

        self.norm_q = nn.LayerNorm(output_dim)
        self.norm_kv = nn.LayerNorm(input_dim)
        self.attn = GroupedAttention(output_dim, heads=heads, qk_dim=input_dim,
                                     v_dim=input_dim, dropout=dropout)
        self.out_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm_ff = nn.LayerNorm(output_dim)
        self.ff = FeedForward(output_dim, widening_factor=widening_factor, dropout=dropout)

    def forward(self, inputs: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = inputs.shape[0]
        G_out = self.num_output_groups
        N_out = self.output_tokens_per_group

        queries = repeat(self.latent_queries, "l d -> b l d", b=B)
        queries = rearrange(queries, "b (g n) d -> b g n d", g=G_out)
        if skip is not None:
            queries = queries + skip

        inputs_regrouped = regroup(inputs, G_out)

        q = self.norm_q(queries)
        kv = self.norm_kv(inputs_regrouped)
        out = queries + self.out_proj(self.attn(q, kv))
        out = out + self.ff(self.norm_ff(out))
        return out


class HiPBlock(nn.Module):
    """Cross-attention (regroup) followed by self-attention layers."""

    def __init__(self, input_dim: int, output_dim: int, num_output_groups: int,
                 output_tokens_per_group: int, num_self_attend_layers: int = 1,
                 self_attend_heads: int = 4, widening_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(
            input_dim, output_dim, num_output_groups, output_tokens_per_group,
            heads=1, widening_factor=1, dropout=dropout,
        )
        self.self_attns = nn.ModuleList([
            SelfAttentionBlock(output_dim, heads=self_attend_heads,
                               widening_factor=widening_factor, dropout=dropout)
            for _ in range(num_self_attend_layers)
        ])

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.cross_attn(x, skip=skip)
        for sa in self.self_attns:
            x = sa(x)
        return x


class FourierPosEmb(nn.Module):
    def __init__(self, d_model: int, coord_dim: int = 3, max_freq: float = 10.0):
        super().__init__()
        n_bands = max(1, d_model // (2 * coord_dim))
        freqs = torch.linspace(1.0, max_freq, n_bands)
        self.register_buffer("freqs", freqs)
        self.coord_dim = coord_dim
        self.proj = nn.Linear(2 * coord_dim * n_bands, d_model)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        p_scaled = p.unsqueeze(-1) * self.freqs
        pe = torch.cat([p_scaled.sin(), p_scaled.cos()], dim=-1)
        pe = rearrange(pe, "... d f -> ... (d f)")
        return self.proj(pe)


@register("hip_forecaster")
class HiPForecaster(nn.Module):
    """HiP-structured U-Net forecaster: z_t → delta_z.

    U-Net group schedule: G → G/4 → ... → 1 (processor) → ... → G/4 → G,
    with skip connections from encoder to decoder.
    """

    def __init__(
        self,
        *,
        num_groups: int = 36,
        tokens_per_group: int = 4,
        token_dim: int = 16,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        depth: int = 3,
        self_attend_layers: int = 2,
        num_heads: int = 4,
        widening_factor: int = 4,
        dropout: float = 0.0,
        pos_max_freq: float = 10.0,
        use_within_group_token_id_emb: bool = True,
        use_within_group_slot_pos: bool = False,
        predict_residual: bool = True,
    ):
        super().__init__()
        G = int(num_groups)
        K = int(tokens_per_group)
        C = int(token_dim)
        D = int(hidden_dim)
        self.G = G
        self.K = K
        self.C = C
        self.D = D
        self.predict_residual = bool(predict_residual)
        self.use_within_group_slot_pos = bool(use_within_group_slot_pos)
        self.coord_dim = int(coord_dim)
        total_tokens = G * K

        self.input_proj = nn.Linear(C, D)
        self.pos_emb = FourierPosEmb(D, coord_dim, pos_max_freq)

        self.within_group_id_emb = None
        if use_within_group_token_id_emb:
            self.within_group_id_emb = nn.Embedding(K, D)

        # Geometry-conditioned slot offsets (only used when slot_pos enabled)
        self.slot_offsets = None
        if use_within_group_slot_pos:
            self.slot_offsets = nn.Parameter(torch.zeros(K, coord_dim))
            with torch.no_grad():
                if K == 1:
                    self.slot_offsets.zero_()
                else:
                    self.slot_offsets.normal_(mean=0.0, std=0.25)

        # Group schedule: G → G/4 → ... → 1 (processor) → ... → G/4 → G
        enc_groups = [G]
        g = G
        for _ in range(depth - 1):
            g = max(1, g // 4)
            enc_groups.append(g)
        enc_groups.append(1)

        dec_groups = list(reversed(enc_groups[:-1]))
        self.group_schedule = enc_groups + dec_groups
        self.processor_idx = len(enc_groups) - 1

        self.encoder = nn.ModuleList()
        for i in range(len(enc_groups)):
            input_g = enc_groups[i - 1] if i > 0 else G
            output_g = enc_groups[i]
            input_k = total_tokens // input_g
            output_k = total_tokens // output_g
            n_sa = self_attend_layers if i == self.processor_idx else 1
            self.encoder.append(HiPBlock(
                D, D, output_g, output_k,
                num_self_attend_layers=n_sa, self_attend_heads=num_heads,
                widening_factor=widening_factor, dropout=dropout,
            ))

        self.decoder = nn.ModuleList()
        for i, output_g in enumerate(dec_groups):
            input_g = self.group_schedule[self.processor_idx + i]
            output_k = total_tokens // output_g
            self.decoder.append(HiPBlock(
                D, D, output_g, output_k,
                num_self_attend_layers=1, self_attend_heads=num_heads,
                widening_factor=widening_factor, dropout=dropout,
            ))

        self.output_norm = nn.LayerNorm(D)
        self.output_proj = nn.Linear(D, C)
        # Zero-init so initial residual prediction is near zero.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, c: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        c : [B, L, C] — current latent tokens (L = G*K).
        p : [L, d] or [B, L, d] — token positions.

        Returns
        -------
        pred : [B, L, C] — predicted delta_z (if residual) or z_{t+1}.
        """
        B, L, C = c.shape
        G, K, D = self.G, self.K, self.D

        if p.ndim == 2:
            p = p.unsqueeze(0).expand(B, -1, -1)

        x = c.view(B, G, K, C)
        p_grouped = p.view(B, G, K, -1)

        x = self.input_proj(x)  # [B, G, K, D]

        if self.use_within_group_slot_pos and self.slot_offsets is not None:
            d = self.coord_dim
            if p_grouped.shape[-1] >= 2 * d:
                mu = p_grouped[..., :d]
                sc = p_grouped[..., d:2*d].clamp_min(1e-6)
                sc_g = sc[:, :, :1, :]
                rel = self.slot_offsets.view(1, 1, K, d).to(dtype=mu.dtype, device=mu.device)
                p_eff = mu[:, :, :1, :] + rel * sc_g
                x = x + self.pos_emb(p_eff)
            else:
                x = x + self.pos_emb(p_grouped)
        else:
            x = x + self.pos_emb(p_grouped)

        if self.within_group_id_emb is not None:
            kid = torch.arange(K, device=x.device)
            x = x + self.within_group_id_emb(kid).view(1, 1, K, D)

        encoder_outputs = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i < self.processor_idx:
                encoder_outputs.append(x)

        for i, block in enumerate(self.decoder):
            skip_idx = self.processor_idx - i - 1
            skip = encoder_outputs[skip_idx] if skip_idx >= 0 else None
            x = block(x, skip=skip)

        x = rearrange(x, "B G K D -> B (G K) D")
        return self.output_proj(self.output_norm(x))  # [B, L, C]
