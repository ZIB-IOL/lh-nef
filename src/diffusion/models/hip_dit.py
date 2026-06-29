"""HiP-style Diffusion Transformer for grouped token latents (adaLN-Zero conditioning)."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Union
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models import register
from .pos_emb import PosEmb

__all__ = ["HiPDiT"]


def regroup(inputs: torch.Tensor, num_output_groups: int, regroup_type: str = 'reshape') -> torch.Tensor:
    """Re-group [B, G, N, C] to [B, G', N', C] where G*N = G'*N'."""
    batch_size, num_input_groups, num_input_latents, num_channels = inputs.shape
    new_index_dim = num_input_groups * num_input_latents // num_output_groups

    if regroup_type == 'transpose_reshape':
        inputs = inputs.transpose(1, 2)

    return inputs.reshape(batch_size, num_output_groups, new_index_dim, num_channels)


class SqReLU(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


class FeedForward(nn.Module):
    def __init__(self, dim: int, widening_factor: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = dim * widening_factor
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.act = SqReLU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.linear2(self.act(self.linear1(x))))


class TimestepEmbedder(nn.Module):
    """Sinusoidal-frequency timestep embedder for discrete t in [0, max_period)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._timestep_embedding(t))

    def _timestep_embedding(self, t: torch.Tensor, max_period: int = 10000) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / float(half)
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class FourierTimeEmbedder(nn.Module):
    """Gaussian Fourier feature time embedder for continuous EDM `c_noise = 0.25 * log(sigma)`.

    Matches NVlabs/edm `FourierEmbedding`: frozen Gaussian frequencies with `scale=16` by
    default, then cat([cos, sin]) features, then MLP to hidden_size. The `scale=16` is
    important: with `c_noise ~ [-1.5, 1.1]` (for sigma in [0.002, 80]), a small std produces
    only ~1.5 periods of resolution across the σ range; the EDM-default 16 gives the model
    enough resolving power to distinguish nearby σ values.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, freq_std: float = 16.0):
        super().__init__()
        self.hidden_size = int(hidden_size)
        half = int(frequency_embedding_size) // 2
        # Frozen N(0, freq_std^2) frequencies; never updated.
        self.register_buffer("freqs", torch.randn(half) * float(freq_std), persistent=True)
        self.mlp = nn.Sequential(
            nn.Linear(2 * half, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] float (continuous c_noise). 2*pi*f matches NVlabs/edm FourierEmbedding.
        x = t.float()[:, None] * (2.0 * math.pi) * self.freqs[None]
        emb = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return self.mlp(emb)


class GroupedAttention(nn.Module):
    """Multi-head attention within each group; operates on [B, G, N, C]."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        qk_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        # Input feature dim of the K/V context. Defaults to `dim` (self-attention).
        # For cross-attention with q-side dim ≠ context dim, pass kv_dim=context_dim.
        # In the uniform-hidden-dim case this equals dim and behavior is unchanged
        # (and old ckpts load: to_k/to_v shapes remain (dim, qk_dim)).
        kv_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        qk_dim = qk_dim or dim
        v_dim = v_dim or qk_dim
        kv_dim = kv_dim or dim

        assert qk_dim % heads == 0
        assert v_dim % heads == 0

        self.heads = heads
        self.qk_head_dim = qk_dim // heads
        self.v_head_dim = v_dim // heads
        self.scale = self.qk_head_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, qk_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, v_dim, bias=False)
        self.to_out = nn.Linear(v_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, G, N, C]; context: [B, G, M, C] (defaults to x).
        context = context if context is not None else x
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, 'b g n (h d) -> b g h n d', h=h, d=self.qk_head_dim)
        k = rearrange(k, 'b g m (h d) -> b g h m d', h=h, d=self.qk_head_dim)
        v = rearrange(v, 'b g m (h d) -> b g h m d', h=h, d=self.v_head_dim)

        sim = torch.einsum('b g h n d, b g h m d -> b g h n m', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b g h n m, b g h m d -> b g h n d', attn, v)
        out = rearrange(out, 'b g h n d -> b g n (h d)')
        return self.to_out(out)


class HiPSelfAttentionBlock(nn.Module):
    """Self-attention within each group with adaLN-Zero conditioning."""
    
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        heads: int = 8,
        widening_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = GroupedAttention(dim, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim, widening_factor=widening_factor, dropout=dropout)

        # adaLN-Zero: 6 modulation params (shift/scale/gate for attn and ff)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, G, N, C]; cond: [B, cond_dim]
        params = self.adaLN(cond)  # [B, 6*C]
        shift1, scale1, gate1, shift2, scale2, gate2 = params.chunk(6, dim=-1)

        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale1[:, None, None, :]) + shift1[:, None, None, :]
        x = x + gate1[:, None, None, :] * self.attn(x_norm)

        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale2[:, None, None, :]) + shift2[:, None, None, :]
        x = x + gate2[:, None, None, :] * self.ff(x_norm)

        return x


class HiPCrossAttentionBlock(nn.Module):
    """Cross-attend with learnable queries: [B, G, M, C_in] -> [B, G', N, C_out]."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cond_dim: int,
        num_output_groups: int,
        output_tokens_per_group: int,
        heads: int = 1,
        widening_factor: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_output_groups = num_output_groups
        self.output_tokens_per_group = output_tokens_per_group
        self.output_dim = output_dim

        # HiP uses VarianceScaling(1.0) => std = 1/sqrt(fan_in)
        self.latent_queries = nn.Parameter(
            torch.empty(num_output_groups * output_tokens_per_group, output_dim)
        )
        nn.init.trunc_normal_(self.latent_queries, std=1.0 / sqrt(output_dim))

        self.norm_q = nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-6)
        self.norm_kv = nn.LayerNorm(input_dim, elementwise_affine=False, eps=1e-6)

        self.attn = GroupedAttention(
            dim=output_dim,
            heads=heads,
            qk_dim=input_dim,
            v_dim=input_dim,
            kv_dim=input_dim,                  # context (from input_proj/prev block) comes in at input_dim
            dropout=dropout,
        )
        # GroupedAttention.to_out already projects v_dim → dim=output_dim, so no further
        # projection is needed here. Kept as Identity for ckpt-key stability (old ckpts
        # with uniform hidden_dim recorded out_proj=Identity, which has zero params).
        self.out_proj = nn.Identity()

        self.norm_ff = nn.LayerNorm(output_dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(output_dim, widening_factor=widening_factor, dropout=dropout)

        # adaLN-Zero: 6 modulation params (shift/scale/gate × 2). Gates init to zero so
        # the block is identity-at-init, matching HiPSelfAttentionBlock and the DiT /
        # PixArt / SD3 convention. Without gates here, random latent_queries and random
        # attn weights inject a non-trivial residual at init, defeating EDM's "F_θ ≈ 0
        # at init → x0_hat ≈ c_skip · x_noisy" preconditioning design.
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * output_dim))
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(
        self,
        inputs: torch.Tensor,
        cond: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # inputs: [B, G, M, C_in]; skip (U-Net): [B, G', N, C_out].
        B = inputs.shape[0]
        G_out = self.num_output_groups
        N_out = self.output_tokens_per_group

        params = self.adaLN(cond)
        shift1, scale1, gate1, shift2, scale2, gate2 = params.chunk(6, dim=-1)

        queries = repeat(self.latent_queries, 'l d -> b l d', b=B)
        queries = rearrange(queries, 'b (g n) d -> b g n d', g=G_out)

        if skip is not None:
            queries = queries + skip

        inputs_regrouped = regroup(inputs, G_out)

        q_norm = self.norm_q(queries)
        q_norm = q_norm * (1 + scale1[:, None, None, :]) + shift1[:, None, None, :]
        kv_norm = self.norm_kv(inputs_regrouped)

        attn_out = self.attn(q_norm, context=kv_norm)
        attn_out = self.out_proj(attn_out)
        out = queries + gate1[:, None, None, :] * attn_out

        out_norm = self.norm_ff(out)
        out_norm = out_norm * (1 + scale2[:, None, None, :]) + shift2[:, None, None, :]
        out = out + gate2[:, None, None, :] * self.ff(out_norm)

        return out


class HiPBlock(nn.Module):
    """One HiP block: regroup + cross-attention to learnable queries + self-attentions."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cond_dim: int,
        num_output_groups: int,
        output_tokens_per_group: int,
        num_self_attend_layers: int = 2,
        self_attend_heads: int = 8,
        cross_attend_heads: int = 1,
        widening_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_output_groups = num_output_groups
        self.output_tokens_per_group = output_tokens_per_group

        self.cross_attn = HiPCrossAttentionBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            cond_dim=cond_dim,
            num_output_groups=num_output_groups,
            output_tokens_per_group=output_tokens_per_group,
            heads=cross_attend_heads,
            widening_factor=1,
            dropout=dropout,
        )

        self.self_attns = nn.ModuleList([
            HiPSelfAttentionBlock(
                dim=output_dim,
                cond_dim=cond_dim,
                heads=self_attend_heads,
                widening_factor=widening_factor,
                dropout=dropout,
            )
            for _ in range(num_self_attend_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.cross_attn(x, cond, skip=skip)
        for self_attn in self.self_attns:
            z = self_attn(z, cond)
        return z


class FinalLayer(nn.Module):
    """Final output layer with adaLN-Zero."""
    
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * hidden_size))
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.adaLN[1].weight)
        nn.init.zeros_(self.adaLN[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(cond).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
        return self.linear(x)


class HiPDiT(nn.Module):
    """
    HiP-style Diffusion Transformer (U-Net of HiP blocks with skip connections).

    Input shape [B, G*K, C] reshaped to [B, G, K, C]:
      G = spatial groups, K = tokens per group, C = channels per token.
    """

    def __init__(
        self,
        *,
        num_groups: int = 16,        # G (input)
        tokens_per_group: int = 4,   # K
        token_dim: int = 32,         # C
        coord_dim: int = 2,          # d
        pos_freq: float = 1.0,
        # Note: extractor stores p by repeating group centers K times, so all K tokens in a group
        # share the same positional embedding. Without one of the two options below, the model is
        # permutation-symmetric within each group (source of checkerboard artifacts).
        use_within_group_token_id_emb: bool = True,
        # If True, p must include per-group bbox extents (last-dim = 2*coord_dim) and per-token
        # slot offsets become: p_eff[g,k] = mu_g + slot_offset[k] * lambda_g.
        use_within_group_slot_pos: bool = False,
        # FiLM modulation conditioned on (slot_offset, log(lambda_g)).
        use_slot_scale_film: bool = False,
        # Per-level hidden dim. int → uniform across all blocks (legacy). list of length
        # depth+1 → encoder-side schedule (decoder is mirrored automatically so skip
        # connections line up). cond_dim for adaLN inputs = max(enc_dims).
        # Example widening: [384, 384, 512, 768] for G=32 → 8 → 2 → 1.
        # All values must be divisible by `num_heads`.
        hidden_dim: Union[int, Sequence[int]] = 256,
        depth: int = 3,
        # Per-block self-attention depth. int → legacy (only the processor block gets
        # this value; encoder/decoder blocks get 1). list of length 2*depth+1 → explicit
        # per-block depth ordered [enc_0, enc_1, ..., processor, dec_0, ..., dec_{depth-1}].
        # Example uniform: [2,2,2,4,2,2,2]; example pyramid: [1,2,4,4,4,2,1].
        self_attend_layers: Union[int, Sequence[int]] = 2,
        # Optional explicit num_groups schedule for the encoder side, length depth+1,
        # must start with `num_groups` and end with 1, each entry must divide G*K.
        # If None, uses the geometric factor-4 reduction down to 1 (legacy).
        # Example gradual factor-2: [32, 16, 8, 4, 2, 1] with depth=5.
        num_groups_schedule: Optional[Sequence[int]] = None,
        num_heads: int = 8,
        widening_factor: int = 4,
        dropout: float = 0.0,
        learn_sigma: bool = False,
        # 'discrete' = sinusoidal TimestepEmbedder calibrated for t in [0, 1000).
        # 'edm_cnoise' = Gaussian-Fourier feature embedder for continuous EDM c_noise input.
        time_input: str = "discrete",
    ):
        super().__init__()
        self.num_groups = int(num_groups)
        self.tokens_per_group = int(tokens_per_group)
        self.token_dim = int(token_dim)
        self.coord_dim = int(coord_dim)
        self.depth = int(depth)
        self.learn_sigma = bool(learn_sigma)
        self.use_within_group_token_id_emb = bool(use_within_group_token_id_emb)
        self.use_within_group_slot_pos = bool(use_within_group_slot_pos)
        self.use_slot_scale_film = bool(use_slot_scale_film)
        self.time_input = str(time_input).lower().strip()
        if self.time_input not in ("discrete", "edm_cnoise"):
            raise ValueError(f"time_input must be 'discrete' or 'edm_cnoise', got {time_input!r}")

        G, K, C, depth_v = self.num_groups, self.tokens_per_group, self.token_dim, self.depth
        out_channels = C * (2 if self.learn_sigma else 1)
        total_tokens = G * K

        # ----- Resolve num_groups schedule (encoder side, length depth+1) -----
        if num_groups_schedule is None:
            enc_groups = [G]
            g = G
            for _ in range(depth_v - 1):
                g = max(1, g // 4)
                enc_groups.append(g)
            enc_groups.append(1)  # processor always at G=1
        else:
            enc_groups = [int(v) for v in num_groups_schedule]
            if len(enc_groups) != depth_v + 1:
                raise ValueError(
                    f"num_groups_schedule must have length depth+1={depth_v+1}, got {len(enc_groups)}"
                )
            if enc_groups[0] != G:
                raise ValueError(
                    f"num_groups_schedule[0] must equal num_groups={G}, got {enc_groups[0]}"
                )
            if enc_groups[-1] != 1:
                raise ValueError(
                    f"num_groups_schedule[-1] must equal 1 (processor), got {enc_groups[-1]}"
                )
            for g in enc_groups:
                if total_tokens % g != 0:
                    raise ValueError(
                        f"num_groups_schedule entry {g} does not divide total tokens G*K={total_tokens}"
                    )

        dec_groups = list(reversed(enc_groups[:-1]))   # length depth
        all_groups = enc_groups + dec_groups            # length 2*depth+1
        self.group_schedule = all_groups
        self.processor_idx = len(enc_groups) - 1
        total_blocks = len(all_groups)

        # ----- Resolve hidden_dim schedule (encoder + processor, length depth+1) -----
        if isinstance(hidden_dim, int):
            enc_dims = [int(hidden_dim)] * (depth_v + 1)
        else:
            enc_dims = [int(v) for v in hidden_dim]
            if len(enc_dims) != depth_v + 1:
                raise ValueError(
                    f"hidden_dim list must have length depth+1={depth_v+1} "
                    f"(encoder + processor); got {len(enc_dims)}"
                )
        # decoder block i takes skip from encoder block (depth-1-i), so its output dim
        # must match encoder block (depth-1-i)'s output dim (skip is added to queries).
        dec_dims = list(reversed(enc_dims[:-1]))
        all_dims = enc_dims + dec_dims                  # length 2*depth+1
        for d in all_dims:
            if d % num_heads != 0:
                raise ValueError(
                    f"hidden_dim entry {d} is not divisible by num_heads={num_heads}. "
                    f"All hidden_dim values (and their mirrored decoder counterparts) "
                    f"must be multiples of num_heads."
                )
        self.enc_dims = enc_dims
        self.dec_dims = dec_dims
        self.all_dims = all_dims
        # Back-compat: external code reads .hidden_dim as a scalar.
        self.hidden_dim = enc_dims[0]
        # cond_dim is the timestep-embedding output dim, the same input to every block's
        # adaLN. Pick max(enc_dims) so the cond has enough capacity for the widest block.
        cond_dim = max(enc_dims)
        self.cond_dim = cond_dim

        # ----- Resolve self-attend schedule (per-block, length 2*depth+1) -----
        if isinstance(self_attend_layers, int):
            # Legacy: only processor gets this depth; encoder/decoder get 1 each.
            sa_sched = [int(self_attend_layers) if i == self.processor_idx else 1
                        for i in range(total_blocks)]
        else:
            sa_sched = [int(v) for v in self_attend_layers]
            if len(sa_sched) != total_blocks:
                raise ValueError(
                    f"self_attend_layers list must have length 2*depth+1={total_blocks}; "
                    f"got {len(sa_sched)}. Schedule order: "
                    f"[enc_0, ..., enc_{depth_v-1}, processor, dec_0, ..., dec_{depth_v-1}]."
                )
        self.self_attend_schedule = sa_sched

        # ----- Build embeddings (all at enc_dims[0]) -----
        D0 = enc_dims[0]
        self.input_proj = nn.Linear(C, D0)
        self.pos_embed = PosEmb(D0, coord_dim=self.coord_dim, freq=float(pos_freq))

        self.within_group_id_emb = None
        if self.use_within_group_token_id_emb:
            self.within_group_id_emb = nn.Embedding(int(K), int(D0))

        self.slot_offsets = None
        self.slot_film = None
        if self.use_within_group_slot_pos:
            self.slot_offsets = nn.Parameter(torch.zeros((int(K), int(self.coord_dim)), dtype=torch.float32))
            with torch.no_grad():
                if int(self.coord_dim) == 2 and int(K) == 4:
                    init = torch.tensor(
                        [[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]],
                        dtype=torch.float32,
                    )
                    self.slot_offsets.copy_(init)
                elif int(K) == 1:
                    self.slot_offsets.zero_()
                elif int(self.coord_dim) == 2:
                    ang = torch.linspace(0.0, 2.0 * math.pi, steps=int(K) + 1, dtype=torch.float32)[:-1]
                    init = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1) * 0.5
                    self.slot_offsets.copy_(init)
                else:
                    self.slot_offsets.normal_(mean=0.0, std=0.25)

            if self.use_slot_scale_film:
                in_dim = int(2 * self.coord_dim)
                self.slot_film = nn.Sequential(
                    nn.Linear(in_dim, int(D0)),
                    nn.SiLU(),
                    nn.Linear(int(D0), int(2 * D0)),
                )
                nn.init.zeros_(self.slot_film[-1].weight)
                nn.init.zeros_(self.slot_film[-1].bias)

        if self.time_input == "edm_cnoise":
            self.t_embed = FourierTimeEmbedder(cond_dim)
        else:
            self.t_embed = TimestepEmbedder(cond_dim)

        # ----- Build encoder blocks -----
        self.encoder = nn.ModuleList()
        for i in range(len(enc_groups)):
            input_dim = enc_dims[i - 1] if i > 0 else D0
            output_dim = enc_dims[i]
            output_g = enc_groups[i]
            output_k = total_tokens // output_g

            self.encoder.append(HiPBlock(
                input_dim=input_dim,
                output_dim=output_dim,
                cond_dim=cond_dim,
                num_output_groups=output_g,
                output_tokens_per_group=output_k,
                num_self_attend_layers=sa_sched[i],
                self_attend_heads=num_heads,
                widening_factor=widening_factor,
                dropout=dropout,
            ))

        # ----- Build decoder blocks -----
        self.decoder = nn.ModuleList()
        for i, output_g in enumerate(dec_groups):
            block_idx = len(enc_groups) + i             # absolute index in all_dims/sa_sched
            input_dim = all_dims[block_idx - 1]         # previous block's output
            output_dim = dec_dims[i]
            output_k = total_tokens // output_g

            self.decoder.append(HiPBlock(
                input_dim=input_dim,
                output_dim=output_dim,
                cond_dim=cond_dim,
                num_output_groups=output_g,
                output_tokens_per_group=output_k,
                num_self_attend_layers=sa_sched[block_idx],
                self_attend_heads=num_heads,
                widening_factor=widening_factor,
                dropout=dropout,
            ))

        # Final block reads at the last decoder dim (= enc_dims[0]).
        self.final = FinalLayer(D0, out_channels, cond_dim)

    def forward(self, *, c_noisy: torch.Tensor, p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        c_noisy: [B, G*K, C]
        p:       [G*K, d] or [B, G*K, d]
        t:       [B] — long timestep index in [0, T) for time_input='discrete',
                       or float `c_noise = 0.25*log(sigma)` for time_input='edm_cnoise'.
        Returns: [B, G*K, C]. Interpretation depends on the diffusion framework
                       (eps/v for DDPM; raw F for EDM, which the trainer wraps via precond_forward).
        """
        B = c_noisy.shape[0]
        G, K, C = self.num_groups, self.tokens_per_group, self.token_dim
        D = self.hidden_dim

        if c_noisy.shape[1] != G * K or c_noisy.shape[2] != C:
            raise ValueError(f"Expected c_noisy [B, {G*K}, {C}], got {tuple(c_noisy.shape)}")

        if p.ndim == 2:
            p = p.unsqueeze(0).expand(B, -1, -1)  # [B, G*K, d]

        x = c_noisy.view(B, G, K, C)
        p_grouped = p.view(B, G, K, -1)  # [B, G, K, d]

        x = self.input_proj(x)  # [B, G, K, D]

        if self.use_within_group_slot_pos:
            if p_grouped.shape[-1] != int(2 * self.coord_dim):
                raise ValueError(
                    "use_within_group_slot_pos=True expects p to contain [center, scale] per token "
                    f"with last-dim=2*coord_dim={2*self.coord_dim}, but got p.shape[-1]={int(p_grouped.shape[-1])}. "
                    "Enable dataset args: include_group_scales_in_p=true (and re-extract latents if needed)."
                )
            mu = p_grouped[..., : self.coord_dim]  # [B,G,K,d]
            sc = p_grouped[..., self.coord_dim :].clamp_min(1e-6)  # [B,G,K,d] (scale repeated within group)
            # Per-group scale is constant across K; take k=0 slice for stability.
            sc_g = sc[:, :, :1, :]  # [B,G,1,d]
            rel = self.slot_offsets.view(1, 1, K, self.coord_dim).to(dtype=mu.dtype, device=mu.device)  # [1,1,K,d]
            p_eff = mu[:, :, :1, :] + rel * sc_g
            x = x + self.pos_embed(p_eff)

            if self.slot_film is not None:
                log_sc = torch.log(sc_g).expand(B, G, K, self.coord_dim)
                rel_b = rel.expand(B, G, K, self.coord_dim)
                film_in = torch.cat([rel_b, log_sc], dim=-1)  # [B,G,K,2d]
                gamma_beta = self.slot_film(film_in)  # [B,G,K,2D]
                gamma, beta = gamma_beta.chunk(2, dim=-1)
                x = x * (1.0 + gamma) + beta
        else:
            x = x + self.pos_embed(p_grouped)

        if self.within_group_id_emb is not None:
            kid = torch.arange(K, device=x.device, dtype=torch.long)  # [K]
            kid_emb = self.within_group_id_emb(kid).view(1, 1, K, D)  # [1,1,K,D]
            x = x + kid_emb

        cond = self.t_embed(t)  # [B, D]

        encoder_outputs = []
        for i, block in enumerate(self.encoder):
            x = block(x, cond)
            if i < self.processor_idx:  # don't keep processor output as a skip
                encoder_outputs.append(x)

        for i, block in enumerate(self.decoder):
            skip_idx = self.processor_idx - i - 1
            skip = encoder_outputs[skip_idx] if skip_idx >= 0 else None
            x = block(x, cond, skip=skip)

        out = self.final(x, cond)  # [B, G, K, C]
        return out.view(B, G * K, -1)


@register("hip_dit")
def make_hip_dit(**kwargs) -> HiPDiT:
    return HiPDiT(**kwargs)
