# Copyright 2024 - PyTorch port of DeepMind's Hierarchical Perceiver (HiP)
# Original JAX implementation: https://github.com/google-deepmind/hierarchical_perceiver
# Paper: https://arxiv.org/abs/2202.10890

"""
Hierarchical Perceiver (HiP) - PyTorch Implementation

HiP extends Perceiver with:
1. Hierarchical processing with local groups (locality)
2. U-Net style encoder-decoder with skip connections
3. Regrouping mechanism to change locality at each level
"""

from typing import Optional, Sequence, Dict, List, Tuple, Union
from math import pi, log, sqrt, ceil

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat, reduce

# ==============================================================================
# Helper functions
# ==============================================================================

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def padding_to_make_divisible(index_dim: int, num_groups: int) -> int:
    """Compute padding needed to make index_dim divisible by num_groups."""
    return num_groups * ceil(index_dim / num_groups) - index_dim

def assign_groups_to_modalities(
    num_groups: int, 
    index_dim_per_modality: Sequence[int]
) -> Tuple[List[int], int]:
    """Computes the number of groups assigned to each modality."""
    num_modalities = len(index_dim_per_modality)
    if num_modalities > num_groups:
        raise ValueError(f'{num_modalities} > {num_groups}. Cannot handle this case.')
    
    extra_groups = num_groups - num_modalities
    num_groups_per_modality = [1] * num_modalities
    index_dim_per_group = list(index_dim_per_modality)
    
    for _ in range(extra_groups):
        modality = int(torch.argmax(torch.tensor(index_dim_per_group)).item())
        num_groups_per_modality[modality] += 1
        index_dim_per_group[modality] = (
            index_dim_per_modality[modality] / num_groups_per_modality[modality]
        )
    
    index_dim_per_group = ceil(max(index_dim_per_group))
    return num_groups_per_modality, index_dim_per_group

def regroup(
    inputs: Tensor,
    num_output_groups: int,
    regroup_type: str = 'reshape',
) -> Tensor:
    """Re-group an input array from [B, G, N, C] to [B, G', N', C].
    
    This is the key mechanism in HiP for changing locality at each level.
    
    Args:
        inputs: Tensor of shape [B, G, N, C]
        num_output_groups: Target number of groups G'
        regroup_type: 'reshape' or 'transpose_reshape'
    
    Returns:
        Tensor of shape [B, G', N', C] where G*N = G'*N'
    """
    batch_size, num_input_groups, num_input_latents, num_channels = inputs.shape
    
    if regroup_type in ['reshape', 'transpose_reshape']:
        new_index_dim = num_input_groups * num_input_latents // num_output_groups
        
        if regroup_type == 'transpose_reshape':
            # [B, G, N, C] -> [B, N, G, C]
            # This leads to mixing between all input groups
            inputs = inputs.transpose(1, 2)
        
        outputs = inputs.reshape(batch_size, num_output_groups, new_index_dim, num_channels)
    else:
        raise ValueError(f'Unknown regroup_type: {regroup_type}')
    
    return outputs

# ==============================================================================
# Geometry helpers
# ==============================================================================

def _karcher_mean_s2(
    points: Tensor,
    init: Tensor,
    *,
    n_iters: int = 5,
    eps: float = 1e-6,
) -> Tensor:
    mu = init
    cos_clamp = 1.0 - float(eps)
    for _ in range(int(n_iters)):
        cos_t = (mu.unsqueeze(1) * points).sum(dim=-1).clamp(-cos_clamp, cos_clamp)  # [B, M]
        theta = torch.arccos(cos_t)
        sin_t = torch.sin(theta).clamp_min(eps)
        proj = points - cos_t.unsqueeze(-1) * mu.unsqueeze(1)
        log_x = proj * (theta / sin_t).unsqueeze(-1)  # [B, M, 3]
        v_bar = log_x.mean(dim=1)  # [B, 3]
        v_norm = v_bar.norm(dim=-1, keepdim=True).clamp_min(eps)
        mu = torch.cos(v_norm) * mu + torch.sin(v_norm) * v_bar / v_norm
        mu = mu / mu.norm(dim=-1, keepdim=True).clamp_min(eps)  # numerical safety
    return mu

# ==============================================================================
# Core Building Blocks
# ==============================================================================

class FourierFeatures(nn.Module):
    """Fourier positional encoding for any-dimensional coordinates."""

    def __init__(
        self,
        coord_dim: int,
        num_bands: int = 64,
        max_freq: float = 10.0,
    ):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_bands = num_bands

        freqs = torch.linspace(1.0, max_freq / 2, num_bands)
        self.register_buffer('freqs', freqs)
        self.out_dim = coord_dim * (2 * num_bands + 1)
    
    def forward(self, coords: Tensor) -> Tensor:
        """
        Args:
            coords: [B, N, coord_dim]
        Returns:
            [B, N, out_dim]
        """
        coords_expanded = coords.unsqueeze(-1) * self.freqs * pi
        encoded = torch.cat([coords_expanded.sin(), coords_expanded.cos()], dim=-1)
        encoded = rearrange(encoded, 'b n d f -> b n (d f)')
        return torch.cat([encoded, coords], dim=-1)

class SqReLU(nn.Module):
    """Squared ReLU activation, used in HiP."""
    def forward(self, x):
        return F.relu(x) ** 2

class GEGLU(nn.Module):
    """GEGLU activation."""
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    """Feedforward with configurable activation."""
    
    def __init__(
        self, 
        dim: int, 
        widening_factor: int = 4, 
        dropout: float = 0.0,
        activation: str = 'sq_relu'
    ):
        super().__init__()
        hidden_dim = dim * widening_factor
        
        if activation == 'geglu':
            self.linear1 = nn.Linear(dim, hidden_dim * 2)
            self.act = GEGLU()
        elif activation == 'sq_relu':
            self.linear1 = nn.Linear(dim, hidden_dim)
            self.act = SqReLU()
        else:
            self.linear1 = nn.Linear(dim, hidden_dim)
            self.act = nn.GELU()
        
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    """Multi-head attention supporting both self and cross attention.
    
    Operates on grouped tensors of shape [B, G, N, C].
    Attention is computed WITHIN each group (locality).
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        qk_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        output_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        context_dim = default(context_dim, query_dim)

        qk_channels = default(qk_channels, context_dim)
        v_channels = default(v_channels, qk_channels)
        output_channels = default(output_channels, v_channels)

        assert qk_channels % heads == 0, f'qk_channels ({qk_channels}) must be divisible by heads ({heads})'
        assert v_channels % heads == 0, f'v_channels ({v_channels}) must be divisible by heads ({heads})'

        self.heads = heads
        self.qk_head_dim = qk_channels // heads
        self.v_head_dim = v_channels // heads
        self.scale = self.qk_head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, qk_channels, bias=False)
        self.to_k = nn.Linear(context_dim, qk_channels, bias=False)
        self.to_v = nn.Linear(context_dim, v_channels, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(v_channels, output_channels)

    def forward(
        self, 
        x: Tensor, 
        context: Optional[Tensor] = None, 
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: Query tensor [B, G, N, C] or [B, N, C]
            context: Key/Value tensor [B, G, M, C] or [B, M, C] (defaults to x)
            mask: Optional attention mask
        """
        h = self.heads
        has_groups = x.dim() == 4
        
        if has_groups:
            b, g, n, _ = x.shape
        else:
            b, n, _ = x.shape
            g = 1
            x = x.unsqueeze(1)
        
        context = default(context, x)
        if context.dim() == 3:
            context = context.unsqueeze(1)
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = rearrange(q, 'b g n (h d) -> b g h n d', h=h, d=self.qk_head_dim)
        k = rearrange(k, 'b g m (h d) -> b g h m d', h=h, d=self.qk_head_dim)
        v = rearrange(v, 'b g m (h d) -> b g h m d', h=h, d=self.v_head_dim)

        sim = torch.einsum('b g h n d, b g h m d -> b g h n m', q, k) * self.scale

        if exists(mask):
            max_neg = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('b g h n m, b g h m d -> b g h n d', attn, v)
        out = rearrange(out, 'b g h n d -> b g n (h d)')
        out = self.to_out(out)

        if not has_groups:
            out = out.squeeze(1)

        return out

class SelfAttention(nn.Module):
    """Self-attention block with pre-norm and residual."""
    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        qk_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        widening_factor: int = 4,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        activation: str = 'sq_relu'
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            query_dim=dim,
            heads=heads,
            qk_channels=qk_channels,
            v_channels=v_channels,
            output_channels=dim,
            dropout=attn_dropout
        )
        self.attn_output_dropout = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, widening_factor=widening_factor, dropout=dropout, activation=activation)
        
        self.drop_path_rate = drop_path_rate
    
    def drop_path(self, x: Tensor, training: bool) -> Tensor:
        """Stochastic depth / drop path (no scaling)."""
        if not training or self.drop_path_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * mask
    
    def forward(self, x: Tensor) -> Tensor:
        attn_out = self.attn(self.norm1(x))
        attn_out = self.attn_output_dropout(attn_out)
        x = x + self.drop_path(attn_out, self.training)
        x = x + self.drop_path(self.ff(self.norm2(x)), self.training)
        return x

class HiPCrossAttention(nn.Module):
    """Cross-attention for HiP with learnable latent queries.
    
    Maps [B, G, M, C] to [B, G, N, D] where N and D are output dimensions.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        output_index_dim: int,
        num_groups: int,
        heads: int = 1,
        widening_factor: int = 1,
        dropout: float = 0.0,
        activation: str = 'sq_relu',
        use_post_attention_residual: bool = True,
        residual_dim: Optional[int] = None
    ):
        super().__init__()
        self.output_index_dim = output_index_dim
        self.num_groups = num_groups
        self.output_dim = output_dim
        self.use_post_attention_residual = use_post_attention_residual
        
        # Learnable latent queries: [G * N, D]
        self.latent_queries = nn.Parameter(
            torch.empty(num_groups * output_index_dim, output_dim)
        )
        nn.init.trunc_normal_(self.latent_queries, std=1.0 / sqrt(output_dim))
        
        # Pre-attention residual projection (for U-Net skip connections)
        self.residual_dim = residual_dim
        if use_post_attention_residual and residual_dim is not None and residual_dim != output_dim:
            self.pre_residual_proj = nn.Linear(residual_dim, output_dim)
        else:
            self.pre_residual_proj = None
        
        self.norm_q = nn.LayerNorm(output_dim)
        self.norm_kv = nn.LayerNorm(input_dim)
        
        self.attn = Attention(
            query_dim=output_dim,
            context_dim=input_dim,
            heads=heads,
            qk_channels=input_dim,
            v_channels=input_dim,
            output_channels=output_dim,
            dropout=dropout,
        )
        
        self.norm_ff = nn.LayerNorm(output_dim)
        self.ff = FeedForward(output_dim, widening_factor=widening_factor, dropout=dropout, activation=activation)
    
    def forward(
        self,
        inputs: Tensor,
        pre_attention_residual: Optional[Tensor] = None,
        query_inputs: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            inputs: [B, G, M, C] tensor to cross-attend to
            pre_attention_residual: Optional [B, G, N, D] for U-Net skip
            query_inputs: Optional explicit queries (for reconstruction)
        
        Returns:
            [B, G, N, D] output
        """
        batch_size, num_groups, _, _ = inputs.shape
        
        if query_inputs is None:
            queries = repeat(self.latent_queries, 'l d -> b l d', b=batch_size)
            queries = rearrange(queries, 'b (g n) d -> b g n d', g=num_groups)
        else:
            queries = query_inputs
        
        if exists(pre_attention_residual):
            if self.pre_residual_proj is not None:
                pre_attention_residual = self.pre_residual_proj(pre_attention_residual)
            queries = queries + pre_attention_residual
        
        attn_out = self.attn(self.norm_q(queries), context=self.norm_kv(inputs))
        
        if self.use_post_attention_residual:
            attn_out = attn_out + queries
        
        out = attn_out + self.ff(self.norm_ff(attn_out))
        
        return out

class PerceiverBlock(nn.Module):
    """A single block of HiP: regroup -> cross-attention -> self-attentions.
    
    This is the core building block that:
    1. Optionally regroups inputs to change locality
    2. Cross-attends to compressed latent representation
    3. Processes with multiple self-attention layers
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_output_groups: int,
        output_index_dim: int,
        num_self_attend_layers: int,
        num_self_attend_heads: int = 8,
        self_attend_widening_factor: int = 4,
        num_cross_attend_heads: int = 1,
        cross_attend_widening_factor: int = 1,
        regroup_inputs: bool = True,
        regroup_type: str = 'reshape',
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        activation: str = 'sq_relu',
        use_post_attention_residual: bool = True,
        residual_dim: Optional[int] = None
    ):
        super().__init__()
        self.num_output_groups = num_output_groups
        self.output_index_dim = output_index_dim
        self.regroup_inputs = regroup_inputs
        self.regroup_type = regroup_type
        
        self.cross_attn = HiPCrossAttention(
            input_dim=input_dim,
            output_dim=output_dim,
            output_index_dim=output_index_dim,
            num_groups=num_output_groups,
            heads=num_cross_attend_heads,
            widening_factor=cross_attend_widening_factor,
            dropout=dropout,
            activation=activation,
            use_post_attention_residual=use_post_attention_residual,
            residual_dim=residual_dim
        )
        
        self.self_attns = nn.ModuleList([
            SelfAttention(
                dim=output_dim,
                heads=num_self_attend_heads,
                widening_factor=self_attend_widening_factor,
                dropout=dropout,
                attn_dropout=dropout,
                drop_path_rate=drop_path_rate,
                activation=activation
            )
            for _ in range(num_self_attend_layers)
        ])
    
    def forward(
        self,
        inputs: Tensor,
        pre_attention_residual: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            inputs: [B, G, N, C] grouped tensor
            pre_attention_residual: Optional skip connection from encoder
        
        Returns:
            [B, G', N', D] processed tensor
        """
        if self.regroup_inputs:
            inputs = regroup(inputs, self.num_output_groups, self.regroup_type)
        
        z = self.cross_attn(inputs, pre_attention_residual=pre_attention_residual)
        
        for self_attn in self.self_attns:
            z = self_attn(z)
        
        return z

# ==============================================================================
# Groupers: Handle input grouping for HiP
# ==============================================================================

class ConstNumGrouper(nn.Module):
    """Groups inputs into a constant number of groups.
    
    Handles padding so that inputs can be evenly divided.
    """
    
    def __init__(self, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self._group_info = None
    
    def group(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Args:
            inputs: Dict of {modality: [B, N, C]} tensors
        
        Returns:
            [B, G, N_per_group, C] grouped tensor
        """
        index_dims = [v.shape[1] for v in inputs.values()]
        num_groups_per_modality, index_dim_per_group = assign_groups_to_modalities(
            self.num_groups, index_dims
        )
        
        # Store group info for ungrouping
        self._group_info = []
        grouped_inputs = []
        next_group_id = 0
        
        for (name, value), num_modality_groups in zip(inputs.items(), num_groups_per_modality):
            index_dim = value.shape[1]
            assigned_groups = list(range(next_group_id, next_group_id + num_modality_groups))
            next_group_id += num_modality_groups
            
            final_padding = padding_to_make_divisible(index_dim, num_modality_groups)
            local_index_dim_per_group = (index_dim + final_padding) // num_modality_groups
            group_padding = index_dim_per_group - local_index_dim_per_group
            
            self._group_info.append({
                'name': name,
                'groups': assigned_groups,
                'final_padding': final_padding,
                'group_padding': group_padding,
                'orig_index_dim': index_dim
            })
            
            # Pad and reshape
            if final_padding > 0:
                value = F.pad(value, (0, 0, 0, final_padding))
            
            # [B, N, C] -> [B, G, N/G, C]
            value = rearrange(value, 'b (g n) c -> b g n c', g=num_modality_groups)
            
            if group_padding > 0:
                value = F.pad(value, (0, 0, 0, group_padding))
            
            grouped_inputs.append(value)
        
        return torch.cat(grouped_inputs, dim=1)
    
    def ungroup(self, latents: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            latents: [B, G, N, C] grouped tensor
        
        Returns:
            Dict of {modality: [B, N_orig, C]} tensors
        """
        out = {}
        
        for info in self._group_info:
            # Select relevant groups
            x = latents[:, info['groups'], :, :]
            
            # Remove per-group padding
            if info['group_padding'] > 0:
                x = x[:, :, :-info['group_padding'], :]
            
            # Flatten groups
            x = rearrange(x, 'b g n c -> b (g n) c')
            
            # Remove final padding
            if info['final_padding'] > 0:
                x = x[:, :-info['final_padding'], :]
            
            out[info['name']] = x
        
        return out

class ConcatenateGrouper(nn.Module):
    """Simply concatenates all modalities into a single group."""
    
    def __init__(self):
        super().__init__()
        self._input_info = None
    
    def group(self, inputs: Dict[str, Tensor]) -> Tensor:
        """[B, N_i, C] -> [B, 1, sum(N_i), C]"""
        self._input_info = [(name, v.shape[1]) for name, v in inputs.items()]
        grouped = torch.cat(list(inputs.values()), dim=1)
        return grouped.unsqueeze(1)  # Add group dimension
    
    def ungroup(self, latents: Tensor) -> Dict[str, Tensor]:
        """[B, 1, N, C] -> {name: [B, N_i, C]}"""
        latents = latents.squeeze(1)
        out = {}
        start_idx = 0
        for name, index_dim in self._input_info:
            out[name] = latents[:, start_idx:start_idx + index_dim, :]
            start_idx += index_dim
        return out

# ==============================================================================
# Embedders and Position Encoders
# ==============================================================================

class Embedder(nn.Module):
    """Projects inputs to a common embedding dimension."""
    
    def __init__(self, num_embedding_channels: int):
        super().__init__()
        self.num_embedding_channels = num_embedding_channels
        self._embed_layers = nn.ModuleDict()
        self._unembed_layers = nn.ModuleDict()
        self._orig_channels = {}
    
    def embed(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Project each modality to embedding dimension."""
        out = {}
        for name, value in inputs.items():
            self._orig_channels[name] = value.shape[-1]
            
            if name not in self._embed_layers:
                layer = nn.Linear(value.shape[-1], self.num_embedding_channels)
                # VarianceScaling(1.0) initialization
                nn.init.trunc_normal_(layer.weight, std=1.0 / sqrt(value.shape[-1]))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                # Move to same device as input
                layer = layer.to(value.device)
                self._embed_layers[name] = layer
            
            out[name] = self._embed_layers[name](value)
        
        return out
    
    def unembed(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Project back to original channel dimensions."""
        out = {}
        for name, value in inputs.items():
            if name not in self._unembed_layers:
                layer = nn.Linear(self.num_embedding_channels, self._orig_channels[name])
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                layer = layer.to(value.device)
                self._unembed_layers[name] = layer
            out[name] = self._unembed_layers[name](value)
        return out

class PositionEncoder(nn.Module):
    """Adds position encodings to inputs (learned or Fourier)."""
    
    def __init__(
        self, 
        num_channels: int,
        num_position_encoding_channels: Optional[int] = None,
        pos_encoding: str = 'fourier',
        fourier_bands: int = 64,
        fourier_max_freq: float = 10.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_position_encoding_channels = num_position_encoding_channels
        self.pos_encoding = str(pos_encoding)
        self.fourier_bands = int(fourier_bands)
        self.fourier_max_freq = float(fourier_max_freq)
        self._pos_embs = nn.ParameterDict()
        self._pos_proj = None  # Lazy init if needed
        self._fourier = nn.ModuleDict()
        self._fourier_proj = nn.ModuleDict()
    
    def forward(
        self,
        inputs: Dict[str, Tensor],
        coords: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Add position encodings and return both encoded inputs and raw encodings.

        If pos_encoding=='fourier', caller must provide coords for each modality:
          coords[name]: [B, N, coord_dim] aligned with inputs[name] tokens.
        """
        out = {}
        pos_encodings = {}
        
        # Determine position encoding channels
        if self.num_position_encoding_channels is None:
            pos_enc_channels = self.num_channels
        else:
            pos_enc_channels = self.num_position_encoding_channels
        
        for name, value in inputs.items():
            index_dim = value.shape[1]
            device = value.device

            if self.pos_encoding == 'learned':
                if name not in self._pos_embs:
                    # JAX uses VarianceScaling(1.0): std = 1/sqrt(fan_in) = 1/sqrt(num_channels)
                    pos_emb = nn.Parameter(torch.empty(index_dim, self.num_channels, device=device))
                    nn.init.trunc_normal_(pos_emb, std=1.0 / sqrt(self.num_channels))
                    self._pos_embs[name] = pos_emb
                pos_enc = self._pos_embs[name].unsqueeze(0).expand(value.shape[0], -1, -1)
            elif self.pos_encoding == 'fourier':
                if coords is None or name not in coords:
                    raise ValueError(f"PositionEncoder(pos_encoding='fourier') requires coords['{name}'] aligned with inputs.")
                c = coords[name]
                if c.shape[0] != value.shape[0] or c.shape[1] != value.shape[1]:
                    raise ValueError(f"coords['{name}'] must be [B,N,*] aligned with inputs; got {tuple(c.shape)} vs {tuple(value.shape)}")
                coord_dim = int(c.shape[-1])
                if name not in self._fourier:
                    self._fourier[name] = FourierFeatures(coord_dim=coord_dim, num_bands=self.fourier_bands, max_freq=self.fourier_max_freq).to(device)
                    proj = nn.Linear(self._fourier[name].out_dim, self.num_channels, bias=False).to(device)
                    nn.init.trunc_normal_(proj.weight, std=1.0 / sqrt(self._fourier[name].out_dim))
                    self._fourier_proj[name] = proj
                pos_enc = self._fourier_proj[name](self._fourier[name](c))
            else:
                raise ValueError(f"Unknown pos_encoding={self.pos_encoding!r}. Use 'learned' or 'fourier'.")
            
            # Project if needed (JAX supports different pos encoding channels)
            if pos_enc_channels != self.num_channels:
                if self._pos_proj is None:
                    self._pos_proj = nn.Linear(pos_enc_channels, self.num_channels, bias=False).to(device)
                    nn.init.trunc_normal_(self._pos_proj.weight, std=1.0 / sqrt(pos_enc_channels))
                pos_enc = self._pos_proj(pos_enc)
            
            pos_encodings[name] = pos_enc
            out[name] = value + pos_enc
        
        return out, pos_encodings

class ReconstructionHead(nn.Module):
    """Produces reconstruction from latents using cross-attention to position queries.
    
    This uses external query_inputs (position encodings), NOT learned latent queries.
    Therefore it doesn't need to know output_index_dim at init time.
    
    The cross-attention maps: latents [B, G, N, C_latent] -> output [B, G, M, C_out]
    where M comes from the mae_query at runtime.
    """
    
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        heads: int = 1,
        widening_factor: int = 1,
    ):
        super().__init__()
        # Normalization
        self.norm_latents = nn.LayerNorm(latent_dim)
        self.norm_query = nn.LayerNorm(output_dim)
        
        # Cross-attention: latents -> query positions
        # Uses external queries (mae_query), not learned queries
        self.attn = Attention(
            query_dim=output_dim,
            context_dim=latent_dim,
            heads=heads,
            qk_channels=latent_dim,
            v_channels=latent_dim,
            output_channels=output_dim,
            dropout=0.0,
        )
        
        # Feedforward
        self.norm_ff = nn.LayerNorm(output_dim)
        self.ff = FeedForward(output_dim, widening_factor=widening_factor, dropout=0.0)
    
    def forward(self, latents: Tensor, mae_query: Tensor) -> Tensor:
        """
        Args:
            latents: [B, G, N, C_latent] final latents from encoder
            mae_query: [B, G, M, C_out] position encodings to decode to
        
        Returns:
            [B, G, M, C_out] reconstructions
        """
        # Cross-attention: query (mae positions) attends to latents
        attn_out = self.attn(
            self.norm_query(mae_query),
            context=self.norm_latents(latents)
        )
        
        # No post-attention residual (following JAX: use_post_attention_residual=False)
        out = attn_out + self.ff(self.norm_ff(attn_out))
        
        return out

# ==============================================================================
# HiP Model Variants
# ==============================================================================

HIP_VARIANTS = {
    # CIFAR-10 variants

    # 2k floats: G=32, K=4, C=16 → 32×4×16 = 2,048
    'C10_4STAGE_G128_LAST32_C16_K4': {
        'num_groups': (128, 64, 32, 1, 32, 64, 128),
        'num_self_attends_per_block': (1, 2, 18, 2, 2, 1, 1),
        'z_index_dim': (4, 4, 4, 128, 4, 4, 4),
        'num_z_channels': (64, 96, 16, 32, 16, 96, 64),
        'num_cross_attend_heads': (1, 1, 1, 1, 1, 1, 1),
        'num_self_attend_heads': (4, 4, 4, 8, 4, 4, 4),
        'cross_attend_widening_factor': (1, 1, 1, 1, 1, 1, 1),
        'self_attend_widening_factor': (4, 4, 4, 4, 4, 4, 4),
        'num_embedding_channels': 32,
    },

    # ShapeNet16 3D voxel occupancy variants (32^3 occupancy grids, coord_dim=3, value_dim=1).

    # 1.5k floats: G=64, K=3, C=8 → 64×3×8 = 1,536
    'SN16_G64_K3_C8': {
        'num_groups':                    (256, 128,  64,   1,  64, 128, 256),
        'num_self_attends_per_block':    (  1,   2,  24,   2,   2,   1,   1),
        'z_index_dim':                   (  4,   4,   3, 256,   3,   4,   4),
        'num_z_channels':                ( 64,  32,   8,  64,   8,  32,  64),
        'num_cross_attend_heads':        (  1,   1,   1,   1,   1,   1,   1),
        'num_self_attend_heads':         (  4,   4,   2,   8,   2,   4,   4),
        'cross_attend_widening_factor':  (  1,   1,   1,   1,   1,   1,   1),
        'self_attend_widening_factor':   (  4,   4,   4,   4,   4,   4,   4),
        'num_embedding_channels': 32,
    },

    # ERA5
    # 2,304 floats: G=72, K=2, C=16 → 72×2×16 = 2,304
    'ERA5_G72_K2_C16_3B': {
        'num_groups':                    (288, 144,  72,   1,  72, 144, 288),
        'num_self_attends_per_block':    (  1,   3,  36,   4,   1,   1,   1),
        'z_index_dim':                   (  4,   4,   2, 288,   2,   4,   4),
        'num_z_channels':                ( 96, 128,  16, 256,  16, 128,  96),
        'num_cross_attend_heads':        (  1,   1,   1,   1,   1,   1,   1),
        'num_self_attend_heads':         (  4,   8,   4,   8,   4,   8,   4),
        'cross_attend_widening_factor':  (  1,   1,   1,   1,   1,   1,   1),
        'self_attend_widening_factor':   (  4,   4,   4,   4,   4,   4,   4),
        'num_embedding_channels': 32,
    },

    # CelebA-HQ 64x64 variants

    # 3K floats: G=64, K=2, C=24 → 64×2×24 = 3,072
    'CELEBAHQ_64_LAST64_K2_C24': {
        'num_groups': (256, 128, 64, 1, 64, 128, 256),
        'num_self_attends_per_block': (1, 3, 36, 4, 1, 1, 1),
        'z_index_dim': (4, 4, 2, 256, 2, 4, 4),
        'num_z_channels': (96, 128, 24, 256, 24, 128, 96),
        'num_cross_attend_heads': (1, 1, 1, 1, 1, 1, 1),
        'num_self_attend_heads': (4, 8, 8, 8, 8, 8, 4),
        'cross_attend_widening_factor': (1, 1, 1, 1, 1, 1, 1),
        'self_attend_widening_factor': (4, 4, 4, 4, 4, 4, 4),
        'num_embedding_channels': 32,
    },

    # =========================================================================
    # ImageNet variants (encoder at 128x128 = 16384 tokens, renderer at 256x256).
    # =========================================================================

    # 14.3K floats: G=128, K=4, C=28 → 128×4×28 = 14,336 (paper variant)
    'IN128_G128_K4_C28': {
        'num_groups':                    (1024, 512, 256, 128,   1, 128, 256, 512, 1024),
        'num_self_attends_per_block':    (   1,   2,  24,   6,   2,   1,   1,   1,    1),
        'z_index_dim':                   (   4,   4,   4,   4, 256,   4,   4,   4,    4),
        'num_z_channels':                ( 64,  96, 128,  28, 256,  28, 128,  96,   64),
        'num_cross_attend_heads':        (   1,   1,   1,   1,   1,   1,   1,   1,    1),
        'num_self_attend_heads':         (   4,   4,   8,   4,   8,   4,   8,   4,    4),
        'cross_attend_widening_factor':  (   1,   1,   1,   1,   1,   1,   1,   1,    1),
        'self_attend_widening_factor':   (   4,   4,   4,   4,   4,   4,   4,   4,    4),
        'num_embedding_channels': 32,
    },

}

# ==============================================================================
# Main HiP Model
# ==============================================================================

class HiP(nn.Module):
    """Hierarchical Perceiver (HiP).
    
    A U-Net style architecture with:
    - Encoder blocks: Decreasing number of groups (more global)
    - Processor block: Single group (fully global attention)  
    - Decoder blocks: Increasing number of groups (more local)
    - Skip connections between encoder and decoder
    
    Paper: https://arxiv.org/abs/2202.10890
    """
    
    def __init__(
        self,
        # Per-block configuration (lists of length num_blocks)
        num_groups: Sequence[int],
        num_self_attends_per_block: Sequence[int],
        z_index_dim: Sequence[int],
        num_z_channels: Sequence[int],
        num_cross_attend_heads: Sequence[int],
        num_self_attend_heads: Sequence[int],
        cross_attend_widening_factor: Sequence[int],
        self_attend_widening_factor: Sequence[int],
        # Global configuration
        num_embedding_channels: int,
        coord_grouping: str = "none",
        coord_bits: int = 8,
        coord_range: Tuple[float, float] = (-1.0, 1.0),
        normalize_sphere_centers: bool = False,
        build_decoder: bool = True,
        build_reconstruction_head: bool = True,
        regroup_type: str = 'reshape',
        activation: str = 'sq_relu',
        dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        use_post_attention_residual: bool = True,
        pos_encoding: str = 'learned',
        fourier_bands: int = 64,
        fourier_max_freq: float = 10.0,
    ):
        super().__init__()

        build_decoder = bool(build_decoder)
        build_reconstruction_head = bool(build_reconstruction_head)

        full_num_blocks = len(num_groups)
        assert full_num_blocks >= 3, 'At least 3 blocks needed for HiP'
        assert full_num_blocks % 2 == 1, 'HiP needs odd number of blocks'

        processor_idx_full = full_num_blocks // 2
        assert num_groups[processor_idx_full] == 1, 'Processor must have 1 group'

        if not build_decoder:
            keep = int(processor_idx_full) + 1

            def _trunc(x):
                return list(x)[:keep]

            num_groups = _trunc(num_groups)
            num_self_attends_per_block = _trunc(num_self_attends_per_block)
            z_index_dim = _trunc(z_index_dim)
            num_z_channels = _trunc(num_z_channels)
            num_cross_attend_heads = _trunc(num_cross_attend_heads)
            num_self_attend_heads = _trunc(num_self_attend_heads)
            cross_attend_widening_factor = _trunc(cross_attend_widening_factor)
            self_attend_widening_factor = _trunc(self_attend_widening_factor)

            self.num_blocks = keep
            self.processor_idx = processor_idx_full  # last block index after truncation
        else:
            self.num_blocks = full_num_blocks
            self.processor_idx = processor_idx_full

        self.num_groups = list(num_groups)
        self.z_index_dim = list(z_index_dim)
        self.num_z_channels = list(num_z_channels)
        self.num_embedding_channels = num_embedding_channels
        self.regroup_type = regroup_type
        self.use_post_attention_residual = use_post_attention_residual
        self.build_decoder = bool(build_decoder)
        self.build_reconstruction_head = bool(build_reconstruction_head)
        self.num_embedding_channels = num_embedding_channels
        self.regroup_type = regroup_type
        self.use_post_attention_residual = use_post_attention_residual
        
        self.embedder = Embedder(num_embedding_channels)
        self.pos_encoder = PositionEncoder(
            num_embedding_channels,
            pos_encoding=pos_encoding,
            fourier_bands=fourier_bands,
            fourier_max_freq=fourier_max_freq,
        )

        self.coord_grouping = str(coord_grouping or "none").lower().strip()
        if self.coord_grouping in ("kd", "kdtree", "kd_tree", "kd-tree"):
            self.coord_grouping = "kdtree"
        if self.coord_grouping in ("healpy", "healpix", "heal"):
            self.coord_grouping = "healpix"
        if self.coord_grouping in ("s2", "s2cell", "s2_cell"):
            self.coord_grouping = "s2"
        self.coord_bits = int(coord_bits)
        self.coord_range = (float(coord_range[0]), float(coord_range[1]))
        if self.coord_grouping not in ("none", "voxel", "kdtree", "healpix", "s2"):
            raise ValueError(f"coord_grouping must be one of: none|voxel|kdtree|healpix|s2, got {self.coord_grouping!r}")
        if self.coord_bits <= 0 or self.coord_bits > 30:
            raise ValueError(f"coord_bits must be in [1,30], got {self.coord_bits}")
        self.normalize_sphere_centers = bool(normalize_sphere_centers)
        if not (self.coord_range[0] < self.coord_range[1]):
            raise ValueError(f"coord_range must be (min,max) with min<max, got {self.coord_range}")
        
        self.grouper = ConstNumGrouper(self.num_groups[0])
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            # Input dimension is from previous block (or embedding)
            if i == 0:
                input_dim = num_embedding_channels
            else:
                input_dim = num_z_channels[i - 1]
            
            # For decoder blocks (i > processor_idx), compute residual dimension
            # from the corresponding encoder skip connection
            if i > self.processor_idx:
                skip_idx = self.num_blocks - i - 1
                residual_dim = self.num_z_channels[skip_idx]
            else:
                residual_dim = None  # Encoder blocks don't receive skips
            
            block = PerceiverBlock(
                input_dim=input_dim,
                output_dim=self.num_z_channels[i],
                num_output_groups=self.num_groups[i],
                output_index_dim=self.z_index_dim[i],
                num_self_attend_layers=num_self_attends_per_block[i],
                num_self_attend_heads=num_self_attend_heads[i],
                self_attend_widening_factor=self_attend_widening_factor[i],
                num_cross_attend_heads=num_cross_attend_heads[i],
                cross_attend_widening_factor=cross_attend_widening_factor[i],
                regroup_inputs=(i > 0),  # First block doesn't regroup (grouper does it)
                regroup_type=regroup_type,
                dropout=dropout,
                drop_path_rate=drop_path_rate,
                activation=activation,
                use_post_attention_residual=use_post_attention_residual,
                residual_dim=residual_dim
            )
            self.blocks.append(block)

        self.reconstruction_head = None
        if self.build_reconstruction_head:
            self.reconstruction_head = ReconstructionHead(
                latent_dim=self.num_z_channels[-1],
                output_dim=num_embedding_channels,
                heads=1,
                widening_factor=1,
            )

        # Group-region cache toggle for grid data. Off by default
        self._cache_group_regions = False
        self._group_regions_cache = None    # cached output dict
        self._group_regions_cache_key = None  # (data_ptr, shape, dtype, device)

    def set_cache_group_regions(self, cache: bool) -> None:
        """When True, cache the per-block group-region tensors keyed by the
        input coord tensor's storage. For inputs whose coords are constant
        (e.g. uniform image grids), this skips the per-step Python loop in
        ``_compute_group_regions_single_modality``. Set to False to disable
        and reset the cache."""
        self._cache_group_regions = bool(cache)
        if not cache:
            self._group_regions_cache = None
            self._group_regions_cache_key = None

    def _group_regions_cache_lookup(self, coords: Tensor):
        if not self._cache_group_regions or self._group_regions_cache is None:
            return None
        key = (int(coords.untyped_storage().data_ptr()),
               tuple(coords.shape), coords.dtype, coords.device)
        if key == self._group_regions_cache_key:
            return self._group_regions_cache
        return None

    def _group_regions_cache_store(self, coords: Tensor, regions) -> None:
        if not self._cache_group_regions:
            return
        self._group_regions_cache_key = (
            int(coords.untyped_storage().data_ptr()),
            tuple(coords.shape), coords.dtype, coords.device,
        )
        self._group_regions_cache = regions

    def forward(
        self, 
        inputs: Dict[str, Tensor],
        coords: Optional[Dict[str, Tensor]] = None,
        return_latents: bool = False,
        return_bottleneck: bool = False,
        return_pyramid: bool = False,
        pyramid_hw: Optional[Dict[str, Dict[int, Tuple[int, int]]]] = None,
        pyramid_blocks: Optional[Sequence[int]] = None,
        # NEW: expose grouped block outputs (token sets) for downstream decoders/renderers.
        # This is primarily intended for encoder-side blocks where group->input provenance is well-defined.
        return_block_latents: bool = False,
        block_indices: Optional[Sequence[int]] = None,
        # Perf: optionally skip the decoder half (blocks > processor_idx) entirely.
        # This is useful when downstream only consumes encoder-side block latents (e.g., group-aware renderers).
        run_decoder: bool = True,
        # Perf: optionally skip the reconstruction head.
        # HiPEncoderLHNeF does not use reconstructions; it only needs block latents and/or bottleneck.
        return_reconstruction: bool = True,
        # NEW: return per-block group regions for coordinate-space routing (centers/scales).
        # Only supported for single-modality inputs.
        return_group_regions: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Args:
            inputs: Dict of {modality_name: [B, N, C]} tensors
            return_latents: Whether to return final decoder latents (expanded)
            return_bottleneck: Whether to return bottleneck (processor) latents (compressed)
            return_pyramid: Whether to return a multi-resolution grid pyramid per modality.
            pyramid_hw: Optional explicit mapping used to reshape token sequences into grids
                without guessing. Format:
                    {
                      "image": {
                        576: (24, 24),
                        36: (6, 6),
                      },
                      ...
                    }
                Keys are modality names, inner keys are token counts N, values are (H, W).
                Only entries present in this mapping will be returned in the pyramid.
        
        Returns:
            Dict with 'reconstruction' and optionally 'latents' and/or 'bottleneck'
        """
        # Capture a stable identity-fingerprint of each modality's coord
        # tensor BEFORE the locality-preserving permutation loop below mutates coords[name].
        cache_id_coords: Dict[str, Tensor] = {}
        if coords is not None:
            for name, c in coords.items():
                if c is not None:
                    cache_id_coords[name] = c

        if self.coord_grouping != "none":
            if coords is None:
                raise ValueError("HiP.forward: coord_grouping requires coords to be provided.")
            # Avoid mutating caller-owned dicts.
            inputs = dict(inputs)
            coords = dict(coords)
            for name, x in list(inputs.items()):
                c = coords.get(name, None)
                if c is None:
                    raise ValueError(f"HiP.forward: coord_grouping requires coords[{name!r}] to be provided.")
                if c.shape[:2] != x.shape[:2]:
                    raise ValueError(f"HiP.forward: inputs[{name!r}] has shape {tuple(x.shape)} but coords has {tuple(c.shape)}")
                perm = self._coord_sort_perm(c)
                inputs[name] = self._gather_seq(x, perm)
                coords[name] = self._gather_seq(c, perm)

        z_0 = self.embedder.embed(inputs)
        z, mae_query = self.pos_encoder(z_0, coords=coords)

        pyramid = None
        if return_pyramid:
            pyramid = {}
            hw_map = pyramid_hw or {}
            for name, value in z.items():
                n = int(value.shape[1])
                hw = (hw_map.get(name, {}) or {}).get(n, None)
                if hw is None:
                    continue
                h, w = int(hw[0]), int(hw[1])
                if h * w != n:
                    raise ValueError(f"pyramid_hw for {name!r} maps N={n} to {(h,w)} but H*W != N")
                grid = value.reshape(value.shape[0], h, w, value.shape[2]).permute(0, 3, 1, 2).contiguous()
                pyramid.setdefault(name, {})[f"{h}x{w}"] = grid
        
        z = self.grouper.group(z)
        mae_query = self.grouper.group(mae_query)
        
        run_decoder = bool(run_decoder)
        return_reconstruction = bool(return_reconstruction)
        if (not self.build_decoder) and run_decoder:
            run_decoder = False
        if (not run_decoder) and bool(return_latents):
            raise ValueError("HiP.forward: return_latents=True requires run_decoder=True (latents are final decoder outputs).")

        want_group_regions = bool(return_group_regions)
        group_regions = None
        if want_group_regions:
            if len(inputs) != 1:
                raise ValueError("return_group_regions currently supports only single-modality inputs.")
            name = next(iter(inputs.keys()))
            c0 = coords.get(name, None) if coords is not None else None
            if c0 is None:
                raise ValueError("return_group_regions=True requires coords to be provided for the single modality.")
            
            # Cache key uses the caller's original coord tensor as an identity fingerprint
            cache_coord = cache_id_coords.get(name, c0)
            cached = self._group_regions_cache_lookup(cache_coord)
            if cached is not None:
                group_regions = cached
            else:
                group_regions = self._compute_group_regions_single_modality(
                    coords=c0,
                    num_groups=list(self.num_groups[: self.processor_idx + 1]),
                    z_index_dim=list(self.z_index_dim[: self.processor_idx + 1]),
                    normalize_centers_to_sphere=self.normalize_sphere_centers,
                )
                self._group_regions_cache_store(cache_coord, group_regions)

        hidden_z = []
        want_block_latents = bool(return_block_latents)
        block_idx_set = set(int(i) for i in (block_indices or [])) if want_block_latents else set()
        block_latents = {} if want_block_latents else None
        if run_decoder:
            max_i = int(self.num_blocks - 1)
        else:
            needs_bottleneck = (
                bool(return_bottleneck)
                or (self.processor_idx in block_idx_set)
                or (return_pyramid and self.processor_idx in set(int(i) for i in (pyramid_blocks or [])))
            )
            max_i = int(self.processor_idx) if needs_bottleneck else int(self.processor_idx) - 1
        for i, block in enumerate(self.blocks):
            if i > max_i:
                break
            if i > self.processor_idx:
                skip_idx = self.num_blocks - i - 1
                pre_attention_residual = hidden_z[skip_idx]
            else:
                pre_attention_residual = None
            
            z = block(z, pre_attention_residual=pre_attention_residual)
            hidden_z.append(z)

            if want_block_latents and (i in block_idx_set):
                if len(inputs) != 1:
                    raise ValueError("return_block_latents currently supports only single-modality inputs.")
                block_latents[f"block{i}"] = z

            if return_pyramid and (pyramid is not None) and (pyramid_blocks is not None) and (i in set(pyramid_blocks)):
                hw_map = pyramid_hw or {}
                if len(inputs) == 1:
                    name = next(iter(inputs.keys()))
                    n_total = int(z.shape[1] * z.shape[2])  # G*N
                    hw = (hw_map.get(name, {}) or {}).get(n_total, None)
                    if hw is not None:
                        h, w = int(hw[0]), int(hw[1])
                        if h * w != n_total:
                            raise ValueError(f"pyramid_hw for {name!r} maps N={n_total} to {(h,w)} but H*W != N")
                        flat = z.reshape(z.shape[0], n_total, z.shape[3])
                        grid = flat.reshape(flat.shape[0], h, w, flat.shape[2]).permute(0, 3, 1, 2).contiguous()
                        pyramid.setdefault(name, {})[f"{h}x{w}"] = grid
        
        output: Dict[str, Tensor] = {}
        # Reconstruction (optional; requires full run through decoder blocks)
        if return_reconstruction:
            if not run_decoder:
                raise ValueError("HiP.forward: return_reconstruction=True requires run_decoder=True.")
            if self.reconstruction_head is None:
                raise ValueError("HiP.forward: reconstruction_head is not built (build_reconstruction_head=false).")
            reconstruction_z = self.reconstruction_head(z, mae_query)
            reconstruction_z = self.grouper.ungroup(reconstruction_z)
            reconstruction = self.embedder.unembed(reconstruction_z)
            output['reconstruction'] = reconstruction
        
        if return_latents:
            output['latents'] = self.grouper.ungroup(z)

        if return_pyramid:
            if run_decoder and int(self.num_groups[-1]) == int(self.grouper.num_groups):
                hw_map = pyramid_hw or {}
                lat_dict = self.grouper.ungroup(z)
                for name, value in lat_dict.items():
                    n = int(value.shape[1])
                    hw = (hw_map.get(name, {}) or {}).get(n, None)
                    if hw is None:
                        continue
                    h, w = int(hw[0]), int(hw[1])
                    if h * w != n:
                        raise ValueError(f"pyramid_hw for {name!r} maps N={n} to {(h,w)} but H*W != N")
                    grid = value.reshape(value.shape[0], h, w, value.shape[2]).permute(0, 3, 1, 2).contiguous()
                    pyramid.setdefault(name, {})[f"{h}x{w}"] = grid

            output['pyramid'] = pyramid or {}

        if want_block_latents:
            output['block_latents'] = block_latents or {}

        if want_group_regions:
            output["group_regions"] = group_regions or {}
        
        if return_bottleneck:
            bottleneck = hidden_z[self.processor_idx]
            B, G, N, C = bottleneck.shape
            output['bottleneck'] = bottleneck.reshape(B, G * N, C)
            output['bottleneck_dim'] = C
        
        return output

    def _gather_seq(self, x: Tensor, perm: Tensor) -> Tensor:
        """Gather along the token dimension (dim=1) with a per-sample permutation."""
        if perm.ndim != 2:
            raise ValueError(f"perm must be [B,N], got {tuple(perm.shape)}")
        return torch.gather(x, dim=1, index=perm[..., None].expand(-1, -1, x.shape[-1]))

    def _coord_sort_perm(self, coords: Tensor) -> Tensor:
        """Return per-sample permutation that (approximately) preserves coord locality."""
        # coords: [B, N, d]
        if coords.ndim != 3:
            raise ValueError(f"coords must be [B,N,d], got {tuple(coords.shape)}")
        lo, hi = self.coord_range
        bits = self.coord_bits
        B, N, d = coords.shape

        def _coords_same_across_batch() -> bool:
            if int(B) <= 1:
                return True
            probe = torch.tensor([0, int(N // 3), int(2 * N // 3), int(N - 1)], device=coords.device, dtype=torch.long)
            ref = coords[0].index_select(0, probe)
            max_b = min(int(B) - 1, 2)
            for bb in range(1, max_b + 1):
                cur = coords[bb].index_select(0, probe)
                if not torch.equal(cur, ref):
                    return False
            return True

        same_batch = _coords_same_across_batch()
        if same_batch:
            sig_idx = torch.tensor([0, 1, 2, 3, max(0, int(N - 4)), max(0, int(N - 3)), max(0, int(N - 2)), max(0, int(N - 1))],
                                   device=coords.device, dtype=torch.long)
            sig = coords[0].index_select(0, sig_idx).detach().to('cpu')
            cache_key = (
                str(self.coord_grouping),
                int(bits),
                float(lo), float(hi),
                int(N), int(d),
                str(coords.device),
                str(coords.dtype),
                sig.numpy().tobytes(),
            )
            cache = getattr(self, "_coord_perm_cache", None)
            if cache is None:
                cache = {}
                setattr(self, "_coord_perm_cache", cache)
            perm0 = cache.get(cache_key, None)
            if perm0 is not None:
                return perm0.expand(int(B), -1)

        if self.coord_grouping == "kdtree":
            import numpy as _np

            c_cpu = coords.detach().to("cpu").numpy()  # [B,N,d]
            perms = []

            def kd_order(points: _np.ndarray) -> _np.ndarray:
                idx = _np.arange(points.shape[0], dtype=_np.int64)

                def rec(idxs: _np.ndarray) -> _np.ndarray:
                    if idxs.size <= 1:
                        return idxs
                    pts = points[idxs]
                    span = pts.max(axis=0) - pts.min(axis=0)
                    axis = int(span.argmax())
                    sidx = idxs[_np.argsort(points[idxs, axis], kind="mergesort")]
                    mid = int(sidx.size // 2)
                    return _np.concatenate([rec(sidx[:mid]), rec(sidx[mid:])], axis=0)

                return rec(idx)

            for b in range(int(B)):
                perms.append(kd_order(c_cpu[b]))
            perm = torch.from_numpy(_np.stack(perms, axis=0)).to(device=coords.device, dtype=torch.long)
            if same_batch:
                perm0 = perm[:1].contiguous()
                getattr(self, "_coord_perm_cache")[cache_key] = perm0
                return perm0.expand(int(B), -1)
            return perm

        if self.coord_grouping == "healpix":
            import numpy as _np
            import healpy as _hp

            c_cpu = coords.detach().to("cpu").numpy()  # [B,N,d]
            if d != 3:
                raise ValueError(f"healpix coord_grouping requires coord_dim=3 (unit sphere xyz), got d={d}")
            perms = []
            for b in range(int(B)):
                xyz = c_cpu[b]  # [N, 3]
                # Convert (x,y,z) on unit sphere to HEALPix (colatitude theta, azimuth phi).
                # healpy.ang2pix expects theta in [0, pi] (colatitude) and phi in [0, 2pi).
                r = _np.sqrt((xyz ** 2).sum(axis=-1)).clip(1e-12)
                theta = _np.arccos(_np.clip(xyz[:, 2] / r, -1.0, 1.0))  # colatitude from z
                phi = _np.arctan2(xyz[:, 1], xyz[:, 0]) % (2.0 * _np.pi)  # azimuth from x,y
                # Choose nside to give at least N pixels (12 * nside^2 >= N).
                nside = max(1, int(_np.ceil(_np.sqrt(int(N) / 12.0))))
                # Round up to power of 2 (required for NESTED scheme).
                nside = int(2 ** int(_np.ceil(_np.log2(max(nside, 1)))))
                # NESTED scheme gives a locality-preserving pixel index (hierarchical
                # quad-tree subdivision of the 12 HEALPix base pixels).
                pix = _hp.ang2pix(nside, theta, phi, nest=True)
                perms.append(_np.argsort(pix, kind="stable").astype(_np.int64))
            perm = torch.from_numpy(_np.stack(perms, axis=0)).to(device=coords.device, dtype=torch.long)
            if same_batch:
                perm0 = perm[:1].contiguous()
                getattr(self, "_coord_perm_cache")[cache_key] = perm0
                return perm0.expand(int(B), -1)
            return perm

        if self.coord_grouping == "s2":
            import numpy as _np
            from s2cell import lat_lon_to_token as _s2_token

            c_cpu = coords.detach().to("cpu").numpy()  # [B,N,d]
            if d != 3:
                raise ValueError(f"s2 coord_grouping requires coord_dim=3 (unit sphere xyz), got d={d}")
            # S2 cell level controls resolution. Level L gives 6 * 4^L cells.
            # Choose level so that total cells >= N for fine-grained ordering.
            s2_level = max(1, min(30, int(_np.ceil(_np.log(max(int(N), 1) / 6.0) / _np.log(4.0)))))
            perms = []
            for b in range(int(B)):
                xyz = c_cpu[b]  # [N, 3]
                r = _np.sqrt((xyz ** 2).sum(axis=-1)).clip(1e-12)
                # Convert to lat/lon in degrees for s2cell.
                lat_deg = _np.degrees(_np.arcsin(_np.clip(xyz[:, 2] / r, -1.0, 1.0)))
                lon_deg = _np.degrees(_np.arctan2(xyz[:, 1], xyz[:, 0]))
                # s2cell.lat_lon_to_token returns hex string S2 cell tokens.
                # Sorting these strings gives Hilbert-curve order on the sphere
                # because S2 cell IDs are constructed from face + Hilbert position.
                tokens = [_s2_token(float(lat_deg[i]), float(lon_deg[i]), s2_level)
                          for i in range(int(N))]
                perms.append(_np.argsort(tokens, kind="stable").astype(_np.int64))
            perm = torch.from_numpy(_np.stack(perms, axis=0)).to(device=coords.device, dtype=torch.long)
            if same_batch:
                perm0 = perm[:1].contiguous()
                getattr(self, "_coord_perm_cache")[cache_key] = perm0
                return perm0.expand(int(B), -1)
            return perm

        t = (coords - lo) / (hi - lo)
        t = t.clamp(0.0, 1.0)
        qmax = (1 << bits) - 1
        q = torch.round(t * float(qmax)).to(torch.int64)

        if d <= 3:
            # Proper Z-curve / Morton bit-interleaving for 2D and 3D
            key = torch.zeros((B, N), device=q.device, dtype=torch.int64)
            for b in range(bits):
                for j in range(d):
                    key |= ((q[..., j] >> b) & 1) << (b * d + j)
        else:
            # Fallback to lexicographic for d > 3
            base = (1 << bits)
            key = q[..., 0].clone()
            for j in range(1, d):
                key = key * base + q[..., j]

        perm = torch.argsort(key, dim=1, stable=True)
        if same_batch:
            perm0 = perm[:1].contiguous()
            getattr(self, "_coord_perm_cache")[cache_key] = perm0
            return perm0.expand(int(B), -1)
        return perm

    def _compute_group_regions_single_modality(
        self,
        coords: Tensor,
        num_groups: List[int],
        z_index_dim: List[int],
        eps: float = 1e-6,
        normalize_centers_to_sphere: bool = False,
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Build per-block group regions in coordinate space, using the same "contiguous interval"
        provenance used by reshape-regrouping.

        Returns a dict:
          {
            "routing_space": "coord",
            "coord_dim": d,
            "blocks": {
               "block0": {"centers": [B,G,d], "scales": [B,G,d], "ends": [G+1] (int64 on CPU)},
               ...
            }
          }
        """
        if coords.ndim != 3:
            raise ValueError(f"coords must be [B,N,d], got {tuple(coords.shape)}")
        B, N0, d = coords.shape
        if len(num_groups) != len(z_index_dim):
            raise ValueError("num_groups and z_index_dim must have the same length for encoder-side blocks.")
        if len(num_groups) < 1:
            raise ValueError("need at least one block to compute group regions")

        G0 = int(num_groups[0])
        pad0 = int(padding_to_make_divisible(int(N0), int(G0)))
        Npad = int(N0 + pad0)
        if pad0 > 0:
            coords_pad = torch.zeros((B, pad0, d), device=coords.device, dtype=coords.dtype)
            coords_all = torch.cat([coords, coords_pad], dim=1)  # [B,Npad,d]
        else:
            coords_all = coords
        valid = torch.arange(Npad, device=coords.device)[None, :] < int(N0)  # [1,Npad]

        # Token supports in the "sorted token index" space [0, Npad), where padding tokens map to an empty support.
        token_supports: List[Tuple[int, int]] = [(i, i + 1) for i in range(int(N0))] + [(int(N0), int(N0))] * pad0

        blocks: Dict[str, Dict[str, Tensor]] = {}
        for bi, (G, K) in enumerate(zip(num_groups, z_index_dim)):
            G = int(G)
            K = int(K)
            Nin = len(token_supports)
            if Nin % G != 0:
                raise ValueError(f"Cannot compute group regions: Nin={Nin} not divisible by G={G} at block{bi}.")
            per_group = Nin // G
            ends: List[int] = [0]
            centers = torch.zeros((B, G, d), device=coords.device, dtype=coords.dtype)
            scales = torch.ones((B, G, d), device=coords.device, dtype=coords.dtype)
            bbox_min = torch.zeros((B, G, d), device=coords.device, dtype=coords.dtype)
            bbox_max = torch.zeros((B, G, d), device=coords.device, dtype=coords.dtype)

            next_token_supports: List[Tuple[int, int]] = []
            for g in range(G):
                a = g * per_group
                b = (g + 1) * per_group
                s0 = token_supports[a][0]
                e1 = token_supports[b - 1][1]
                ends.append(int(e1))

                # Compute center/scale from underlying original coords slice [s0:e1).
                if e1 <= s0:
                    # Empty region (only padding)
                    ctr = torch.zeros((B, d), device=coords.device, dtype=coords.dtype)
                    scl = torch.ones((B, d), device=coords.device, dtype=coords.dtype)
                    cmin = ctr
                    cmax = ctr
                else:
                    c_slice = coords_all[:, s0:e1, :]  # [B,m,d]
                    m_slice = valid[:, s0:e1].to(coords.dtype)  # [1,m]
                    mask = m_slice[0].view(1, -1, 1)  # [1,m,1] -> broadcast over batch
                    denom = mask.sum(dim=1).clamp_min(1.0)  # [1,1]
                    ctr = (c_slice * mask).sum(dim=1) / denom  # [B,d] arithmetic mean
                    # bbox extents as scale
                    m_bool = (valid[:, s0:e1])[0]  # [m]
                    if bool(m_bool.any().item()):
                        c_valid = c_slice[:, m_bool, :]
                        cmin = c_valid.amin(dim=1)
                        cmax = c_valid.amax(dim=1)
                        scl = (cmax - cmin).clamp_min(eps)

                        # Centroid is Frechet (Karcher) mean on manifolds (era5)
                        if bool(normalize_centers_to_sphere) and int(d) == 3:
                            init_mu = ctr / ctr.norm(dim=-1, keepdim=True).clamp_min(eps)
                            ctr = _karcher_mean_s2(c_valid, init=init_mu, n_iters=5, eps=eps)
                    else:
                        cmin = ctr
                        cmax = ctr
                        scl = torch.ones((B, d), device=coords.device, dtype=coords.dtype)

                centers[:, g, :] = ctr
                scales[:, g, :] = scl
                bbox_min[:, g, :] = cmin
                bbox_max[:, g, :] = cmax

                # Propagate support to next block: each output token in this group depends on the whole group's support.
                next_token_supports.extend([(s0, e1)] * K)

            # For spherical data (e.g. ERA5 on the unit sphere), project group
            # centers back onto the sphere so they remain valid surface points.
            # Without this, the mean of points on a sphere drifts toward the
            # interior, biasing KNN distances for groups that span large arcs.
            if normalize_centers_to_sphere:
                norms = centers.norm(dim=-1, keepdim=True).clamp_min(eps)  # [B,G,1]
                centers = centers / norms

            blocks[f"block{bi}"] = {
                "G": torch.tensor(G, device="cpu", dtype=torch.int64),
                "K": torch.tensor(K, device="cpu", dtype=torch.int64),
                "centers": centers,
                "scales": scales,
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "ends": torch.tensor(ends, device="cpu", dtype=torch.int64),
            }
            token_supports = next_token_supports

        return {
            "routing_space": "coord",
            "coord_dim": torch.tensor(d, device="cpu", dtype=torch.int64),
            "blocks": blocks,
        }
    
    @classmethod
    def from_variant(cls, variant_name: str, **kwargs) -> 'HiP':
        """Create HiP from a predefined variant."""
        if variant_name not in HIP_VARIANTS:
            raise ValueError(f'Unknown variant: {variant_name}. Choose from {list(HIP_VARIANTS.keys())}')
        
        config = HIP_VARIANTS[variant_name].copy()
        config.update(kwargs)
        return cls(**config)

# ==============================================================================
# Test / Demo
# ==============================================================================

if __name__ == '__main__':
    # Test basic HiP
    print("Testing HiP...")
    model = HiP.from_variant('Mini')
    
    # Create dummy input
    batch_size = 2
    num_pixels = 256
    inputs = {
        'image': torch.randn(batch_size, num_pixels, 3)
    }
    
    output = model(inputs, return_latents=True)
    print(f"Input shape: {inputs['image'].shape}")
    print(f"Reconstruction shape: {output['reconstruction']['image'].shape}")
    print(f"Latents shape: {output['latents']['image'].shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nHiP Mini params: {total_params:,}")
