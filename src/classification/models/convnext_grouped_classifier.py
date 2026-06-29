"""ConvNeXt classifier on the LH-NeF grouped token representation.

Treats ``[B, G, K, C]`` grouped tokens as a 2D feature map by sorting groups and
within-group slots into row-major grids ``(Hg, Wg)`` and ``(Hk, Wk)``, then
reshaping to ``[B, C, Hg*Hk, Wg*Wk]``. The 2D arrangement is exact for Morton/voxel
grouping on a regular sampling; for kd-tree groupings it is only approximate.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models import register

from .latent_augs import LatentAugMixer

__all__ = ["ConvNeXtGroupedClassifier"]


def _lex_sort_perm_2d(coords: torch.Tensor) -> torch.Tensor:
    """Return permutation that sorts ``coords`` lexicographically by (axis 0, axis 1).

    Parameters
    ----------
    coords : torch.Tensor
        Shape ``[N, 2]``. Treated as (y, x) since LH-NeF's coord convention is (y, x).
    """
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError(f"coords must be [N,2], got {tuple(coords.shape)}")
    arr = coords.detach().cpu().numpy()
    # np.lexsort: last key is primary; want y primary -> keys=(x, y)
    perm = np.lexsort((arr[:, 1], arr[:, 0]))
    return torch.from_numpy(perm.astype(np.int64))


def _infer_grid_h(coords: torch.Tensor, total: int, *, decimals: int = 4) -> int:
    """Infer Hg from a 2D coordinate set by counting unique y values up to ``decimals``."""
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError(f"coords must be [N,2], got {tuple(coords.shape)}")
    y = coords[:, 0].detach().cpu().numpy()
    uniq = np.unique(np.round(y, decimals=decimals))
    h = int(uniq.size)
    if h <= 0 or total % h != 0:
        raise RuntimeError(
            f"Could not infer grid height from coordinates: total={total}, unique_y={h}. "
            f"Pass `group_grid` / `slot_grid` explicitly in the config."
        )
    return h


class _LayerNorm2d(nn.Module):
    """LayerNorm applied over the channel dimension of a [B,C,H,W] tensor."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(int(dim), eps=float(eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        return rearrange(x, "b h w c -> b c h w")


class _DropPath(nn.Module):
    """Stochastic depth (per-sample drop)."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


class _ConvNeXtBlock(nn.Module):
    """Standard ConvNeXt-v1 block (depthwise + LN + 2-layer MLP, with LayerScale)."""

    def __init__(
        self,
        dim: int,
        *,
        kernel_size: int = 7,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        pad = int(kernel_size) // 2
        self.dwconv = nn.Conv2d(int(dim), int(dim), kernel_size=int(kernel_size),
                                padding=pad, groups=int(dim))
        self.norm = nn.LayerNorm(int(dim), eps=1e-6)
        hidden = int(round(float(mlp_ratio) * int(dim)))
        self.pwconv1 = nn.Linear(int(dim), hidden)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden, int(dim))
        self.gamma: Optional[nn.Parameter]
        if layer_scale_init > 0:
            self.gamma = nn.Parameter(float(layer_scale_init) * torch.ones(int(dim)))
        else:
            self.gamma = None
        self.drop_path = _DropPath(float(drop_path)) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = rearrange(x, "b c h w -> b h w c")
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = rearrange(x, "b h w c -> b c h w")
        return residual + self.drop_path(x)


@register("convnext_grouped_classifier")
class ConvNeXtGroupedClassifier(nn.Module):
    """ConvNeXt over the spatialised grouped token representation.

    Parameters
    ----------
    token_dim : int
        Channel dimension ``C`` of the grouped tokens.
    num_classes : int
        Number of output classes.
    num_groups, tokens_per_group : int
        ``G`` and ``K`` of the latent layout ``[B, G, K, C]``.
    coord_dim : int
        Currently only ``2`` is supported.
    group_grid : tuple[int, int] or None
        ``(Hg, Wg)`` such that ``Hg * Wg == G``. If ``None``, inferred from
        group centers on the first forward pass.
    slot_grid : tuple[int, int] or None
        ``(Hk, Wk)`` such that ``Hk * Wk == K``. If ``None``, inferred from
        ``p_token`` (group 0) on the first forward pass; falls back to
        ``(1, K)`` when ``p_token`` is missing.
    dims, depths : sequence[int]
        Per-stage channel widths and number of ``_ConvNeXtBlock`` per stage.
    kernel_size : int
        Depthwise conv kernel size. Default 5 (small spatial maps).
    mlp_ratio : float
        MLP expansion in each block (default 4).
    drop_path : float
        Maximum stochastic-depth rate (linearly scheduled across blocks).
    layer_scale_init : float
        LayerScale initial value (set to 0 to disable).
    head_dropout : float
        Dropout before the classifier head.
    label_smoothing : float
        CE label smoothing.
    downsample_kernel : int
        Stride/kernel of inter-stage strided conv. Default 2.
    mixup_alpha, cutmix_alpha, token_drop_p : float
        Latent-space augmentations; each disabled at 0.
    mix_prob : float
        Probability of firing Mixup/CutMix per batch. Default 1.0.
    switch_prob : float
        When both Mixup and CutMix are enabled, probability of CutMix. Default 0.5.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        num_classes: int,
        num_groups: int,
        tokens_per_group: int,
        coord_dim: int = 2,
        group_grid: Optional[Sequence[int]] = None,
        slot_grid: Optional[Sequence[int]] = None,
        dims: Sequence[int] = (128, 256),
        depths: Sequence[int] = (3, 3),
        kernel_size: int = 5,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
        layer_scale_init: float = 1e-6,
        head_dropout: float = 0.0,
        label_smoothing: float = 0.0,
        downsample_kernel: int = 2,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        token_drop_p: float = 0.0,
        mix_prob: float = 1.0,
        switch_prob: float = 0.5,
    ):
        super().__init__()
        if int(coord_dim) != 2:
            raise NotImplementedError("ConvNeXtGroupedClassifier only supports coord_dim=2.")
        self.coord_dim = int(coord_dim)
        self.num_classes = int(num_classes)
        self.G = int(num_groups)
        self.K = int(tokens_per_group)
        self.C = int(token_dim)
        self.label_smoothing = float(label_smoothing)

        depths = tuple(int(d) for d in depths)
        dims = tuple(int(d) for d in dims)
        if len(depths) != len(dims):
            raise ValueError(f"len(depths)={len(depths)} must equal len(dims)={len(dims)}")
        if len(depths) < 1:
            raise ValueError("Need at least one ConvNeXt stage.")
        self.depths = depths
        self.dims = dims

        self._given_group_grid = (
            None if group_grid is None else (int(group_grid[0]), int(group_grid[1]))
        )
        self._given_slot_grid = (
            None if slot_grid is None else (int(slot_grid[0]), int(slot_grid[1]))
        )
        if self._given_group_grid is not None:
            Hg, Wg = self._given_group_grid
            if Hg * Wg != self.G:
                raise ValueError(f"group_grid {self._given_group_grid} does not multiply to G={self.G}.")
        if self._given_slot_grid is not None:
            Hk, Wk = self._given_slot_grid
            if Hk * Wk != self.K:
                raise ValueError(f"slot_grid {self._given_slot_grid} does not multiply to K={self.K}.")

        # Permutations as buffers (move with .to(device) and DDP); filled lazily on first forward.
        self.register_buffer("_group_perm", torch.arange(self.G, dtype=torch.long), persistent=False)
        self.register_buffer("_slot_perm", torch.arange(self.K, dtype=torch.long), persistent=False)
        self._layout_resolved = False
        self._group_grid: Optional[Tuple[int, int]] = self._given_group_grid
        self._slot_grid: Optional[Tuple[int, int]] = self._given_slot_grid

        self.stem = nn.Conv2d(self.C, dims[0], kernel_size=1)
        self.stem_norm = _LayerNorm2d(dims[0], eps=1e-6)

        # Linearly scheduled stochastic depth.
        total_blocks = sum(depths)
        if total_blocks > 1:
            dp_rates: List[float] = [
                float(drop_path) * float(i) / float(total_blocks - 1) for i in range(total_blocks)
            ]
        else:
            dp_rates = [float(drop_path)] if total_blocks == 1 else []

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        bi = 0
        for si, (d, dim_s) in enumerate(zip(depths, dims)):
            blocks = []
            for _ in range(int(d)):
                blocks.append(_ConvNeXtBlock(
                    dim=dim_s,
                    kernel_size=int(kernel_size),
                    mlp_ratio=float(mlp_ratio),
                    drop_path=dp_rates[bi],
                    layer_scale_init=float(layer_scale_init),
                ))
                bi += 1
            self.stages.append(nn.Sequential(*blocks))
            if si < len(depths) - 1:
                self.downsamples.append(nn.Sequential(
                    _LayerNorm2d(dim_s, eps=1e-6),
                    nn.Conv2d(dim_s, dims[si + 1],
                              kernel_size=int(downsample_kernel),
                              stride=int(downsample_kernel)),
                ))

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head_drop = nn.Dropout(float(head_dropout)) if head_dropout > 0 else nn.Identity()
        self.head = nn.Linear(dims[-1], self.num_classes)

        self.aug = LatentAugMixer(
            mixup_alpha=float(mixup_alpha),
            cutmix_alpha=float(cutmix_alpha),
            token_drop_p=float(token_drop_p),
            mix_prob=float(mix_prob),
            switch_prob=float(switch_prob),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _resolve_layout(
        self,
        p_group: torch.Tensor,
        p_token: Optional[torch.Tensor],
    ) -> None:
        """Compute group/slot row-major permutations from positions in the batch."""
        # Positions are constant across the dataset; accept [B,G,d] or [G,d].
        if p_group.ndim == 3:
            p_group = p_group[0]
        if p_group.ndim != 2 or int(p_group.shape[0]) != self.G or int(p_group.shape[-1]) != 2:
            raise ValueError(f"Expected p [G={self.G},2], got {tuple(p_group.shape)}")
        p_group_cpu = p_group.detach().cpu()

        if self._given_group_grid is None:
            Hg = _infer_grid_h(p_group_cpu, total=self.G)
            Wg = self.G // Hg
            self._group_grid = (Hg, Wg)

        group_perm = _lex_sort_perm_2d(p_group_cpu)

        if self.K == 1:
            self._slot_grid = (1, 1)
            slot_perm = torch.zeros(1, dtype=torch.long)
        else:
            if p_token is None:
                if self._given_slot_grid is None:
                    self._slot_grid = (1, self.K)
                slot_perm = torch.arange(self.K, dtype=torch.long)
            else:
                if p_token.ndim == 3:
                    p_token = p_token[0]
                expected_L = self.G * self.K
                if p_token.ndim != 2 or int(p_token.shape[0]) != expected_L:
                    raise ValueError(
                        f"Expected p_token [L={expected_L},2], got {tuple(p_token.shape)}"
                    )
                slots0 = p_token.detach().cpu()[: self.K]
                if self._given_slot_grid is None:
                    Hk = _infer_grid_h(slots0, total=self.K)
                    Wk = self.K // Hk
                    self._slot_grid = (Hk, Wk)
                slot_perm = _lex_sort_perm_2d(slots0)

        self._group_perm = group_perm.to(self._group_perm.device)
        self._slot_perm = slot_perm.to(self._slot_perm.device)
        self._layout_resolved = True

    def _spatialize(self, c: torch.Tensor) -> torch.Tensor:
        """Reshape [B, G, K, C] into a 2D feature map [B, C, Hg*Hk, Wg*Wk]."""
        if self._group_grid is None or self._slot_grid is None:
            raise RuntimeError("Layout not resolved before _spatialize().")
        Hg, Wg = self._group_grid
        Hk, Wk = self._slot_grid
        c = c.index_select(dim=1, index=self._group_perm)
        c = c.index_select(dim=2, index=self._slot_perm)
        return rearrange(
            c,
            "b (hg wg) (hk wk) c -> b c (hg hk) (wg wk)",
            hg=Hg, wg=Wg, hk=Hk, wk=Wk,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        c = batch.get("c", None)
        if c is None:
            raise KeyError("batch missing 'c'")
        if c.ndim == 3:
            B, L, Cc = c.shape
            if int(L) != int(self.G * self.K):
                raise ValueError(f"Expected L=G*K={self.G * self.K}, got L={L}")
            c = c.view(B, self.G, self.K, Cc)
        elif c.ndim != 4:
            raise ValueError(f"c must be [B,L,C] or [B,G,K,C], got {tuple(c.shape)}")
        if int(c.shape[-1]) != self.C:
            raise ValueError(f"token_dim mismatch: expected {self.C}, got {int(c.shape[-1])}")
        c = c.float()

        if not self._layout_resolved:
            p_group = batch.get("p", None)
            p_token = batch.get("p_token", None)
            if p_group is None:
                raise KeyError("batch missing 'p' (group centers); required to infer 2D layout.")
            self._resolve_layout(p_group, p_token)

        x = self._spatialize(c)         # [B, C, H, W]

        y = batch.get("y", None)
        if y is not None and y.ndim != 1:
            y = y.view(-1)
        y_a, y_b, lam = y, y, 1.0
        if y is not None and self.aug.any_active and self.training:
            x, y_a, y_b, lam = self.aug(x, y)

        x = self.stem(x)  # [B, dims[0], H, W]
        x = self.stem_norm(x)

        for si, stage in enumerate(self.stages):
            x = stage(x)
            if si < len(self.stages) - 1:
                x = self.downsamples[si](x)

        x = x.mean(dim=[-2, -1])        # GAP -> [B, dims[-1]]
        x = self.norm(x)
        x = self.head_drop(x)
        logits = self.head(x)

        if y is None:
            return {"logits": logits}
        if lam < 1.0:
            loss = (
                lam * F.cross_entropy(logits, y_a, label_smoothing=self.label_smoothing)
                + (1.0 - lam) * F.cross_entropy(logits, y_b, label_smoothing=self.label_smoothing)
            )
        else:
            loss = F.cross_entropy(logits, y_a, label_smoothing=self.label_smoothing)
        # Accuracy against original (un-permuted) labels (timm convention).
        acc = (logits.argmax(dim=-1) == y).float().mean()
        return {"loss": loss, "acc": acc}
