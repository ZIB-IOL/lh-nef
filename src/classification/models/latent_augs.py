"""Latent-space augmentations (Mixup / CutMix / TokenDrop) on a spatialised feature map.

Each knob (``mixup_alpha``, ``cutmix_alpha``, ``token_drop_p``) is disabled at 0.
Returns ``(x', y_a, y_b, lam)``; consumer computes
``loss = lam * CE(logits, y_a) + (1 - lam) * CE(logits, y_b)``.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["LatentAugMixer"]


class LatentAugMixer(nn.Module):
    """Apply Mixup / CutMix / TokenDrop on a spatialised latent feature map."""

    def __init__(
        self,
        *,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        token_drop_p: float = 0.0,
        mix_prob: float = 1.0,
        switch_prob: float = 0.5,
    ):
        super().__init__()
        if float(mixup_alpha) < 0:
            raise ValueError(f"mixup_alpha must be >= 0, got {mixup_alpha}")
        if float(cutmix_alpha) < 0:
            raise ValueError(f"cutmix_alpha must be >= 0, got {cutmix_alpha}")
        if not (0.0 <= float(token_drop_p) < 1.0):
            raise ValueError(f"token_drop_p must be in [0, 1), got {token_drop_p}")
        if not (0.0 <= float(mix_prob) <= 1.0):
            raise ValueError(f"mix_prob must be in [0, 1], got {mix_prob}")
        if not (0.0 <= float(switch_prob) <= 1.0):
            raise ValueError(f"switch_prob must be in [0, 1], got {switch_prob}")
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.token_drop_p = float(token_drop_p)
        self.mix_prob = float(mix_prob)
        self.switch_prob = float(switch_prob)

    @property
    def any_active(self) -> bool:
        return (
            self.mixup_alpha > 0.0
            or self.cutmix_alpha > 0.0
            or self.token_drop_p > 0.0
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply the mixer.

        Parameters
        ----------
        x : torch.Tensor
            Spatialised feature map ``[B, C, H, W]``.
        y : torch.Tensor
            Integer class labels ``[B]``.

        Returns
        -------
        (x', y_a, y_b, lam)
            ``lam`` is the mixup weight (1.0 when no label-mixing aug fired).
        """
        if not self.training or y is None:
            return x, y, y, 1.0

        y_a, y_b, lam = y, y, 1.0

        do_mix = (
            (self.mixup_alpha > 0.0 or self.cutmix_alpha > 0.0)
            and (torch.rand((), device=x.device).item() < self.mix_prob)
        )
        if do_mix:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = torch.rand((), device=x.device).item() < self.switch_prob
            else:
                use_cutmix = self.cutmix_alpha > 0.0

            perm = torch.randperm(x.size(0), device=x.device)
            x_b = x[perm]
            y_b = y[perm]

            if use_cutmix:
                lam_init = self._sample_beta(self.cutmix_alpha)
                x, lam = self._cutmix(x, x_b, lam_init)
            else:
                lam = self._sample_beta(self.mixup_alpha)
                x = lam * x + (1.0 - lam) * x_b

        # Token drop: zero random spatial cells; no inverted-dropout rescaling
        # (MAE/BEiT-style token-masking convention).
        if self.token_drop_p > 0.0:
            B, _, H, W = x.shape
            keep = 1.0 - self.token_drop_p
            mask = torch.empty(B, 1, H, W, device=x.device, dtype=x.dtype).bernoulli_(keep)
            x = x * mask

        return x, y_a, y_b, float(lam)

    @staticmethod
    def _sample_beta(alpha: float) -> float:
        a = float(alpha)
        if a <= 0.0:
            return 1.0
        beta = torch.distributions.Beta(torch.tensor(a), torch.tensor(a))
        return float(beta.sample().item())

    @staticmethod
    def _cutmix(
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        lam: float,
    ) -> Tuple[torch.Tensor, float]:
        """Paste a (1-lam)-area rectangle from ``x_b`` into ``x_a``.

        ``lam`` is recomputed from the realised patch area (timm CutMix convention).
        """
        B, C, H, W = x_a.shape
        cut_ratio = math.sqrt(max(0.0, 1.0 - float(lam)))
        cut_h = int(round(float(H) * cut_ratio))
        cut_w = int(round(float(W) * cut_ratio))
        if cut_h <= 0 or cut_w <= 0:
            return x_a, 1.0
        cy = int(torch.randint(0, H, (1,)).item())
        cx = int(torch.randint(0, W, (1,)).item())
        y0 = max(0, cy - cut_h // 2)
        y1 = min(H, cy + (cut_h - cut_h // 2))
        x0 = max(0, cx - cut_w // 2)
        x1 = min(W, cx + (cut_w - cut_w // 2))
        if y1 <= y0 or x1 <= x0:
            return x_a, 1.0
        out = x_a.clone()
        out[:, :, y0:y1, x0:x1] = x_b[:, :, y0:y1, x0:x1]
        true_lam = 1.0 - (float(y1 - y0) * float(x1 - x0)) / float(H * W)
        return out, float(true_lam)
