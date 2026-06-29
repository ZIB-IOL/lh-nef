"""EDM preconditioning (Karras et al. 2022, arXiv:2206.00364).

Continuous-sigma framework for diffusion training and sampling. Network predicts a raw
`F_theta(c_in(sigma) * x_noisy, c_noise(sigma))`, and
    x0_hat = c_skip(sigma) * x_noisy + c_out(sigma) * F_theta(...)

Loss is `lambda(sigma) * ||x0_hat - x0||^2`, with sigma sampled lognormally during training:
    ln(sigma) ~ N(P_mean, P_std^2)

Sampling uses Heun's 2nd-order ODE on a Karras sigma-schedule. Default σ_min=0.002, σ_max=80, ρ=7.

Reference impl: https://github.com/NVlabs/edm (training/networks.py:EDMPrecond, training/loss.py:EDMLoss,
generate.py:edm_sampler).

NOTE: σ_data MUST match the dataset statistics. For per-channel-normalized HiP latents with
norm_scale=1.0, σ_data ≈ 1.0 (not 0.5). Measure with `measure_sigma_data` before training.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["EDMSchedule", "TokenEDM"]


@dataclass
class EDMSchedule:
    """EDM continuous-sigma schedule + preconditioning coefficients."""
    sigma_data: float = 1.0
    P_mean: float = -1.2
    P_std: float = 1.2
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0

    def c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()

    def c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / (self.sigma_data ** 2 + sigma ** 2).sqrt()

    def c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        # Per EDM: 0.25 * log(sigma). Tiny conditioning scalar; the model's time-embedder
        # must be calibrated for this range (~[-1.5, 1.1] for σ in [0.002, 80]).
        return 0.25 * sigma.log()

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def sample_training_sigma(self, n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Returns [n,1,1] sigma sampled lognormally."""
        eps = torch.randn(n, 1, 1, device=device, dtype=dtype)
        return (eps * self.P_std + self.P_mean).exp()

    def karras_sigmas(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Returns [num_steps + 1] sigma grid descending from sigma_max to 0 (last is exactly 0)."""
        n = int(num_steps)
        i = torch.arange(n, device=device, dtype=torch.float64)
        sm, sM, rho = float(self.sigma_min), float(self.sigma_max), float(self.rho)
        sig = (sM ** (1.0 / rho) + i / max(1, n - 1) * (sm ** (1.0 / rho) - sM ** (1.0 / rho))) ** rho
        # EDM appends a zero terminal sigma for the last (clean) step.
        return torch.cat([sig, torch.zeros(1, device=device, dtype=sig.dtype)]).to(torch.float32)


def measure_sigma_data(latents: torch.Tensor) -> float:
    """Compute σ_data as the global std of normalized latents. Pass already-normalized latents."""
    return float(latents.float().std().item())


def measure_sigma_data_from_manifest(
    manifest_path: str,
    split: str = "train",
    max_shards: int = 2,
    norm_scale: float = 1.0,
) -> float:
    """Compute σ_data from a token-latents manifest. Loads up to `max_shards` shards,
    applies per-channel `(c - mean) / (std * norm_scale)` matching
    `latent_dataset.HipTokenLatents`, returns the global std.
    """
    import json
    with open(manifest_path, "r") as fh:
        manifest = json.load(fh)
    sp = manifest.get("splits", {}).get(split, None)
    if sp is None:
        raise KeyError(f"split={split!r} not found in manifest {manifest_path!r}")
    shards = list(sp.get("shards") or [])
    if not shards:
        raise ValueError(f"manifest split {split!r} has no shards")
    mean = torch.tensor(sp.get("mean", []), dtype=torch.float32)
    std = torch.tensor(sp.get("std", []), dtype=torch.float32).clamp_min(1e-6) * float(norm_scale)
    vals = []
    for sh in shards[: int(max_shards)]:
        payload = torch.load(str(sh), map_location="cpu", weights_only=False)
        c = payload["c"].to(dtype=torch.float32)  # [N,L,C]
        # Broadcast mean/std layouts: [C] or [L,C].
        if mean.ndim == 1:
            c = (c - mean.view(1, 1, -1)) / std.view(1, 1, -1)
        elif mean.ndim == 2:
            c = (c - mean.view(1, *mean.shape)) / std.view(1, *std.shape)
        else:
            raise ValueError(f"unsupported mean ndim {mean.ndim}")
        vals.append(c.flatten())
    return float(torch.cat(vals).std().item())


class TokenEDM:
    """EDM preconditioning wrapper around an arbitrary token denoiser.

    The model is called as `model(c_noisy=c_in*x_noisy, p=p, t=c_noise)` and is expected to
    return a raw F_theta of shape [B, L, C]. The trainer must configure the model's time
    embedder to handle continuous c_noise input (small range, ~[-1.5, 1.1]); for HiPDiT,
    set `time_input='edm_cnoise'` so it uses FourierTimeEmbed instead of the discrete-t
    sinusoidal TimestepEmbedder.
    """

    def __init__(self, schedule: EDMSchedule):
        self.sched = schedule

    def precond_forward(
        self,
        model: nn.Module,
        *,
        x_noisy: torch.Tensor,
        sigma: torch.Tensor,
        p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_noisy: [B,L,C]
        sigma:   [B,1,1]
        Returns (x0_hat, F_raw).
        """
        c_in = self.sched.c_in(sigma)
        c_out = self.sched.c_out(sigma)
        c_skip = self.sched.c_skip(sigma)
        # c_noise: scalar per batch element, shape [B] for the time embedder.
        t_in = self.sched.c_noise(sigma).view(-1)
        F = model(c_noisy=(c_in * x_noisy), p=p, t=t_in)
        x0_hat = c_skip * x_noisy + c_out * F
        return x0_hat, F

    def training_loss(
        self,
        model: nn.Module,
        *,
        x0: torch.Tensor,
        p: torch.Tensor,
        sigma: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """EDM training loss: weighted MSE between denoised prediction and clean x0.

        Returns (loss, x0_hat).
        """
        B = int(x0.shape[0])
        if sigma is None:
            sigma = self.sched.sample_training_sigma(B, x0.device, dtype=x0.dtype)
        if noise is None:
            noise = torch.randn_like(x0)
        x_noisy = x0 + noise * sigma
        x0_hat, _F = self.precond_forward(model, x_noisy=x_noisy, sigma=sigma, p=p)
        w = self.sched.loss_weight(sigma)
        # Mean over (L, C) per sample then weighted mean over batch — match TokenDDPM behavior.
        per_sample_mse = (x0_hat - x0).pow(2).mean(dim=(1, 2))  # [B]
        w_b = w.view(B)
        loss = (per_sample_mse * w_b).mean()
        return loss, x0_hat

    @torch.no_grad()
    def sample_heun(
        self,
        model: nn.Module,
        *,
        p: torch.Tensor,
        shape: Tuple[int, int, int],
        num_steps: int = 18,
        device: Optional[torch.device] = None,
        S_churn: float = 0.0,
        S_min: float = 0.0,
        S_max: float = float("inf"),
        S_noise: float = 1.0,
    ) -> torch.Tensor:
        """
        EDM Heun 2nd-order deterministic-or-stochastic ODE sampler (Karras et al. 2022, Alg. 2).
        Default S_churn=0 is fully deterministic (recommended for CIFAR-10).
        """
        B, L, C = map(int, shape)
        device = device or (p.device if torch.is_tensor(p) else torch.device("cpu"))
        sigmas = self.sched.karras_sigmas(int(num_steps), device=device).to(torch.float32)
        # Initialize at sigma_max.
        x = torch.randn((B, L, C), device=device, dtype=torch.float32) * sigmas[0]

        for i in range(int(num_steps)):
            s = sigmas[i]
            s_next = sigmas[i + 1]

            # Optional stochasticity (S_churn > 0).
            if S_churn > 0 and S_min <= float(s.item()) <= S_max:
                gamma = min(float(S_churn) / float(num_steps), math.sqrt(2.0) - 1.0)
            else:
                gamma = 0.0
            s_hat = s * (1.0 + gamma)
            if gamma > 0:
                noise = torch.randn_like(x) * S_noise
                x = x + (s_hat ** 2 - s ** 2).clamp_min(0.0).sqrt() * noise

            # 1st-order Euler step (denoise at s_hat).
            sig_b = s_hat.view(1, 1, 1).expand(B, 1, 1).to(x.dtype)
            x0, _ = self.precond_forward(model, x_noisy=x, sigma=sig_b, p=p)
            d = (x - x0) / s_hat
            x_next = x + (s_next - s_hat) * d

            # 2nd-order correction (Heun) when not at the final clean step.
            if float(s_next.item()) > 0:
                sig_b2 = s_next.view(1, 1, 1).expand(B, 1, 1).to(x.dtype)
                x0_2, _ = self.precond_forward(model, x_noisy=x_next, sigma=sig_b2, p=p)
                d2 = (x_next - x0_2) / s_next
                x_next = x + (s_next - s_hat) * 0.5 * (d + d2)
            x = x_next

        return x
