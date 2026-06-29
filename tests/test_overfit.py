"""Tier 2 -- tiny-batch overfit test.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import models  # noqa: F401
from run import parse_cfg_file
from utils.utils import make_optimizer


REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "src/cfgs/ae-hip-cifar10-generic-nef.yaml"


def _make_smooth_batch(bs: int = 4, H: int = 32, W: int = 32,
                       device: torch.device = torch.device("cpu"),
                       seed: int = 0) -> dict:
    """Smooth low-frequency RGB fields. Easy to overfit; unlike pure noise, the
    test signals the model can fit *any* coherent signal, not just memorize.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    imgs = []
    for _ in range(bs):
        ph = torch.rand(3, 4, generator=g) * (2 * math.pi)
        fx = torch.rand(3, 4, generator=g) * 2 + 0.5
        fy = torch.rand(3, 4, generator=g) * 2 + 0.5
        img = torch.zeros(3, H, W)
        for c in range(3):
            for k in range(4):
                img[c] += torch.sin(fx[c, k] * x * math.pi + ph[c, k]) \
                          * torch.cos(fy[c, k] * y * math.pi)
        img = img / img.abs().max().clamp_min(1e-6)
        imgs.append(img)
    img = torch.stack(imgs).to(device)  # [B, 3, H, W] in [-1, 1]
    coord = torch.stack([y, x], dim=-1).reshape(-1, 2).to(device)
    coord = coord.unsqueeze(0).expand(bs, -1, -1).contiguous()
    value = img.permute(0, 2, 3, 1).reshape(bs, -1, 3).contiguous()
    gt_cell = torch.full_like(coord, 2.0 / H)
    return {"inp": {"coord": coord, "value": value},
            "gt_coord": coord, "gt_cell": gt_cell, "gt": value}


def _psnr_pm1(mse: float) -> float:
    # Targets in [-1, 1] => peak-to-peak = 2 => PSNR = 10*log10(4/mse).
    return 10.0 * math.log10(4.0 / max(mse, 1e-12))


@pytest.mark.gpu
def test_overfit_cifar_cfg(device: torch.device):
    cfg = parse_cfg_file(str(CFG_PATH))
    torch.manual_seed(0)

    model = models.make(cfg.model).to(device)
    model.train()

    optims = {}
    for name, spec in cfg.optimizers.items():
        if spec is None:
            continue
        params = list(model.get_params(name))
        optims[name] = make_optimizer(params, OmegaConf.to_container(spec, resolve=True))

    data = _make_smooth_batch(bs=4, device=device, seed=0)
    has_opt = {k: True for k in optims}

    n_iters = 300
    last_psnr = 0.0
    for i in range(n_iters):
        for o in optims.values():
            o.zero_grad(set_to_none=True)
        ret = model(data, mode="loss", has_opt=has_opt)
        loss = ret["loss"]
        loss.backward()
        for o in optims.values():
            o.step()

        if (i + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(data, mode="pred")
                mse = float((pred - data["gt"]).pow(2).mean().item())
            last_psnr = _psnr_pm1(mse)
            model.train()

    assert last_psnr > 25.0, (
        f"overfit failed: PSNR={last_psnr:.2f} dB after {n_iters} iters "
        f"(expected > 25 dB). Encoder/renderer/loss/optimizer wiring may be broken."
    )
