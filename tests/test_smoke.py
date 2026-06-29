"""Tier 1 -- static/shape smoke tests.

For each stage entry point:
  - resolves the YAML config (incl. _base_ chain),
  - checks the trainer name is in the registry,
  - builds the model,
  - runs a forward pass on synthetic data and asserts a finite loss.

CPU-only; under a minute. Catches the most common breakages (renamed cfg keys,
dead imports, registry typos, shape mismatches between encoder and renderer).
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import models  # noqa: F401
import datasets  # noqa: F401
import trainers  # noqa: F401
import diffusion.models  # noqa: F401
import classification.models  # noqa: F401
from run import parse_cfg_file
from trainers import trainers_dict


REPO_ROOT = Path(__file__).resolve().parents[1]

STAGE1_CFGS = [
    "src/cfgs/ae-hip-cifar10-generic-nef.yaml",
]


def _synthetic_image_batch(bs: int = 2, H: int = 32, W: int = 32) -> dict:
    img = torch.rand(bs, 3, H, W) * 2 - 1
    coord = torch.stack(torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
    ), dim=-1).reshape(-1, 2)
    coord = coord.unsqueeze(0).expand(bs, -1, -1).contiguous()  # [B, H*W, 2]
    value = img.permute(0, 2, 3, 1).reshape(bs, -1, 3).contiguous()  # [B, H*W, 3]
    gt_cell = torch.full_like(coord, 2.0 / H)
    return {"inp": {"coord": coord, "value": value},
            "gt_coord": coord, "gt_cell": gt_cell, "gt": value}


@pytest.mark.parametrize("cfg_path", STAGE1_CFGS)
def test_stage1_forward(cfg_path: str):
    cfg = parse_cfg_file(str(REPO_ROOT / cfg_path))
    assert cfg.trainer in trainers_dict, f"trainer {cfg.trainer!r} not registered"

    model = models.make(cfg.model)
    for name, spec in cfg.optimizers.items():
        if spec is None:
            continue
        params = list(model.get_params(name))
        assert len(params) > 0, f"get_params({name!r}) returned empty list"

    data = _synthetic_image_batch()
    model.eval()
    with torch.no_grad():
        ret_enc = model(data, mode="encode")
        assert "z_pre" in ret_enc
        ret = model(data, mode="loss",
                    has_opt={"encoder": True, "decoder": True, "renderer": True})
    assert "loss" in ret
    loss = ret["loss"]
    loss_val = float(loss.item()) if torch.is_tensor(loss) else float(loss)
    assert math.isfinite(loss_val), f"loss not finite: {loss_val}"


def test_stage2a_diffusion_model_builds():
    cfg_path = REPO_ROOT / "src/cfgs/diffusion/edm/cifar10.yaml"
    cfg = parse_cfg_file(str(cfg_path))
    assert cfg.trainer in trainers_dict

    spec = OmegaConf.to_container(cfg.model, resolve=True)
    args = spec["args"]
    # Stage-2 cfgs reference manifest-derived fields with the string "from_manifest";
    # patch them with small mock values so we can build the model without extraction.
    mocks = {"num_groups": 64, "tokens_per_group": 16, "channels_per_token": 16,
             "token_dim": 16, "num_tokens": 1024}
    for k, v in mocks.items():
        if args.get(k) == "from_manifest":
            args[k] = v
    inst = models.make({"name": spec["name"], "args": args})
    assert inst is not None


def test_stage2b_classifier_model_builds():
    cfg_path = REPO_ROOT / "src/cfgs/classification/classify_cifar10.yaml"
    cfg = parse_cfg_file(str(cfg_path))
    assert cfg.trainer in trainers_dict

    spec = OmegaConf.to_container(cfg.model, resolve=True)
    args = spec["args"]
    gg = args.get("group_grid")
    n_groups = int(gg[0] * gg[1]) if isinstance(gg, (list, tuple)) and len(gg) == 2 else 64
    sg = args.get("slot_grid")
    n_slots = int(sg[0] * sg[1]) if isinstance(sg, (list, tuple)) and len(sg) == 2 else 16
    mocks = {"num_classes": 10, "num_groups": n_groups, "tokens_per_group": n_slots,
             "channels_per_token": 16, "token_dim": 16, "num_tokens": n_groups * n_slots}
    for k, v in mocks.items():
        if args.get(k) == "from_manifest":
            args[k] = v
    inst = models.make({"name": spec["name"], "args": args})
    assert inst is not None
