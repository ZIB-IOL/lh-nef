"""Parse args, build cfg, and spawn the configured trainer."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch.distributed as dist
import yaml
from omegaconf import OmegaConf

# Mirror src/eval_ckpt.py: support both `import src` and legacy absolute imports
# like `import datasets`, `import models`, `import trainers` regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parents[0]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import ensure_path  # noqa: F401
from trainers import trainers_dict


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='cfgs/_.yaml')
    # --opt accepts either repeated `--opt k v` or a single `--opt k1 v1 k2 v2 ...`.
    parser.add_argument('--opt', nargs='*', action='append', default=[])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--resume-mode', '-r', default='replace')

    # None -> fall back to cfg.save_root / cfg._env.save_root / LHNEF_SAVE_ROOT (avoids writing into CWD).
    parser.add_argument('--save-root', default=None)
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--wandb', '-w', action='store_true')
    # Also accept W&B-sweep-style overrides: `--key=value`, `--key value`, or bare `key=value`.
    args, unknown = parser.parse_known_args()
    extra_opt: list[str] = []

    i = 0
    while i < len(unknown):
        tok = str(unknown[i])
        if tok.startswith('--'):
            key = tok[2:]
            if '=' in key:
                extra_opt.append(key)
                i += 1
                continue
            if i + 1 >= len(unknown) or str(unknown[i + 1]).startswith('--'):
                raise ValueError(f"Flag {tok} is missing a value")
            val = str(unknown[i + 1])
            extra_opt.append(f"{key}={val}")
            i += 2
            continue
        if '=' in tok:
            extra_opt.append(tok)
        i += 1

    if extra_opt:
        args.opt.append(extra_opt)  # args.opt is list-of-lists (action='append')
    return args


def parse_cfg(cfg):
    if cfg.get('_base_') is not None:
        fnames = cfg.pop('_base_')
        if isinstance(fnames, str):
            fnames = [fnames]
        base_cfg = OmegaConf.merge(*[parse_cfg(OmegaConf.load(_)) for _ in fnames])
        cfg = OmegaConf.merge(base_cfg, cfg)
    return cfg


def parse_cfg_file(cfg_path: str):
    """Load a YAML config, resolving `_base_` entries relative to the file's directory."""
    cfg = OmegaConf.load(cfg_path)
    base_dir = os.path.dirname(cfg_path)

    def _merge_from(cfg_obj, cur_dir):
        if cfg_obj.get('_base_') is not None:
            fnames = cfg_obj.pop('_base_')
            if isinstance(fnames, str):
                fnames = [fnames]
            base_cfgs = []
            for fname in fnames:
                if os.path.isabs(fname):
                    candidates = [fname]
                else:
                    # Try, in order: this file's dir, its parent, SRC_ROOT, REPO_ROOT.
                    candidates = [
                        os.path.join(cur_dir, fname),
                        os.path.join(os.path.dirname(cur_dir), fname),  # fallback: relative to parent (e.g., src)
                        os.path.join(str(SRC_ROOT), fname),
                        os.path.join(str(REPO_ROOT), fname),
                    ]
                base_path = None
                for cand in candidates:
                    if os.path.exists(cand):
                        base_path = cand
                        break
                if base_path is None:
                    raise FileNotFoundError(f"Could not resolve base config '{fname}' relative to '{cur_dir}' or its parent.")
                base_cfgs.append(parse_cfg_file(base_path))
            return OmegaConf.merge(*base_cfgs, cfg_obj)
        return cfg_obj

    return _merge_from(cfg, base_dir)


def make_cfg(args):
    cfg = parse_cfg_file(args.cfg)
    # Flatten possibly repeated --opt lists; supports both `--opt k v` and `--opt k=v`.
    raw_opt: list[str] = []
    if isinstance(args.opt, list):
        for chunk in args.opt:
            if chunk is None:
                continue
            if isinstance(chunk, (list, tuple)):
                raw_opt.extend([str(x) for x in chunk])
            else:
                raw_opt.append(str(chunk))
    else:
        raw_opt = [str(x) for x in list(args.opt)]

    opt: list[str] = []
    for tok in raw_opt:
        if isinstance(tok, str) and ("=" in tok):
            k, v = tok.split("=", 1)
            if k.strip() == "":
                opt.append(tok)
            else:
                opt.extend([k, v])
        else:
            opt.append(tok)

    if len(opt) % 2 != 0:
        raise ValueError(f"--opt must contain key/value pairs, got odd number of tokens: {len(opt)} ({opt})")
    for i in range(0, len(opt), 2):
        k, v = opt[i: i + 2]
        # Cast values so W&B sweep args like `--key=[0.9, 0.999]` become real lists, not strings.
        vv = v
        if isinstance(v, str):
            s = v.strip()
            try:
                vv = yaml.safe_load(s)
            except Exception:
                vv = v
        OmegaConf.update(cfg, k, vv)
    cfg.random_seed = args.seed

    env = OmegaConf.create()
    if args.name is None:
        exp_name = os.path.splitext(os.path.basename(args.cfg))[0]
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag
    env.exp_name = exp_name
    # Save-root resolution priority: CLI --save-root, then env var, then cfg.save_root / cfg._env.save_root.
    # INFD_SAVE_ROOT is kept as a back-compat fallback for the renamed LHNEF_SAVE_ROOT.
    save_root = args.save_root
    if save_root is None:
        save_root = os.environ.get("LHNEF_SAVE_ROOT", None) or os.environ.get("INFD_SAVE_ROOT", None) or "save"
    try:
        if str(save_root) == "save":
            if cfg.get("save_root") is not None:
                save_root = str(cfg.get("save_root"))
            else:
                sr2 = OmegaConf.select(cfg, "_env.save_root", default=None)
                if sr2 is not None:
                    save_root = str(sr2)
    except Exception:
        pass

    exp_group = None
    try:
        exp_group = OmegaConf.select(cfg, "_env.exp_group", default=None)
    except Exception:
        exp_group = None

    # save_root can be derived from a base checkpoint dir, so stage-2 runs land next to stage-1.
    save_root_from_ckpt = False
    try:
        sr = str(save_root).strip().lower()
        if sr in ("from_base_ckpt", "from_base_ckpt_dir", "base_ckpt_dir", "ckpt_dir"):
            ckpt_p = None
            try:
                ckpt_p = OmegaConf.select(cfg, "model.args.base_ckpt", default=None)
            except Exception:
                ckpt_p = None
            if ckpt_p is None:
                raise ValueError("save_root requests base_ckpt dir but cfg.model.args.base_ckpt is missing.")
            # Guard against unfilled placeholder paths.
            if str(ckpt_p).startswith("/ABS/"):
                raise ValueError(
                    "cfg.save_root=from_base_ckpt requires a real cfg.model.args.base_ckpt, "
                    f"but it is still the placeholder: {ckpt_p}. "
                    "Pass: --opt model.args.base_ckpt /path/to/best-model.pth"
                )
            save_root = os.path.dirname(str(ckpt_p))
            save_root_from_ckpt = True
        elif sr in ("from_stage1_dir", "from_stage1", "stage1_dir", "stage1"):
            stage1_dir = OmegaConf.select(cfg, "stage1_dir", default=None)
            if stage1_dir is None:
                # Fallback: derive from an explicit stage-1 ckpt path.
                stage1_ckpt = OmegaConf.select(cfg, "sample_vis.stage1_ckpt", default=None)
                if stage1_ckpt is None:
                    stage1_ckpt = OmegaConf.select(cfg, "sample.stage1_ckpt", default=None)
                if stage1_ckpt is None:
                    raise ValueError("save_root=from_stage1_dir requires cfg.stage1_dir or sample(_vis).stage1_ckpt")
                stage1_dir = os.path.dirname(str(stage1_ckpt))
            if str(stage1_dir).startswith("/ABS/"):
                raise ValueError(
                    "cfg.save_root=from_stage1_dir requires a real cfg.stage1_dir, "
                    f"but it is still the placeholder: {stage1_dir}. "
                    "Pass: --opt stage1_dir /path/to/STAGE1_RUN_DIR"
                )
            save_root = str(stage1_dir)
            save_root_from_ckpt = True
    except Exception:
        pass

    # Stash save_root/exp_group explicitly to bypass BaseTrainer's "default/" auto-nesting
    # when saving directly next to a checkpoint.
    env.save_root = str(save_root)
    if save_root_from_ckpt:
        env.exp_group = ""
    elif exp_group is not None:
        env.exp_group = str(exp_group)

    if getattr(env, "exp_group", None):
        env.save_dir = os.path.join(str(save_root), str(env.exp_group), exp_name)
    else:
        env.save_dir = os.path.join(str(save_root), exp_name)
    env.wandb = args.wandb
    env.resume_mode = args.resume_mode
    
    cfg._env = env
    return cfg


if __name__ == '__main__':
    args = make_args()
    cfg = make_cfg(args)
    # diffusion trainers register lazily; surface the real import error instead of a bare KeyError.
    if cfg.trainer not in trainers_dict:
        try:
            import diffusion.train  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                f"Trainer {cfg.trainer!r} not found. Also failed to import diffusion.train "
                "(required for hip_token_* trainers)."
            ) from e
    trainer = trainers_dict[cfg.trainer](cfg)
    trainer.run()
