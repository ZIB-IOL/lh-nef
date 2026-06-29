"""Evaluate a trained run on val/test split using its saved cfg + checkpoint.

BaseTrainer.evaluate() always evaluates cfg.datasets.val; this script swaps in
cfg.datasets.test (or flips the split field) so test-set numbers are reportable.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

from omegaconf import OmegaConf

# Mirror src/run.py: support both `import src` and legacy absolute imports
# like `import datasets`, `import models` regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parents[0]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure registries are populated
import trainers  # noqa: F401
from trainers.trainers import trainers_dict


def _infer_run_dir(ckpt_path: Path) -> Path:
    return ckpt_path.parent


def _select_ckpt(run_dir: Path, which: str) -> Path:
    which = str(which).strip().lower()
    if which in ("best", "best-model", "best_model"):
        p = run_dir / "best-model.pth"
    elif which in ("last", "last-model", "last_model"):
        p = run_dir / "last-model.pth"
    else:
        raise ValueError(f"--which must be best|last, got {which!r}")
    if not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def _ensure_eval_save_dir(cfg, run_dir: Path, split: str) -> None:
    # Write into a sibling subfolder so the original training run dir is untouched.
    try:
        env = cfg._env
    except Exception:
        env = OmegaConf.create()
        cfg._env = env

    env.wandb = False
    env.resume_mode = "replace"
    env.auto_unique = False
    env.exp_name = f"eval-{split}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
    env.save_root = str(run_dir)
    env.exp_group = "eval"
    env.save_dir = str(run_dir / "eval" / env.exp_name)


def _swap_to_split(cfg, split: str) -> None:
    split = str(split).strip().lower()
    if split not in ("val", "test"):
        raise ValueError(f"--split must be val|test, got {split!r}")

    if split == "val":
        return

    # Prefer explicit datasets.test spec if present.
    if cfg.get("datasets") is not None and cfg.datasets.get("test") is not None:
        cfg.datasets.val = cfg.datasets.test
        return

    # Fallback: flip the underlying dataset spec's split field when available.
    key = "datasets.val.args.dataset.args.split"
    if OmegaConf.select(cfg, key, default=None) is not None:
        OmegaConf.update(cfg, key, "test")
        return

    raise KeyError(
        "Could not switch to test split. Expected either cfg.datasets.test or "
        f"{key} to exist in cfg."
    )


def _apply_eval_batch_size(cfg) -> None:
    bs = None
    for k in ("test_batch_size", "eval_batch_size"):
        v = cfg.get(k, None)
        if v is not None:
            try:
                bs = int(v)
            except Exception:
                bs = None
            if bs is not None:
                break
    if bs is not None and cfg.get("datasets") is not None and cfg.datasets.get("val") is not None:
        if cfg.datasets.val.get("loader") is not None:
            cfg.datasets.val.loader.batch_size = int(bs)


def _verify_split_manifest(cfg) -> None:
    """Verify cfg-defined train/val/test splits are disjoint (CelebA-HQ only).

    Uses the exact dataset args from cfg.yaml; does not invent new seeds.
    """
    try:
        if cfg.get("datasets") is None:
            return
        ds = cfg.datasets
        if ds.get("train") is None or ds.get("val") is None or ds.get("test") is None:
            return
        def _under(spec):
            try:
                return spec.args.dataset
            except Exception:
                return None

        u_tr = _under(ds.train)
        u_va = _under(ds.val)
        u_te = _under(ds.test)
        if any(u is None for u in (u_tr, u_va, u_te)):
            return
        if not (str(u_tr.name) == str(u_va.name) == str(u_te.name) == "celebahq"):
            return

        import datasets as _datasets

        d_tr = _datasets.make({"name": "celebahq", "args": dict(u_tr.args)})
        d_va = _datasets.make({"name": "celebahq", "args": dict(u_va.args)})
        d_te = _datasets.make({"name": "celebahq", "args": dict(u_te.args)})
        ft = set(getattr(d_tr, "files", []) or [])
        fv = set(getattr(d_va, "files", []) or [])
        fe = set(getattr(d_te, "files", []) or [])
        ov_tv = ft.intersection(fv)
        ov_te = ft.intersection(fe)
        ov_ve = fv.intersection(fe)

        print(
            f"[eval] split-manifest check (celebahq): "
            f"|train|={len(ft)}, |val|={len(fv)}, |test|={len(fe)}, "
            f"overlap train∩val={len(ov_tv)}, train∩test={len(ov_te)}, val∩test={len(ov_ve)}"
        )
        if (len(ov_tv) + len(ov_te) + len(ov_ve)) != 0:
            ex = list(ov_tv)[:3] + list(ov_te)[:3] + list(ov_ve)[:3]
            raise RuntimeError(f"Dataset split overlap detected; refusing to evaluate. Examples: {ex[:5]}")
    except Exception as e:
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="", help="Run directory containing cfg.yaml and checkpoints.")
    ap.add_argument("--ckpt", type=str, default="", help="Path to a checkpoint .pth (overrides --run_dir/--which).")
    ap.add_argument("--which", type=str, default="best", help="Which ckpt in run_dir: best|last")
    ap.add_argument("--split", type=str, default="test", help="Split to evaluate: val|test")
    ap.add_argument("--no_vis", action="store_true", help="Disable visualization.")
    # render_res controls both PSNR grid and visualization resolution; encoder input is left unchanged.
    ap.add_argument("--render_res", type=int, default=0,
                    help="If >0, evaluate PSNR on a fixed render grid at this resolution "
                         "(sets datasets.val.args.resize_gt_lb/ub/final_crop_gt) and visualize at this resolution. "
                         "If 0, keep the run's original GT resolution from cfg.yaml (recommended for CelebA-HQ).")
    ap.add_argument("--render_batch_size", type=int, default=0,
                    help="If >0, override datasets.val.loader.batch_size for evaluation (after any test swap).")
    ap.add_argument("--vis_samples", type=int, default=16,
                    help="Max number of images to visualize (only used when visualization is enabled).")
    ap.add_argument("--vis_subset", type=int, default=64,
                    help="If >0, visualize on a fixed random subset of val/test of this size (master only).")
    ap.add_argument("--vis_seed", type=int, default=0)
    ap.add_argument("--keep_train_dataset", action="store_true",
                    help="By default this script drops cfg.datasets.train (and cfg.datasets.test) to avoid confusion "
                         "and speed up eval-only runs. Use this flag to keep them.")
    ap.add_argument("--verify_split_manifest", action="store_true",
                    help="(Recommended for CelebA-HQ) Verify that cfg-defined train/val/test splits are disjoint "
                         "using the exact dataset args from cfg.yaml. Refuses to run if overlap is detected.")

    # Advanced overrides; prefer --render_res / --render_batch_size.
    ap.add_argument("--val_resize_inp", type=int, default=0, help="(advanced) Override datasets.val.args.resize_inp.")
    ap.add_argument("--val_resize_gt", type=int, default=0, help="(advanced) Override datasets.val.args.resize_gt_lb/ub.")
    ap.add_argument("--val_final_crop_gt", type=int, default=0, help="(advanced) Override datasets.val.args.final_crop_gt.")
    ap.add_argument("--val_batch_size", type=int, default=0, help="(advanced) Override datasets.val.loader.batch_size.")
    args = ap.parse_args()

    ckpt_path = None
    run_dir = None
    if args.ckpt:
        ckpt_path = Path(args.ckpt).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--ckpt not found: {ckpt_path}")
        run_dir = _infer_run_dir(ckpt_path)
    else:
        if not args.run_dir:
            raise ValueError("Provide either --ckpt or --run_dir")
        run_dir = Path(args.run_dir).expanduser().resolve()
        ckpt_path = _select_ckpt(run_dir, args.which)

    cfg_path = run_dir / "cfg.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"cfg.yaml not found in run_dir: {cfg_path}")

    cfg = OmegaConf.load(str(cfg_path))
    OmegaConf.set_struct(cfg, False)

    if bool(args.verify_split_manifest):
        _verify_split_manifest(cfg)

    _swap_to_split(cfg, args.split)
    _apply_eval_batch_size(cfg)

    if int(args.val_resize_inp) > 0:
        OmegaConf.update(cfg, "datasets.val.args.resize_inp", int(args.val_resize_inp))
    if int(args.val_resize_gt) > 0:
        OmegaConf.update(cfg, "datasets.val.args.resize_gt_lb", int(args.val_resize_gt))
        OmegaConf.update(cfg, "datasets.val.args.resize_gt_ub", int(args.val_resize_gt))
    if int(args.val_final_crop_gt) > 0:
        OmegaConf.update(cfg, "datasets.val.args.final_crop_gt", int(args.val_final_crop_gt))
    if int(args.val_batch_size) > 0 and cfg.get("datasets") is not None and cfg.datasets.get("val") is not None:
        if cfg.datasets.val.get("loader") is not None:
            cfg.datasets.val.loader.batch_size = int(args.val_batch_size)

    if int(args.render_res) > 0:
        rr = int(args.render_res)
        OmegaConf.update(cfg, "datasets.val.args.resize_gt_lb", rr)
        OmegaConf.update(cfg, "datasets.val.args.resize_gt_ub", rr)
        OmegaConf.update(cfg, "datasets.val.args.final_crop_gt", rr)
        # Leave resize_inp unchanged -> higher-res eval becomes super-res from the trained encoder input.
        if not args.no_vis:
            try:
                cfg.visualize.resolution = rr
                cfg.visualize.ds_samples = int(args.vis_samples)
            except Exception:
                pass

    if int(args.render_batch_size) > 0:
        OmegaConf.update(cfg, "datasets.val.loader.batch_size", int(args.render_batch_size))

    if not args.no_vis and int(args.vis_subset) > 0:
        try:
            if cfg.get("visualize_subset") is None:
                cfg.visualize_subset = OmegaConf.create()
            cfg.visualize_subset.enable = True
            cfg.visualize_subset.n_samples = int(args.vis_subset)
            cfg.visualize_subset.seed = int(args.vis_seed)
            try:
                cfg.visualize_subset.batch_size = int(OmegaConf.select(cfg, "datasets.val.loader.batch_size"))
            except Exception:
                pass
            cfg.visualize_subset.num_workers = 0
        except Exception:
            pass

    _ensure_eval_save_dir(cfg, run_dir, str(args.split))

    if args.no_vis:
        try:
            cfg.visualize.resolution = 0
            cfg.visualize.ds_samples = 0
        except Exception:
            pass
        try:
            cfg.visualize_subset.enable = False
        except Exception:
            pass

    # Manual eval path: LHNeFBase forward uses `has_opt` to decide which modules to run.
    # In eval-only mode no optimizers are constructed, so we must force has_opt below.
    import torch

    if not bool(args.keep_train_dataset):
        try:
            ds_val = cfg.datasets.val
            cfg.datasets = OmegaConf.create({"val": ds_val})
        except Exception:
            pass

    trainer = trainers_dict[cfg.trainer](cfg)
    trainer.make_datasets()
    trainer._maybe_auto_schedule()

    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model_spec = raw["model"]
    trainer.make_model(model_spec)
    trainer.iter = 0

    # models.make() loads strict=False silently; surface missing/unexpected keys here.
    try:
        sd = model_spec.get("sd", None) if isinstance(model_spec, dict) else None
        if isinstance(sd, dict):
            res = trainer.model.load_state_dict(sd, strict=False)
            missing = list(getattr(res, "missing_keys", []) or [])
            unexpected = list(getattr(res, "unexpected_keys", []) or [])
            if trainer.is_master:
                trainer.log(f"[eval] load_state_dict strict=False: missing={len(missing)}, unexpected={len(unexpected)}")
                if len(missing) > 0:
                    trainer.log(f"[eval] missing_keys (first 20): {missing[:20]}")
                if len(unexpected) > 0:
                    trainer.log(f"[eval] unexpected_keys (first 20): {unexpected[:20]}")
    except Exception as e:
        if trainer.is_master:
            trainer.log(f"[eval] could not report state_dict load mismatch: {e}")

    # Force renderer to run during evaluation (no grads: evaluate() wraps no_grad).
    if hasattr(trainer, "has_opt"):
        trainer.has_opt = {"renderer": True}

    ave = trainer.evaluate()
    if trainer.is_master:
        # BaseTrainer.evaluate() fills log_buffer but only flushes in the training loop;
        # flush manually so eval-only runs print metrics.
        try:
            buf = list(getattr(trainer, "log_buffer", []) or [])
            if buf:
                trainer.log(", ".join(buf))
                trainer.log_buffer = []
        except Exception:
            pass

        try:
            n_items = len(trainer.datasets.get("val", []))
        except Exception:
            n_items = None
        msg = f"[eval] split={args.split} (evaluated via datasets.val), n={n_items}"
        # Different modalities expose different metrics (psnr for RGB, mse/mae for ERA5, iou for occupancy).
        metric_keys = ("psnr", "mse", "mae", "iou", "mse_loss", "l1_loss", "loss")
        parts = []
        for k in metric_keys:
            v = ave.get(k, None) if hasattr(ave, "get") else None
            if v is None:
                continue
            try:
                fv = float(v.item()) if hasattr(v, "item") else float(v)
            except Exception:
                continue
            parts.append(f"{k}={fv:.6g}")
        if parts:
            msg += ", " + ", ".join(parts)
        trainer.log(msg)

    if not args.no_vis:
        trainer.visualize()
        if trainer.is_master:
            img_path = Path(cfg._env.save_dir) / "vis_iter_0000000" / "val_recon.png"
            trainer.log(f"[eval] visualization path: {img_path}")


if __name__ == "__main__":
    main()

