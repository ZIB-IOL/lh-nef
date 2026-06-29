from __future__ import annotations

import copy
import math
import os
from typing import Dict

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import trainers
from trainers import register
from trainers.base_trainer import BaseTrainer


def _resolve_manifest_path(cfg) -> str:
    try:
        ds = cfg.get("datasets", {}).get("train", {}) or {}
        args = ds.get("args", {}) or {}
        mp = args.get("manifest_path", None)
        if mp is None:
            return ""
        mp = str(mp)
        mp = os.path.expanduser(os.path.expandvars(mp))
        return mp
    except Exception:
        return ""


def _infer_num_classes_from_manifest(cfg) -> int | None:
    mp = _resolve_manifest_path(cfg)
    if not mp or (not os.path.isfile(mp)):
        return None
    try:
        import json

        with open(mp, "r") as f:
            manifest = json.load(f)
        split = str(cfg.get("datasets", {}).get("train", {}).get("args", {}).get("split", "train"))
        s = (manifest.get("splits", {}) or {}).get(split, {}) or {}
        nc = s.get("num_classes", None)
        if nc is not None:
            return int(nc)
    except Exception:
        return None
    return None


def _infer_token_dim_from_manifest(cfg) -> int | None:
    mp = _resolve_manifest_path(cfg)
    if not mp or (not os.path.isfile(mp)):
        return None
    try:
        import json

        with open(mp, "r") as f:
            manifest = json.load(f)
        split = str(cfg.get("datasets", {}).get("train", {}).get("args", {}).get("split", "train"))
        s = (manifest.get("splits", {}) or {}).get(split, {}) or {}
        shape = s.get("shape", {}) or {}
        C = shape.get("C", None)
        if C is not None:
            return int(C)
    except Exception:
        return None
    return None


def _infer_gk_from_manifest(cfg) -> tuple[int | None, int | None]:
    mp = _resolve_manifest_path(cfg)
    if not mp or (not os.path.isfile(mp)):
        return None, None
    try:
        import json

        with open(mp, "r") as f:
            manifest = json.load(f)
        split = str(cfg.get("datasets", {}).get("train", {}).get("args", {}).get("split", "train"))
        s = (manifest.get("splits", {}) or {}).get(split, {}) or {}
        shape = s.get("shape", {}) or {}
        G = shape.get("G", None)
        K = shape.get("K", None)
        return (int(G) if G is not None else None, int(K) if K is not None else None)
    except Exception:
        return None, None


@register("latent_classifier_trainer")
class LatentClassifierTrainer(BaseTrainer):
    """Train a classifier on cached stage-1 token latents.

    Adds a final TEST evaluation using the best-val checkpoint.
    """

    def run(self):
        # Force registration side-effects rather than relying on try/except imports elsewhere.
        try:
            import classification.data  # noqa: F401
            import classification.models  # noqa: F401
        except Exception:
            pass

        if self.cfg.random_seed is not None:
            self.seed_everything(self.cfg.random_seed, rank_shift=True)

        self.make_datasets()

        # NOTE: BaseTrainer.run() handles eval_only, but this trainer overrides run(),
        # so we must handle eval_only explicitly here.
        if bool(self.cfg.get("eval_only", False)):
            ckpt_path = self.cfg.get("eval_model", None)
            if ckpt_path is None:
                raise ValueError("eval_only=true requires eval_model=/path/to/{best,last}-model.pth")
            # Load the model spec from the checkpoint itself so any "from_manifest"
            # placeholders are already resolved (or at least consistent with training).
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            model_spec = ckpt.get("model", None)
            if model_spec is None:
                raise KeyError(f"checkpoint has no 'model' key: {ckpt_path}")
            self.make_model(model_spec)
            self.iter = 0
            self.evaluate_split("test")
            if self.enable_tb:
                self.writer.close()
            if self.enable_wandb:
                import wandb
                wandb.finish()
            return

        self._maybe_auto_schedule()

        # Resume logic mirrors BaseTrainer.run().
        resume_file = os.path.join(self.cfg._env.save_dir, "last-model.pth")
        if os.path.isfile(resume_file):
            ckpt = torch.load(resume_file, map_location="cpu", weights_only=False)
            model_spec = copy.deepcopy(OmegaConf.to_container(self.cfg.model, resolve=True))
            model_spec["sd"] = ckpt["model"]["sd"]
            self.make_model(model_spec)
            self.make_optimizers()
            opt_dict = ckpt.get("optimizers", {})
            for k, v in opt_dict.items():
                if k in self.optimizers:
                    self.optimizers[k].load_state_dict(v["sd"])
            if self.use_amp and ckpt.get("grad_scaler") is not None:
                self.grad_scaler.load_state_dict(ckpt["grad_scaler"])
            self.log(f"Resumed from checkpoint {resume_file}.")
        else:
            self.make_model()
            self.make_optimizers()

        # Track EMA of params and use at eval time.
        self._init_ema()

        # LR scheduler (cosine + optional linear warmup); stepped per-iter inside train_step.
        self._init_lr_schedulers()

        self.run_training()

        # Final: evaluate test using best-model.pth.
        best_file = os.path.join(self.cfg._env.save_dir, "best-model.pth")
        if os.path.isfile(best_file):
            if self.is_master:
                self.log(f"[classification] evaluating best-model on test: {best_file}")
            ckpt = torch.load(best_file, map_location="cpu", weights_only=False)
            try:
                self.model.load_state_dict(ckpt["model"]["sd"], strict=False)
            except Exception:
                self.model.load_state_dict(ckpt["model"]["sd"], strict=False)
            test_metrics = self.evaluate_split("test")
            if self.is_master:
                try:
                    out_path = os.path.join(self.cfg._env.save_dir, "classification_metrics.json")
                    import json

                    payload = {"test": {k: float(v.item()) for k, v in test_metrics.items()}}
                    with open(out_path, "w") as f:
                        json.dump(payload, f, indent=2)
                    self.log(f"[classification] wrote {out_path}")
                except Exception as e:
                    self.log(f"[classification] could not write metrics json: {e}")
        else:
            if self.is_master:
                self.log("[classification] best-model.pth not found; skipping test eval.")

        if self.enable_tb:
            self.writer.close()
        if self.enable_wandb:
            import wandb

            wandb.finish()

    def make_model(self, model_spec=None):
        # Auto-inject token_dim / num_classes if requested.
        try:
            model_cfg = self.cfg.get("model", {}) or {}
            args = dict(model_cfg.get("args", {}) or {})

            if str(args.get("token_dim", "")).lower() == "from_manifest":
                td = _infer_token_dim_from_manifest(self.cfg)
                if td is not None:
                    args["token_dim"] = int(td)

            if str(args.get("num_classes", "")).lower() == "from_manifest":
                nc = _infer_num_classes_from_manifest(self.cfg)
                if nc is not None:
                    args["num_classes"] = int(nc)

            # For structure-aware models
            if str(args.get("num_groups", "")).lower() == "from_manifest" or str(args.get("tokens_per_group", "")).lower() == "from_manifest":
                G, K = _infer_gk_from_manifest(self.cfg)
                if str(args.get("num_groups", "")).lower() == "from_manifest" and G is not None:
                    args["num_groups"] = int(G)
                if str(args.get("tokens_per_group", "")).lower() == "from_manifest" and K is not None:
                    args["tokens_per_group"] = int(K)

            model_cfg["args"] = args
            self.cfg["model"] = model_cfg
        except Exception:
            pass

        super().make_model(model_spec=model_spec)

    def _init_lr_schedulers(self):
        """Build a cosine (with optional linear warmup) LR schedule, applied per-iter.

        Config (under ``lr_scheduler``):
          name: ``cosine`` | ``none``  (default ``none``)
          warmup_iters: int            (default 0; linear warmup from 0 -> base_lr)
          min_lr_ratio: float in [0,1) (default 0.0; cosine ends at base_lr * min_lr_ratio)
        """
        sched_cfg = self.cfg.get("lr_scheduler", None) or {}
        try:
            name = str(sched_cfg.get("name", "none") or "none").lower().strip()
        except Exception:
            name = "none"
        self._lr_sched_active = (name == "cosine")
        if not self._lr_sched_active:
            return
        max_iter = int(self.cfg.get("max_iter", 0) or 0)
        if max_iter <= 0:
            raise RuntimeError(
                "lr_scheduler=cosine requires max_iter > 0 (set auto_schedule=true and train_epochs>0)."
            )
        warmup = int(sched_cfg.get("warmup_iters", 0) or 0)
        min_ratio = float(sched_cfg.get("min_lr_ratio", 0.0) or 0.0)
        if warmup < 0 or warmup >= max_iter:
            raise ValueError(f"lr_scheduler.warmup_iters must be in [0, max_iter); got {warmup} (max_iter={max_iter}).")
        if not (0.0 <= min_ratio < 1.0):
            raise ValueError(f"lr_scheduler.min_lr_ratio must be in [0, 1); got {min_ratio}.")
        self._lr_sched_max_iter = max_iter
        self._lr_sched_warmup = warmup
        self._lr_sched_min_ratio = min_ratio
        # Snapshot base LRs (one per param_group, per optimizer) so we can scale them every step.
        self._lr_sched_base_lrs = {
            name_o: [float(pg['lr']) for pg in opt.param_groups]
            for name_o, opt in self.optimizers.items()
        }
        if self.is_master:
            self.log(
                f"[lr_scheduler] cosine: max_iter={max_iter} warmup_iters={warmup} "
                f"min_lr_ratio={min_ratio} base_lrs={self._lr_sched_base_lrs}"
            )

    def _lr_factor(self, it: int) -> float:
        """Cosine factor in [min_lr_ratio, 1] with optional linear warmup."""
        w = int(self._lr_sched_warmup)
        m = int(self._lr_sched_max_iter)
        if w > 0 and it < w:
            # Linear from 0 -> 1 across [0, w). At it=0 the factor is 1/w (small but nonzero).
            return float(it + 1) / float(w)
        denom = max(1, m - w)
        progress = (int(it) - w) / float(denom)
        progress = max(0.0, min(1.0, progress))
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self._lr_sched_min_ratio + (1.0 - self._lr_sched_min_ratio) * cos_factor

    def _apply_lr(self) -> float:
        """Set per-step LR on all param groups; return the current scaled LR (for logging)."""
        if not getattr(self, "_lr_sched_active", False):
            # Just report the first-group LR for logging.
            for opt in self.optimizers.values():
                return float(opt.param_groups[0]['lr'])
            return 0.0
        factor = self._lr_factor(int(self.iter))
        cur_lr_log = 0.0
        for name_o, opt in self.optimizers.items():
            base = self._lr_sched_base_lrs[name_o]
            for pg, lr0 in zip(opt.param_groups, base):
                pg['lr'] = float(lr0) * float(factor)
            cur_lr_log = float(opt.param_groups[0]['lr'])
        return cur_lr_log

    def _init_ema(self):
        cfg = self.cfg.get("ema", {}) or {}
        self.ema_enable = bool(cfg.get("enable", False))
        self.ema_decay = float(cfg.get("decay", 0.9999))
        self.ema_use_for_eval = bool(cfg.get("use_for_eval", True))
        self.ema_device = str(cfg.get("device", "cuda")).lower().strip()
        self.ema_sd = None
        if not self.ema_enable:
            return
        # Keep EMA on-device to avoid per-step CPU transfers (memory pressure / instability).
        dev = self.ema_device
        sd = self.model.state_dict()
        ema_sd = {}
        for k, v in sd.items():
            vv = v.detach()
            if torch.is_floating_point(vv):
                vv = vv.to(dtype=torch.float32)
            if dev != "cpu":
                vv = vv.to(device=dev)
            else:
                vv = vv.cpu()
            ema_sd[k] = vv.clone()
        self.ema_sd = ema_sd

    @torch.no_grad()
    def _ema_update(self):
        if not getattr(self, "ema_enable", False):
            return
        if self.ema_sd is None:
            return
        d = float(self.ema_decay)
        sd = self.model.state_dict()
        dev = self.ema_device
        for k, v in sd.items():
            if k not in self.ema_sd:
                vv = v.detach()
                if torch.is_floating_point(vv):
                    vv = vv.to(dtype=torch.float32)
                vv = vv.to(device=dev) if dev != "cpu" else vv.cpu()
                self.ema_sd[k] = vv.clone()
                continue
            vv = v.detach()
            if torch.is_floating_point(vv):
                vv = vv.to(dtype=torch.float32)
            vv = vv.to(device=dev) if dev != "cpu" else vv.cpu()
            if torch.is_floating_point(vv):
                self.ema_sd[k].mul_(d).add_(vv, alpha=(1.0 - d))
            else:
                self.ema_sd[k].copy_(vv)

    def _load_ema_to_model(self):
        if self.ema_sd is None:
            return
        dev = next(self.model.parameters()).device
        model_sd = self.model.state_dict()
        sd = {}
        for k, v in self.ema_sd.items():
            tgt = model_sd.get(k, None)
            if tgt is not None:
                sd[k] = v.to(device=dev, dtype=tgt.dtype)
            else:
                sd[k] = v.to(device=dev)
        self.model.load_state_dict(sd, strict=False)

    @torch.no_grad()
    def evaluate_split(self, split: str) -> Dict[str, "utils.Averager"]:
        # Same semantics as BaseTrainer.evaluate but for an arbitrary split.
        import utils

        if split not in self.loaders:
            return {}

        # Evaluate using EMA weights when enabled (restore after).
        backup = None
        if getattr(self, "ema_enable", False) and getattr(self, "ema_use_for_eval", False) and (self.ema_sd is not None):
            backup = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self._load_ema_to_model()

        loader = self.loaders[split]
        ave_scalars = {}
        self.model_ddp.eval()

        for data in loader:
            def _to_device(x):
                if torch.is_tensor(x):
                    return x.cuda()
                if isinstance(x, (list, tuple)):
                    return [_to_device(_v) for _v in x]
                if isinstance(x, dict):
                    return {kk: _to_device(vv) for kk, vv in x.items()}
                return x

            data = _to_device(data)
            with torch.no_grad():
                ret = self.train_step(data, bp=False)
            bs = 1
            try:
                if isinstance(data, dict) and torch.is_tensor(data.get("y", None)):
                    bs = int(data["y"].shape[0])
            except Exception:
                bs = 1
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=bs)

        self.sync_ave_scalars_(ave_scalars)
        if self.is_master:
            for k, v in ave_scalars.items():
                self.log_scalar(f"{split}/{k}", v.item())
            self.log(f"{split}: " + " ".join([f"{k}={v.item():.4f}" for k, v in ave_scalars.items()]))
        if self.distributed:
            dist.barrier()

        if backup is not None:
            dev = next(self.model.parameters()).device
            self.model.load_state_dict({k: v.to(device=dev) for k, v in backup.items()}, strict=False)
        return ave_scalars

    def evaluate(self):
        # Override BaseTrainer.evaluate() so ckpt_select_metric can use val/acc.
        return self.evaluate_split("val")

    def train_step(self, data, bp=True):
        # BaseTrainer.train_step + cosine LR scaling + EMA update around the optimizer step.
        if bp:
            cur_lr = self._apply_lr()
        else:
            cur_lr = None
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            ret = self.model_ddp(data)
        loss = ret.pop('loss')
        ret['loss'] = loss.item()
        for k in list(ret.keys()):
            v = ret[k]
            if isinstance(v, torch.Tensor) and v.ndim == 0:
                ret[k] = float(v.detach().item())
        if bp:
            self.model_ddp.zero_grad()
            self.grad_scaler.scale(loss).backward()
            for o in self.optimizers.values():
                self.grad_scaler.step(o)
            self.grad_scaler.update()
            self._ema_update()
            if cur_lr is not None:
                ret['lr'] = float(cur_lr)
        return ret

