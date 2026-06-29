from __future__ import annotations

import json
import os
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import time

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import datasets
import models
import utils
from trainers import register
from trainers.base_trainer import BaseTrainer


def _to_device(x, device: torch.device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_device(v, device) for v in x]
    return x


def _block_index_from_key(k: str) -> int:
    if not isinstance(k, str):
        return 10**9
    if k.startswith("block"):
        try:
            return int(k[5:])
        except Exception:
            return 10**9
    return 10**9


def _select_block_key(enc_blocks: Dict[str, torch.Tensor], sel: str | int) -> str:
    keys = [k for k in enc_blocks.keys() if isinstance(k, str) and k.startswith("block")]
    if not keys:
        raise ValueError("enc_blocks has no 'block*' entries.")
    keys = sorted(keys, key=_block_index_from_key)
    if sel == "last":
        return keys[-1]
    if isinstance(sel, int):
        key = f"block{int(sel)}"
        if key not in enc_blocks:
            raise KeyError(f"Requested {key} not found in enc_blocks keys={list(enc_blocks.keys())}")
        return key
    raise ValueError(f"Unknown block selection: {sel!r} (expected 'last' or int)")


@dataclass
class _RunningMoments:
    """
    Streaming mean/std for latent normalization.

    mode="channel":       shapes [C]   (reduce over images + tokens).
    mode="token_channel": shapes [L,C] (reduce over images only).
    """

    mode: str
    n: int
    sum: torch.Tensor  # float64, shape depends on mode
    sumsq: torch.Tensor  # float64, shape depends on mode

    @classmethod
    def create(cls, *, mode: str, L: int, C: int, device: torch.device) -> "_RunningMoments":
        mode = str(mode or "channel").lower().strip()
        if mode not in ("channel", "token_channel"):
            raise ValueError("moments mode must be 'channel' or 'token_channel'")
        if mode == "channel":
            z = torch.zeros((int(C),), device=device, dtype=torch.float64)
        else:
            z = torch.zeros((int(L), int(C)), device=device, dtype=torch.float64)
        return cls(mode=mode, n=0, sum=z.clone(), sumsq=z.clone())

    def update(self, x: torch.Tensor) -> None:
        # x: [B,L,C]
        if x.ndim != 3:
            raise ValueError(f"moments.update expects [B,L,C], got {tuple(x.shape)}")
        xx = x.detach().to(dtype=torch.float64)
        B, L, C = map(int, xx.shape)
        if self.mode == "channel":
            self.n += int(B * L)
            self.sum += xx.sum(dim=(0, 1))
            self.sumsq += (xx * xx).sum(dim=(0, 1))
            return
        if int(self.sum.shape[0]) != int(L) or int(self.sum.shape[1]) != int(C):
            raise ValueError(f"token_channel moments shape mismatch: buffer={tuple(self.sum.shape)} vs x={(L,C)}")
        self.n += int(B)
        self.sum += xx.sum(dim=0)
        self.sumsq += (xx * xx).sum(dim=0)

    def finalize(self, eps: float = 1e-12) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.n <= 0:
            raise RuntimeError("Cannot finalize moments with n=0.")
        mean = self.sum / float(self.n)
        var = (self.sumsq / float(self.n)) - (mean * mean)
        var = torch.clamp(var, min=0.0)
        std = torch.sqrt(var + float(eps))
        return mean.to(dtype=torch.float32), std.to(dtype=torch.float32)


@register("hip_latent_extract_trainer")
class HipLatentExtractTrainer(BaseTrainer):
    """
    DDP-safe extractor running a frozen stage-1 LH-NeF checkpoint and caching token latents.

    Per split outputs: shards/*.pt with (c, p, meta) and stats_rank*.pt for mean/std aggregation.
    Rank0 writes manifest.json.
    """

    def run(self):
        if self.cfg.random_seed is not None:
            self.seed_everything(self.cfg.random_seed, rank_shift=True)

        # Match extraction encoder-input distribution to stage-1 training.
        self._sync_n_inp_from_stage1_ckpt()

        self.make_datasets()
        self._extract_all()

        if self.enable_tb:
            self.writer.close()
        if self.enable_wandb:
            import wandb
            wandb.finish()

    def _sync_n_inp_from_stage1_ckpt(self):
        """
        Inherit encoder-input fields (resize_inp, n_inp, train_val_fraction/seed) from
        the stage-1 checkpoint's training cfg, into the extraction cfg.

        Each field is detected at wrapper vs inner level and written to the same level
        per extract split. train_val_fraction/seed only sync when the split actually uses
        a train/val partition. Mismatches are logged; missing fields are skipped.
        """
        try:
            ex = self.cfg.get("extract", {}) or {}
            ckpt_path = ex.get("stage1_ckpt", None)
            if ckpt_path is None:
                return
            ckpt_path = os.path.expanduser(os.path.expandvars(str(ckpt_path)))
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            s1_cfg = ckpt.get("cfg", {}) or {}
            s1_train = (s1_cfg.get("datasets", {}) or {}).get("train", {}) or {}
            s1_wrapper = (s1_train.get("args", {}) or {})
            s1_inner = (s1_wrapper.get("dataset", {}) or {}).get("args", {}) or {}

            synced: dict = {}  # field_name -> (value, "wrapper" | "inner" | "inner_split")

            for key in ("resize_inp", "n_inp"):
                v_wrap = s1_wrapper.get(key, None)
                v_inner = s1_inner.get(key, None)
                if v_wrap is not None:
                    synced[key] = (v_wrap, "wrapper")
                elif v_inner is not None:
                    synced[key] = (v_inner, "inner")

            # "inner_split" writes only when the split uses train/val partitioning.
            for key in ("train_val_fraction", "train_val_seed"):
                v = s1_inner.get(key, None)
                if v is not None:
                    synced[key] = (v, "inner_split")

            if not synced:
                return

            for k, (v, lvl) in synced.items():
                self.log(f"[extract auto-sync] stage-1 {k}={v} (level={lvl})")

            for split in list((self.cfg.get("datasets", {}) or {}).keys()):
                ds_cfg = self.cfg.datasets[split]
                try:
                    wrapper_args = ds_cfg.args
                    # inner_args is None for wrappers without an inner dataset spec.
                    try:
                        inner_args = ds_cfg.args.dataset.args
                    except Exception:
                        inner_args = None

                    uses_partition = (
                        inner_args is not None
                        and inner_args.get("train_val_split", None) is not None
                    )

                    for k, (v, lvl) in synced.items():
                        if lvl == "wrapper":
                            target = wrapper_args
                        elif lvl == "inner":
                            target = inner_args
                        elif lvl == "inner_split":
                            target = inner_args if uses_partition else None
                        else:
                            target = None
                        if target is None:
                            continue
                        cur = target.get(k, None)
                        if cur is not None and cur != v:
                            self.log(f"[extract auto-sync] {split} {k}={cur} -> {v} (from stage-1 ckpt)")
                        target[k] = v
                except Exception as e:
                    self.log(f"[extract auto-sync] {split} could not sync: {e}")
        except Exception as e:
            self.log(f"[extract auto-sync] could not read fields from stage-1 ckpt: {e}")

    def _load_stage1_model(self, ckpt_path: str):
        ckpt_path = os.path.expanduser(os.path.expandvars(str(ckpt_path)))
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        spec = ckpt["model"]
        net = models.make(spec, load_sd=True)
        net.eval().to(self.device)
        for p in net.parameters():
            p.requires_grad_(False)
        return net, ckpt

    def _extract_all(self):
        cfg = self.cfg
        ex = cfg.get("extract", {}) or {}

        stage1_ckpt = ex.get("stage1_ckpt", None)
        if stage1_ckpt is None:
            raise ValueError("extract.stage1_ckpt is required.")
        net, ckpt = self._load_stage1_model(stage1_ckpt)
        n_params = sum(p.numel() for p in net.parameters())
        self.log(f"Loaded stage-1 model ({n_params:,} params) from {stage1_ckpt}")

        # If extracting from a block not in render_blocks, expand them.
        block_sel = (ex.get("block", "last"))
        if isinstance(block_sel, int):
            encoder = getattr(net, "encoder", None)
            if encoder is not None and hasattr(encoder, "render_blocks"):
                if block_sel not in encoder.render_blocks:
                    encoder.render_blocks = sorted(set(encoder.render_blocks) | {block_sel})
                    self.log(f"[extract] expanded render_blocks to {encoder.render_blocks} to include block{block_sel}")

        out_root_cfg = ex.get("out_root", "diffusion/latents")
        out_root_cfg_s = str(out_root_cfg).strip()
        # Output directory policy:
        #   absolute path                              -> use as-is
        #   "from_stage1_ckpt"/"from_stage1_dir"/...   -> next to stage-1 ckpt
        #   otherwise                                  -> under this run's save_dir
        if os.path.isabs(out_root_cfg_s):
            out_root = Path(out_root_cfg_s)
        else:
            key = out_root_cfg_s.lower()
            if key in ("from_stage1_ckpt", "from_stage1_dir", "stage1_dir", "stage1"):
                stage1_dir = Path(os.path.dirname(str(stage1_ckpt)))
                out_subdir = str(ex.get("out_subdir", "extract")).strip()
                out_root = stage1_dir / out_subdir
            else:
                out_root = Path(self.cfg._env.save_dir) / out_root_cfg_s

        def _sel(path: str, default):
            try:
                return OmegaConf.select(cfg, path, default=default)
            except Exception:
                return default

        splits = list(ex.get("splits", ["train"]))
        if not splits:
            raise ValueError("extract.splits must be a non-empty list.")

        block_sel = ex.get("block", "last")
        if isinstance(block_sel, str) and block_sel.strip().lower() in ("last", "block_last", "max"):
            block_sel = "last"
        elif isinstance(block_sel, int):
            block_sel = int(block_sel)
        else:
            if isinstance(block_sel, str) and block_sel.startswith("block"):
                block_sel = int(block_sel[5:])
            else:
                raise ValueError("extract.block must be 'last' or int or 'block{idx}'.")

        shard_size = int(ex.get("shard_size", 2048))
        if shard_size <= 0:
            raise ValueError("extract.shard_size must be > 0.")

        save_dtype = str(ex.get("save_dtype", "float16")).lower().strip()
        if save_dtype not in ("float16", "fp16", "float32", "fp32"):
            raise ValueError("extract.save_dtype must be float16|float32.")
        save_fp16 = save_dtype in ("float16", "fp16")

        use_amp = bool(ex.get("use_amp", True))
        stats_mode = str(ex.get("stats_mode", "channel")).lower().strip()
        if stats_mode not in ("channel", "token_channel"):
            raise ValueError("extract.stats_mode must be 'channel' or 'token_channel'")

        # If true, persist `label` (or `y`) from each batch into shards.
        save_labels = bool(ex.get("save_labels", False))

        stage1_cfg_digest = None
        if self.is_master:
            try:
                cfg_json = json.dumps(ckpt.get("cfg", {}), sort_keys=True, default=str)
                stage1_cfg_digest = hashlib.sha1(cfg_json.encode("utf-8")).hexdigest()[:10]
            except Exception:
                stage1_cfg_digest = None

        manifest: Dict[str, Any] = {
            "stage1_ckpt": str(stage1_ckpt),
            "stage1_cfg_digest": stage1_cfg_digest,
            "splits": {},
        }

        for split in splits:
            if split not in self.datasets or self.datasets.get(split) is None:
                raise KeyError(f"Requested split {split!r} not present in cfg.datasets.")

            split_dir = out_root / str(split)
            # All ranks mkdir (BaseTrainer only mkdirs save_dir on rank0).
            os.makedirs(split_dir, exist_ok=True)
            if self.distributed:
                dist.barrier()

            ds = self._maybe_repeat_dataset(self.datasets[split], split=split, ex=ex)
            batch_size = int(ex.get("batch_size", _sel(f"datasets.{split}.loader.batch_size", 256)))
            loader, _sampler = self.make_distributed_loader(
                ds,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
                num_workers=int(ex.get("num_workers", _sel(f"datasets.{split}.loader.num_workers", 4))),
            )
            n_batches = len(loader)
            self.log(f"[{split}] Extracting {len(ds)} samples in {n_batches} batches (bs={batch_size})")

            shard_paths: List[str] = []
            moments: Optional[_RunningMoments] = None
            token_pos: Optional[torch.Tensor] = None  # [L,d] float32
            token_pos_src: Optional[Dict[str, Any]] = None  # for provenance
            shape_info: Optional[Dict[str, int]] = None
            group_scales: Optional[torch.Tensor] = None  # [G,d] float32 (bbox extents / lambda_g)
            has_labels_rank = False
            y_min_rank: Optional[int] = None
            y_max_rank: Optional[int] = None

            buf_c: List[torch.Tensor] = []
            buf_y: List[torch.Tensor] = []
            n_in_buf = 0
            shard_idx = 0
            n_extracted = 0
            t_start = time.time()

            for batch_idx, batch in enumerate(loader):
                batch = _to_device(batch, self.device)
                yb = None
                if save_labels:
                    if isinstance(batch, dict):
                        yb = batch.get("label", None)
                        if yb is None:
                            yb = batch.get("y", None)
                    if yb is None:
                        raise KeyError(
                            "extract.save_labels=true but batch has no 'label'/'y'. "
                            "Ensure dataset is configured with return_label=true."
                        )
                    if not torch.is_tensor(yb):
                        yb = torch.tensor(yb, dtype=torch.long, device=self.device)
                    yb = yb.to(dtype=torch.long)
                    if yb.ndim != 1:
                        yb = yb.view(-1)
                    has_labels_rank = True

                    # Strip label keys before calling stage-1 model.
                    if isinstance(batch, dict):
                        batch = dict(batch)
                        batch.pop("label", None)
                        batch.pop("y", None)
                with torch.no_grad():
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        out = net(batch, mode="encode")
                enc_blocks = out.get("enc_blocks", None)
                enc_regions = out.get("enc_regions", None)
                if not isinstance(enc_blocks, dict) or len(enc_blocks) == 0:
                    raise RuntimeError("Stage-1 model did not return enc_blocks in mode='encode'.")
                if not isinstance(enc_regions, dict) or enc_regions.get("routing_space", None) != "coord":
                    raise RuntimeError("Stage-1 model did not return coord-space enc_regions in mode='encode'.")

                block_key = _select_block_key(enc_blocks, block_sel)
                t = enc_blocks[block_key]
                if (not torch.is_tensor(t)) or t.ndim != 4:
                    raise RuntimeError(f"enc_blocks[{block_key}] must be [B,G,K,C], got {type(t)} shape={getattr(t,'shape',None)}")
                B, G, K, C = map(int, t.shape)
                L = G * K

                # Build token positions p once: repeat group centers K times.
                if token_pos is None:
                    blk = (enc_regions.get("blocks", {}) or {}).get(block_key, None)
                    if not isinstance(blk, dict):
                        raise RuntimeError(f"enc_regions.blocks missing {block_key}.")
                    centers = blk.get("centers", None)
                    scales = blk.get("scales", None)
                    if (not torch.is_tensor(centers)) or centers.ndim != 3:
                        raise RuntimeError(f"enc_regions.blocks[{block_key}].centers must be [B,G,d], got {type(centers)} {getattr(centers,'shape',None)}")
                    # `scales` = bbox extents (lambda_g); constant across the dataset for image grids.
                    if (not torch.is_tensor(scales)) or scales.ndim != 3:
                        raise RuntimeError(
                            f"enc_regions.blocks[{block_key}].scales must be [B,G,d], got {type(scales)} {getattr(scales,'shape',None)}"
                        )
                    # centers are constant across batch; take first sample.
                    p = centers[0].detach().to(dtype=torch.float32)  # [G,d]
                    s = scales[0].detach().to(dtype=torch.float32).clamp_min(1e-6)  # [G,d]
                    p = p.repeat_interleave(K, dim=0).contiguous()  # [L,d]
                    token_pos = p.cpu()
                    token_pos_src = {
                        "block_key": block_key,
                        "coord_dim": int(p.shape[-1]),
                        "G": G,
                        "K": K,
                        "C": C,
                        "L": L,
                    }
                    group_scales = s.cpu()
                    shape_info = {"G": G, "K": K, "C": C, "L": L}
                    moments = _RunningMoments.create(mode=stats_mode, L=L, C=C, device=self.device)

                c = t.reshape(B, L, C).detach()
                if moments is not None:
                    moments.update(c)
                if save_fp16:
                    c = c.to(dtype=torch.float16).cpu()
                else:
                    c = c.to(dtype=torch.float32).cpu()

                buf_c.append(c)
                if save_labels:
                    yy = yb.detach().to(device="cpu", dtype=torch.long).contiguous()
                    buf_y.append(yy)
                    try:
                        mn = int(yy.min().item())
                        mx = int(yy.max().item())
                        y_min_rank = mn if y_min_rank is None else min(int(y_min_rank), mn)
                        y_max_rank = mx if y_max_rank is None else max(int(y_max_rank), mx)
                    except Exception:
                        pass
                n_in_buf += int(c.shape[0])
                n_extracted += int(c.shape[0])

                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                    elapsed = time.time() - t_start
                    rate = n_extracted / elapsed if elapsed > 0 else 0
                    self.log(f"[{split}] {batch_idx+1}/{n_batches} batches | {n_extracted} samples | {rate:.0f} samples/s")

                if n_in_buf >= shard_size:
                    shard_path = split_dir / f"rank{self.rank:03d}_shard{shard_idx:06d}.pt"
                    payload = {
                        "c": torch.cat(buf_c, dim=0),
                        "p": token_pos,  # [L,d]
                        "group_scales": group_scales,  # [G,d] (lambda_g)
                        "meta": {
                            "split": split,
                            "rank": int(self.rank),
                            "shard_idx": int(shard_idx),
                            "block_key": str(token_pos_src["block_key"]),
                            "shape": dict(shape_info or {}),
                        },
                    }
                    if save_labels:
                        payload["y"] = torch.cat(buf_y, dim=0)  # [N] int64
                    torch.save(payload, str(shard_path))
                    shard_paths.append(str(shard_path))
                    self.log(f"[{split}] Wrote shard {shard_idx} ({int(payload['c'].shape[0])} samples) -> {shard_path.name}")
                    buf_c = []
                    buf_y = []
                    n_in_buf = 0
                    shard_idx += 1

            if n_in_buf > 0:
                shard_path = split_dir / f"rank{self.rank:03d}_shard{shard_idx:06d}.pt"
                payload = {
                    "c": torch.cat(buf_c, dim=0),
                    "p": token_pos,  # [L,d] CPU
                    "group_scales": group_scales,  # [G,d] CPU (bbox extents / lambda_g)
                    "meta": {
                        "split": split,
                        "rank": int(self.rank),
                        "shard_idx": int(shard_idx),
                        "block_key": str(token_pos_src["block_key"]) if token_pos_src else "unknown",
                        "shape": dict(shape_info or {}),
                    },
                }
                if save_labels:
                    payload["y"] = torch.cat(buf_y, dim=0)  # [N] int64
                torch.save(payload, str(shard_path))
                shard_paths.append(str(shard_path))
                self.log(f"[{split}] Wrote final shard {shard_idx} ({int(payload['c'].shape[0])} samples) -> {shard_path.name}")

            elapsed_total = time.time() - t_start
            self.log(f"[{split}] Done: {n_extracted} samples, {shard_idx + (1 if n_in_buf > 0 else 0)} shards, {elapsed_total:.1f}s")

            if moments is None:
                raise RuntimeError("No data extracted; moments is None.")
            if group_scales is None:
                raise RuntimeError("No group scales extracted; group_scales is None.")
            stats_path = split_dir / f"stats_rank{self.rank:03d}.pt"
            torch.save(
                {
                    "n": int(moments.n),
                    "sum": moments.sum.detach().cpu(),
                    "sumsq": moments.sumsq.detach().cpu(),
                    "shape": shape_info,
                    "token_pos_src": token_pos_src,
                    "group_scales": group_scales,  # [G,d] CPU
                    "has_labels": bool(has_labels_rank),
                    "y_min": y_min_rank,
                    "y_max": y_max_rank,
                    "shards": shard_paths,
                },
                str(stats_path),
            )

            if self.distributed:
                dist.barrier()

            if self.is_master:
                stats_files = sorted([str(p) for p in split_dir.glob("stats_rank*.pt")])
                if not stats_files:
                    raise RuntimeError(f"No stats_rank*.pt files found under {split_dir}")
                agg_n = 0
                agg_sum = None
                agg_sumsq = None
                all_shards: List[str] = []
                shape0 = None
                token_pos_src0 = None
                group_scales0 = None
                has_labels0 = False
                y_min0: Optional[int] = None
                y_max0: Optional[int] = None
                for sf in stats_files:
                    st = torch.load(sf, map_location="cpu", weights_only=False)
                    agg_n += int(st["n"])
                    s = st["sum"]
                    ss = st["sumsq"]
                    if agg_sum is None:
                        agg_sum = s.clone()
                        agg_sumsq = ss.clone()
                        shape0 = st.get("shape", None)
                        token_pos_src0 = st.get("token_pos_src", None)
                        group_scales0 = st.get("group_scales", None)
                        has_labels0 = bool(st.get("has_labels", False))
                        y_min0 = st.get("y_min", None)
                        y_max0 = st.get("y_max", None)
                    else:
                        agg_sum += s
                        agg_sumsq += ss
                        has_labels0 = bool(has_labels0 or bool(st.get("has_labels", False)))
                        ymn = st.get("y_min", None)
                        ymx = st.get("y_max", None)
                        if ymn is not None:
                            y_min0 = int(ymn) if y_min0 is None else min(int(y_min0), int(ymn))
                        if ymx is not None:
                            y_max0 = int(ymx) if y_max0 is None else max(int(y_max0), int(ymx))
                    all_shards.extend(list(st.get("shards", [])))
                if agg_sum is None or agg_sumsq is None or agg_n <= 0:
                    raise RuntimeError("Failed to aggregate stats.")
                mean = agg_sum / float(agg_n)
                var = (agg_sumsq / float(agg_n)) - (mean * mean)
                var = torch.clamp(var, min=0.0)
                std = torch.sqrt(var + 1e-12)

                num_classes = None
                if has_labels0 and (y_min0 is not None) and (y_max0 is not None):
                    # Assumes contiguous integer labels.
                    num_classes = int(y_max0) - int(y_min0) + 1
                manifest["splits"][split] = {
                    "split": split,
                    "block_key": (token_pos_src0 or {}).get("block_key", None),
                    "shape": shape0,
                    "p_is_constant": True,
                    "group_scales_is_constant": True,
                    "group_scales": (group_scales0.to(dtype=torch.float32).tolist() if torch.is_tensor(group_scales0) else None),
                    "has_labels": bool(has_labels0),
                    "y_min": y_min0,
                    "y_max": y_max0,
                    "num_classes": num_classes,
                    "stats_mode": str(stats_mode),
                    "mean": mean.to(dtype=torch.float32).tolist(),
                    "std": std.to(dtype=torch.float32).tolist(),
                    "shards": sorted(set(all_shards)),
                }

        if self.is_master:
            os.makedirs(out_root, exist_ok=True)
            mpath = out_root / "manifest.json"
            mpath.write_text(json.dumps(manifest, indent=2))
            self.log(f"Manifest written to {mpath}")
            for sp, info in manifest["splits"].items():
                n_shards = len(info.get("shards", []))
                shape = info.get("shape", {})
                self.log(f"  {sp}: {n_shards} shards, shape={shape}, has_labels={info.get('has_labels', False)}")

    @staticmethod
    def _maybe_repeat_dataset(ds, *, split: str, ex: dict):
        """Repeat each item `extract.repeat_per_item` times (only on the train split)."""
        try:
            rpt = int(ex.get("repeat_per_item", 1) or 1)
        except Exception:
            rpt = 1
        if rpt <= 1 or str(split) != "train":
            return ds

        from torch.utils.data import Dataset

        class _RepeatIndexDataset(Dataset):
            def __init__(self, base, repeat: int):
                self.base = base
                self.repeat = int(repeat)
                self.N = int(len(base))
                if self.N <= 0:
                    raise ValueError("Cannot repeat empty dataset")

            def __len__(self):
                return int(self.N * self.repeat)

            def __getitem__(self, idx):
                i = int(idx) % int(self.N)
                return self.base[i]

        return _RepeatIndexDataset(ds, rpt)
