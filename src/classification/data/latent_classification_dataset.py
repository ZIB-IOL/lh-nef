from __future__ import annotations

import json
import os
import bisect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from datasets import register

__all__ = ["HipTokenLatentsLabeledDataset"]


def _load_manifest(path: str) -> Dict[str, Any]:
    p = Path(os.path.expanduser(os.path.expandvars(str(path)))).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"manifest.json not found: {p}")
    return json.loads(p.read_text())


@register("hip_token_latents_labeled")
class HipTokenLatentsLabeledDataset(Dataset):
    """Labeled dataset of cached HiP token latents (requires ``extract.save_labels=true``).

    Each item: ``c: [L,C]`` float16/32 (optionally normalized), ``p: [L,d]`` float32
    (constant across dataset), ``y: []`` int64 class id.
    """

    def __init__(
        self,
        *,
        manifest_path: str,
        split: str,
        normalize: bool = True,
        norm_scale: float = 1.0,
        # Split for normalization stats; default "train" prevents val/test leakage.
        norm_split: str = "train",
        dtype: str = "float16",
        # If true, return `c` as [G,K,C] and `p` as [G,d] group centers (for structure-aware models).
        return_grouped: bool = False,
        # If true, include `group_scales: [G,d]` (constant across dataset).
        include_group_scales: bool = True,
        # If false, stream shards from disk (use for large augmented datasets).
        load_in_memory: bool = True,
        shard_cache_size: int = 2,
        max_items: Optional[int] = None,
    ):
        super().__init__()
        self.manifest_path = str(manifest_path)
        self.split = str(split)
        self.normalize = bool(normalize)
        self.norm_scale = float(norm_scale)
        self.norm_split = str(norm_split)
        self.return_grouped = bool(return_grouped)
        self.include_group_scales = bool(include_group_scales)
        self.load_in_memory = bool(load_in_memory)
        self.shard_cache_size = int(shard_cache_size)
        self.max_items = None if max_items is None else int(max_items)

        dt = str(dtype).lower().strip()
        if dt not in ("float16", "fp16", "float32", "fp32"):
            raise ValueError("dtype must be float16|float32")
        self.dtype = torch.float16 if dt in ("float16", "fp16") else torch.float32

        manifest = _load_manifest(self.manifest_path)
        s = (manifest.get("splits", {}) or {}).get(self.split, None)
        if s is None:
            raise KeyError(f"Split {self.split!r} not found in manifest.splits")

        if not bool(s.get("has_labels", False)):
            raise ValueError(
                f"Manifest split {self.split!r} has_labels=false. "
                "Re-extract with extract.save_labels=true and ensure return_label=true in the dataset."
            )

        shape = s.get("shape", {}) or {}
        self.G = int(shape.get("G", 1))
        self.K = int(shape.get("K", 1))
        self.C = int(shape.get("C", 1))
        self.L = int(shape.get("L", self.G * self.K))
        self.L = int(self.G * self.K)  # canonical

        shards = list(s.get("shards", []) or [])
        if not shards:
            raise ValueError(f"Manifest split {self.split!r} has no shards")

        # Use norm_split (default: train) for stats to avoid leakage.
        norm_s = (manifest.get("splits", {}) or {}).get(self.norm_split, None)
        if norm_s is None:
            norm_s = s  # fallback to own split if norm_split not found
        mean_t = torch.tensor(norm_s.get("mean", []), dtype=torch.float32)
        std_t = torch.tensor(norm_s.get("std", []), dtype=torch.float32).clamp_min(1e-6)
        if mean_t.ndim == 1:
            if int(mean_t.numel()) == int(self.C):
                mean = mean_t.view(1, 1, self.C)
                std = std_t.view(1, 1, self.C)
            elif int(mean_t.numel()) == int(self.L * self.C):
                mean = mean_t.view(1, self.L, self.C)
                std = std_t.view(1, self.L, self.C)
            else:
                raise ValueError(
                    f"Unsupported mean/std length: mean={int(mean_t.numel())} (expected C={self.C} or L*C={self.L*self.C})"
                )
        elif mean_t.ndim == 2:
            if tuple(mean_t.shape) != (int(self.L), int(self.C)):
                raise ValueError(f"Unsupported mean/std shape: {tuple(mean_t.shape)} (expected {(self.L, self.C)})")
            mean = mean_t.view(1, self.L, self.C)
            std = std_t.view(1, self.L, self.C)
        else:
            raise ValueError(f"Unsupported mean/std ndim: {int(mean_t.ndim)}")
        self._mean = mean
        self._std = std * float(self.norm_scale)

        # p is constant across dataset; load once from the first shard.
        payload0 = torch.load(str(shards[0]), map_location="cpu", mmap=True, weights_only=False)
        p0 = payload0.get("p", None)
        if (not torch.is_tensor(p0)) or p0.ndim != 2:
            raise ValueError(f"Shard missing p [L,d], got {type(p0)} shape={getattr(p0,'shape',None)}")
        self.p_token = p0.to(dtype=torch.float32).contiguous()  # [L,d]
        if int(self.p_token.shape[0]) != int(self.L):
            raise ValueError(f"Shard p has L={int(self.p_token.shape[0])}, expected {self.L}")

        # Group centers: every K-th token position.
        self.p_group = self.p_token[:: int(self.K)].contiguous()  # [G,d]
        if int(self.p_group.shape[0]) != int(self.G):
            raise ValueError(f"Derived p_group has G={int(self.p_group.shape[0])}, expected {self.G}")

        self.group_scales = None
        if self.include_group_scales:
            gs = payload0.get("group_scales", None)
            if (not torch.is_tensor(gs)) or gs.ndim != 2:
                raise ValueError(
                    f"include_group_scales=true but shard missing group_scales [G,d], got {type(gs)} shape={getattr(gs,'shape',None)}"
                )
            if int(gs.shape[0]) != int(self.G):
                raise ValueError(f"group_scales has G={int(gs.shape[0])}, expected {self.G}")
            self.group_scales = gs.to(dtype=torch.float32).contiguous()

        self.shards = [str(s) for s in shards]
        self._shard_sizes: List[int] = []
        self._cum_sizes: List[int] = []
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_order: List[int] = []

        if self.load_in_memory:
            all_c: List[torch.Tensor] = []
            all_y: List[torch.Tensor] = []
            for sp in self.shards:
                payload = torch.load(str(sp), map_location="cpu", mmap=True, weights_only=False)
                c = payload.get("c", None)
                y = payload.get("y", None)
                if (not torch.is_tensor(c)) or c.ndim != 3:
                    raise ValueError(f"Shard {sp} missing c [N,L,C], got {type(c)} shape={getattr(c,'shape',None)}")
                if (not torch.is_tensor(y)) or y.ndim != 1:
                    raise ValueError(f"Shard {sp} missing y [N], got {type(y)} shape={getattr(y,'shape',None)}")
                all_c.append(c.to(dtype=torch.float32))
                all_y.append(y.to(dtype=torch.long))
            c_cat = torch.cat(all_c, dim=0)
            y_cat = torch.cat(all_y, dim=0)
            if int(c_cat.shape[0]) != int(y_cat.shape[0]):
                raise ValueError(f"c/y length mismatch: c={int(c_cat.shape[0])} y={int(y_cat.shape[0])}")
            if self.max_items is not None and int(c_cat.shape[0]) > int(self.max_items):
                c_cat = c_cat[: int(self.max_items)]
                y_cat = y_cat[: int(self.max_items)]
            if self.normalize:
                c_cat = (c_cat - self._mean) / self._std
            self.c = c_cat.to(dtype=self.dtype)
            self.y = y_cat
            self._N = int(self.c.shape[0])
        else:
            # Streaming path: index shard sizes only; load shards on demand.
            total = 0
            for sp in self.shards:
                payload = torch.load(str(sp), map_location="cpu", mmap=True, weights_only=False)
                c = payload.get("c", None)
                y = payload.get("y", None)
                if (not torch.is_tensor(c)) or c.ndim != 3:
                    raise ValueError(f"Shard {sp} missing c [N,L,C], got {type(c)} shape={getattr(c,'shape',None)}")
                if (not torch.is_tensor(y)) or y.ndim != 1:
                    raise ValueError(f"Shard {sp} missing y [N], got {type(y)} shape={getattr(y,'shape',None)}")
                n = int(c.shape[0])
                self._shard_sizes.append(n)
                total += n
                self._cum_sizes.append(total)
            if self.max_items is not None and total > int(self.max_items):
                total = int(self.max_items)
            self._N = int(total)
            self.c = None
            self.y = None

    def __len__(self) -> int:
        return int(self._N)

    def _get_from_shard(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        si = bisect.bisect_right(self._cum_sizes, int(idx))
        start = 0 if si == 0 else int(self._cum_sizes[si - 1])
        local = int(idx) - start

        # Simple LRU shard cache.
        if si in self._cache:
            c_sh, y_sh = self._cache[si]
        else:
            payload = torch.load(self.shards[si], map_location="cpu", mmap=True, weights_only=False)
            c_sh = payload["c"]
            y_sh = payload["y"]
            self._cache[si] = (c_sh, y_sh)
            self._cache_order.append(si)
            if len(self._cache_order) > max(1, int(self.shard_cache_size)):
                ev = self._cache_order.pop(0)
                if ev in self._cache:
                    self._cache.pop(ev)

        c = c_sh[local].to(dtype=torch.float32)
        y = y_sh[local].to(dtype=torch.long)
        return c, y

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = int(idx)
        if self.load_in_memory:
            c = self.c[i]
            y = self.y[i]
        else:
            c, y = self._get_from_shard(i)
            if self.normalize:
                c = (c - self._mean.squeeze(0)) / self._std.squeeze(0)
            c = c.to(dtype=self.dtype)

        if self.return_grouped:
            c = c.view(int(self.G), int(self.K), int(self.C))
            out = {"c": c, "p": self.p_group, "p_token": self.p_token, "y": y}
            if self.group_scales is not None:
                out["group_scales"] = self.group_scales
            return out
        out = {"c": c, "p": self.p_token, "y": y}
        if self.group_scales is not None:
            out["group_scales"] = self.group_scales
        return out

