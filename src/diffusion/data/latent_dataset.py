from __future__ import annotations

import json
import os
import bisect
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from datasets import register

__all__ = ["HipTokenLatentsDataset"]


@dataclass(frozen=True)
class _ManifestSplit:
    shards: List[str]
    mean: torch.Tensor  # [C] float32
    std: torch.Tensor  # [C] float32
    shape: Dict[str, int]
    block_key: str
    group_scales: Optional[torch.Tensor]  # [G,d] float32 (bbox extents / lambda_g)


def _load_manifest(manifest_path: str) -> Dict[str, Any]:
    mp = Path(os.path.expanduser(os.path.expandvars(str(manifest_path)))).resolve()
    if not mp.is_file():
        raise FileNotFoundError(f"manifest.json not found: {mp}")
    return json.loads(mp.read_text())


def _parse_split(manifest: Dict[str, Any], split: str, norm_split: str = "train") -> _ManifestSplit:
    splits = manifest.get("splits", {}) or {}
    if split not in splits:
        raise KeyError(f"Split {split!r} not found in manifest.splits keys={list(splits.keys())}")
    s = splits[split] or {}
    # Always take normalization stats from norm_split (default: train) to avoid leakage.
    ns = splits.get(norm_split, s) or s
    mean = torch.tensor(ns.get("mean", []), dtype=torch.float32)
    std = torch.tensor(ns.get("std", []), dtype=torch.float32)
    if mean.numel() == 0 or std.numel() == 0:
        raise ValueError(f"Manifest split {split!r} is missing mean/std.")
    shards = list(s.get("shards", []))
    if not shards:
        raise ValueError(f"Manifest split {split!r} has no shards.")
    shape = dict(s.get("shape", {}) or {})
    block_key = str(s.get("block_key", ""))
    gs = s.get("group_scales", None)
    group_scales = None
    if gs is not None:
        try:
            group_scales = torch.tensor(gs, dtype=torch.float32)
        except Exception:
            group_scales = None
    return _ManifestSplit(shards=shards, mean=mean, std=std, shape=shape, block_key=block_key, group_scales=group_scales)


@register("hip_token_latents")
class HipTokenLatentsDataset(Dataset):
    """
    Cached HiP-token latents (p, c) extracted from a stage-1 LH-NeF checkpoint.

    Each item:
      c: [L, C] float32 (optionally normalized) or [G, K*C] when flatten_groups=True.
      p: [L, d] float32 (constant for the dataset).
    """

    def __init__(
        self,
        *,
        manifest_path: str,
        split: str,
        normalize: bool = True,
        # Extra scale factor on std (Spatial-Functa-style knob).
        norm_scale: float = 1.0,
        max_items: Optional[int] = None,
        dtype: str = "float32",  # float32 | float16 (storage in RAM / cache)
        mode: str = "ram",       # ram | stream
        stream_cache_shards: int = 1,
        # If True: c becomes [G, K*C], p becomes [G, d] (unique per group).
        flatten_groups: bool = False,
        # If True: augment p with per-group bbox extents lambda_g (last dim becomes 2*d).
        include_group_scales_in_p: bool = False,
    ):
        super().__init__()
        self.manifest_path = str(manifest_path)
        self.split = str(split)
        self.normalize = bool(normalize)
        self.norm_scale = float(norm_scale)
        self.max_items = None if max_items is None else int(max_items)
        self.dtype = str(dtype).lower().strip()
        if self.dtype not in ("float32", "fp32", "float16", "fp16"):
            raise ValueError("dtype must be float32|float16")
        self.mode = str(mode or "ram").lower().strip()
        if self.mode not in ("ram", "stream"):
            raise ValueError("mode must be 'ram' or 'stream'")
        self.stream_cache_shards = int(stream_cache_shards)
        if self.stream_cache_shards <= 0:
            raise ValueError("stream_cache_shards must be >= 1")
        self.flatten_groups = bool(flatten_groups)
        self.include_group_scales_in_p = bool(include_group_scales_in_p)

        manifest = _load_manifest(self.manifest_path)
        ms = _parse_split(manifest, self.split)
        self.block_key = ms.block_key
        self.shape = ms.shape

        self._G = int(self.shape.get("G", 16))
        self._K = int(self.shape.get("K", 4))
        self._C = int(self.shape.get("C", 32))
        self._d = None  # inferred from p below

        # std is scaled by norm_scale (sampling must invert).
        # mean/std layout: [C] (channel-broadcast) or [L,C] (token-channel-broadcast).
        if ms.mean.ndim == 1:
            C = int(ms.mean.numel())
            self._mean = ms.mean.view(1, 1, C)
            self._std = ms.std.view(1, 1, C).clamp_min(1e-6) * float(self.norm_scale)
        elif ms.mean.ndim == 2:
            Lm, Cm = map(int, ms.mean.shape)
            # Expect L = G*K from manifest shape
            if int(Lm) != int(self._G * self._K) or int(Cm) != int(self._C):
                raise ValueError(
                    f"Manifest mean/std are [L,C]={tuple(ms.mean.shape)} but expected "
                    f"L={self._G*self._K}, C={self._C} from shape info."
                )
            if self.flatten_groups:
                raise ValueError("flatten_groups=True is not supported with token_channel mean/std.")
            self._mean = ms.mean.view(1, Lm, Cm)
            self._std = ms.std.view(1, Lm, Cm).clamp_min(1e-6) * float(self.norm_scale)
        else:
            raise ValueError(f"Unsupported mean/std shape in manifest: mean.ndim={ms.mean.ndim}")

        # p is constant across the dataset; load once from shard 0.
        p0: Optional[torch.Tensor] = None
        first = Path(ms.shards[0])
        payload0 = torch.load(str(first), map_location="cpu", weights_only=False)
        pp0 = payload0.get("p", None)
        if (not torch.is_tensor(pp0)) or pp0.ndim != 2:
            raise ValueError(f"Shard {first} missing p [L,d], got {type(pp0)} shape={getattr(pp0,'shape',None)}")
        p0 = pp0.to(dtype=torch.float32).contiguous()
        self._d = int(p0.shape[-1])

        # Optional per-group bbox extents (lambda_g); manifest first, shard fallback.
        gs0 = ms.group_scales
        if gs0 is None:
            gsp = payload0.get("group_scales", None)
            if torch.is_tensor(gsp) and gsp.ndim == 2:
                gs0 = gsp.to(dtype=torch.float32).contiguous()
        if gs0 is not None:
            if int(gs0.shape[0]) != int(self._G) or int(gs0.shape[1]) != int(self._d):
                raise ValueError(
                    f"group_scales shape mismatch: got {tuple(gs0.shape)} expected {(self._G, self._d)}"
                )
            gs0 = gs0.clamp_min(1e-6)

        if self.flatten_groups:
            # Original p is [G*K, d]; collapse to one position per group.
            if int(p0.shape[0]) == int(self._G * self._K):
                p0 = p0.view(self._G, self._K, -1).mean(dim=1).contiguous()  # [G, d]
            else:
                p0 = p0[:: self._K].contiguous()  # [G, d]
        self._p = p0
        self._group_scales = gs0  # [G,d] or None

        if self.include_group_scales_in_p:
            if self._group_scales is None:
                raise ValueError(
                    "include_group_scales_in_p=True but group_scales were not found in manifest/shards. "
                    "Re-extract latents with the updated extractor, or disable include_group_scales_in_p."
                )
            if self.flatten_groups:
                self._p = torch.cat([self._p, self._group_scales], dim=-1).contiguous()
            else:
                if int(self._p.shape[0]) != int(self._G * self._K):
                    raise ValueError(
                        f"Expected p to have L=G*K={self._G*self._K} rows but got {int(self._p.shape[0])}"
                    )
                scales_rep = self._group_scales.repeat_interleave(self._K, dim=0).contiguous()
                self._p = torch.cat([self._p, scales_rep], dim=-1).contiguous()

        if self.mode == "ram":
            all_c: List[torch.Tensor] = []
            for spath in ms.shards:
                p = Path(spath)
                if not p.is_file():
                    raise FileNotFoundError(f"Shard not found: {p}")
                payload = torch.load(str(p), map_location="cpu", weights_only=False)
                c = payload.get("c", None)
                pp = payload.get("p", None)
                if self.include_group_scales_in_p:
                    gsp = payload.get("group_scales", None)
                    if not torch.is_tensor(gsp):
                        raise ValueError(f"Shard {p} missing group_scales [G,d] required by include_group_scales_in_p.")
                    if gsp.ndim != 2:
                        raise ValueError(f"Shard {p} group_scales must be [G,d], got {tuple(gsp.shape)}")
                if (not torch.is_tensor(c)) or c.ndim != 3:
                    raise ValueError(f"Shard {p} missing c [N,L,C], got {type(c)} shape={getattr(c,'shape',None)}")
                if (not torch.is_tensor(pp)) or pp.ndim != 2:
                    raise ValueError(f"Shard {p} missing p [L,d], got {type(pp)} shape={getattr(pp,'shape',None)}")
                all_c.append(c)

            c_cat = torch.cat(all_c, dim=0)
            if self.max_items is not None and int(c_cat.shape[0]) > int(self.max_items):
                c_cat = c_cat[: int(self.max_items)]

            c_cat = c_cat.to(dtype=torch.float32)
            if self.normalize:
                c_cat = (c_cat - self._mean) / self._std

            if self.flatten_groups:
                N = c_cat.shape[0]
                c_cat = c_cat.view(N, self._G, self._K, self._C)
                c_cat = c_cat.reshape(N, self._G, self._K * self._C)  # [N, G, K*C]

            if self.dtype in ("float16", "fp16"):
                c_cat = c_cat.to(dtype=torch.float16)
            self._c = c_cat  # [N,L,C] or [N,G,K*C] if flatten_groups
            self._len = int(self._c.shape[0])
            self._shards = None
            self._shard_offsets = None
            self._cache = None
            return

        # stream mode
        self._c = None
        self._shards = [str(Path(s)) for s in ms.shards]
        self._cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()

        lengths: List[int] = []
        for spath in self._shards:
            p = Path(spath)
            payload = torch.load(str(p), map_location="cpu", weights_only=False)
            c = payload.get("c", None)
            pp = payload.get("p", None)
            if (not torch.is_tensor(c)) or c.ndim != 3:
                raise ValueError(f"Shard {p} missing c [N,L,C], got {type(c)} shape={getattr(c,'shape',None)}")
            if (not torch.is_tensor(pp)) or pp.ndim != 2:
                raise ValueError(f"Shard {p} missing p [L,d], got {type(pp)} shape={getattr(pp,'shape',None)}")
            # When include_group_scales_in_p=True, self._p has the scales concatenated, so we can't
            # compare against the raw shard p directly; skip the constancy check in that case.
            if not self.include_group_scales_in_p:
                if pp.shape != self._p.shape or (pp.to(dtype=torch.float32) != self._p).any():
                    raise ValueError("This dataset assumes constant token positions p across shards.")
            lengths.append(int(c.shape[0]))
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + int(n))
        self._shard_offsets = offsets
        self._len = int(offsets[-1])
        if self.max_items is not None:
            self._len = min(self._len, int(self.max_items))

    def _load_shard(self, shard_idx: int) -> torch.Tensor:
        # LRU cache keyed by shard index.
        assert self._shards is not None
        if shard_idx in self._cache:
            t = self._cache.pop(shard_idx)
            self._cache[shard_idx] = t
            return t
        spath = self._shards[int(shard_idx)]
        payload = torch.load(str(spath), map_location="cpu", weights_only=False)
        c = payload["c"].to(dtype=torch.float32)
        if self.normalize:
            c = (c - self._mean) / self._std
        if self.flatten_groups:
            N = c.shape[0]
            c = c.view(N, self._G, self._K, self._C)
            c = c.reshape(N, self._G, self._K * self._C)
        if self.dtype in ("float16", "fp16"):
            c = c.to(dtype=torch.float16)
        self._cache[shard_idx] = c
        while len(self._cache) > int(self.stream_cache_shards):
            self._cache.popitem(last=False)
        return c

    def __len__(self) -> int:
        if self.mode == "ram":
            return int(self._len)
        return int(self._len)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = int(idx)
        if i < 0 or i >= int(self._len):
            raise IndexError(i)
        if self.mode == "ram":
            c = self._c[i]
            return {"c": c, "p": self._p}

        # stream mode: global index -> shard index + in-shard index
        assert self._shard_offsets is not None
        s = bisect.bisect_right(self._shard_offsets, i) - 1
        s = max(0, min(int(s), len(self._shard_offsets) - 2))
        base = int(self._shard_offsets[s])
        j = int(i - base)
        shard_c = self._load_shard(s)
        c = shard_c[j]
        return {"c": c, "p": self._p}

