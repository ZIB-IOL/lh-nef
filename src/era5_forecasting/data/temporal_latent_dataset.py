"""Paired (z_t, z_{t+1}) latent dataset built from extracted ERA5 latents.

Timestamps are parsed from ERA5 filenames; pairs are consecutive samples
1 hour apart.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

import datasets
from datasets import register


def _parse_era5_timestamp(filename: str) -> Optional[datetime]:
    """Parse ERA5 filename like '1979-01-01T01.npz' into datetime."""
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})", filename)
    if m is None:
        return None
    return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)))


@register("era5_temporal_latents")
class ERA5TemporalLatentsDataset(Dataset):
    """Dataset of (z_t, z_{t+1}) pairs from extracted ERA5 HiP latents.

    Normalization stats are always taken from ``norm_split`` (default "train")
    to avoid leakage when loading val/test.
    """

    def __init__(
        self,
        *,
        manifest_path: str,
        split: str = "train",
        era5_root: str = "",
        normalize: bool = True,
        norm_scale: float = 1.0,
        # Split from which to take normalization stats (default: train, no leakage).
        norm_split: str = "train",
        max_items: Optional[int] = None,
    ):
        super().__init__()
        self.split = str(split)
        self.normalize = bool(normalize)
        self.norm_scale = float(norm_scale)

        manifest_path = os.path.expanduser(os.path.expandvars(str(manifest_path)))
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        split_meta = manifest["splits"][self.split]
        self.shape = split_meta["shape"]
        self.G = int(self.shape["G"])
        self.K = int(self.shape["K"])
        self.C = int(self.shape["C"])
        self.L = int(self.shape["L"])

        # Norm stats from norm_split to avoid val/test leakage.
        norm_meta = manifest["splits"].get(norm_split, split_meta)
        if self.normalize and "mean" in norm_meta and "std" in norm_meta:
            mean = torch.tensor(norm_meta["mean"], dtype=torch.float32)
            std = torch.tensor(norm_meta["std"], dtype=torch.float32)
            if mean.ndim == 1 and mean.shape[0] == self.L * self.C:
                mean = mean.reshape(self.L, self.C)
                std = std.reshape(self.L, self.C)
            elif mean.shape[0] == self.C:
                mean = mean.reshape(1, self.C)
                std = std.reshape(1, self.C)
            self.mean = mean
            self.std = std * self.norm_scale
        else:
            self.mean = None
            self.std = None

        shard_paths = split_meta["shards"]
        all_c = []
        for sp in shard_paths:
            sp = os.path.expanduser(os.path.expandvars(sp))
            shard = torch.load(sp, map_location="cpu", weights_only=False)
            all_c.append(shard["c"])  # [N_shard, L, C]
        self.all_c = torch.cat(all_c, dim=0).float()  # [N_total, L, C]

        # Token positions are constant across the dataset; read from first shard.
        first_shard = torch.load(
            os.path.expanduser(os.path.expandvars(shard_paths[0])),
            map_location="cpu", weights_only=False,
        )
        self.p = first_shard["p"].float()  # [L, d]

        era5_root = os.path.expanduser(os.path.expandvars(str(era5_root)))
        split_dir = os.path.join(era5_root, f"era5_temp2m_16x_{self.split}")
        filenames = sorted([f for f in os.listdir(split_dir) if f.endswith(".npz")])

        # Extraction must preserve dataset ordering — latent i must correspond to filename i.
        if len(filenames) != self.all_c.shape[0]:
            raise RuntimeError(
                f"Mismatch: {len(filenames)} ERA5 files vs {self.all_c.shape[0]} extracted latents "
                f"in split '{self.split}'. Ensure extraction used the same dataset ordering."
            )

        timestamps = []
        for fn in filenames:
            ts = _parse_era5_timestamp(fn)
            if ts is None:
                raise ValueError(f"Cannot parse timestamp from: {fn}")
            timestamps.append(ts)

        self.pairs = []  # (idx_t, idx_t1)
        one_hour = timedelta(hours=1)
        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
        for i, ts in enumerate(timestamps):
            ts_next = ts + one_hour
            if ts_next in ts_to_idx:
                self.pairs.append((i, ts_to_idx[ts_next]))

        if max_items is not None and int(max_items) > 0:
            self.pairs = self.pairs[:int(max_items)]

    def __len__(self) -> int:
        return len(self.pairs)

    def _normalize(self, c: torch.Tensor) -> torch.Tensor:
        if self.mean is not None and self.std is not None:
            return (c - self.mean) / self.std.clamp_min(1e-6)
        return c

    def _denormalize(self, c: torch.Tensor) -> torch.Tensor:
        if self.mean is not None and self.std is not None:
            return c * self.std.clamp_min(1e-6) + self.mean
        return c

    def __getitem__(self, idx: int):
        i_t, i_t1 = self.pairs[idx]
        c_t = self.all_c[i_t]    # [L, C]
        c_t1 = self.all_c[i_t1]  # [L, C]

        if self.normalize:
            c_t = self._normalize(c_t)
            c_t1 = self._normalize(c_t1)

        return {
            "c": c_t,        # [L, C] input
            "c_next": c_t1,  # [L, C] target
            "p": self.p,      # [L, d] token positions on the sphere
            "idx_t": torch.tensor(i_t, dtype=torch.long),
            "idx_t1": torch.tensor(i_t1, dtype=torch.long),
        }
