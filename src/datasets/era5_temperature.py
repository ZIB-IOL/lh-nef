"""
ERA5 2m temperature dataset (coord/value interface).

On-disk layout (from Dupont et al. preprocessed):
  root_path/era5_temp2m_16x_{train,val,test}/*.npz
Each .npz contains latitude (46,), longitude (90,), temperature (46,90) in Kelvin.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import register

# Global min/max from Dupont et al. (2021) for normalization to [0,1].
T_MIN = 202.66  # Kelvin
T_MAX = 320.93  # Kelvin


def _latlon_to_sphere(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Map lat/lon (degrees) to [N,3] unit-sphere coordinates."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


@register("era5_temperature")
class ERA5Temperature(Dataset):
    def __init__(
        self,
        *,
        split: str = "train",
        root_path: str = "load/era5",
        n_inp: int = 2048,
        n_query: int = 4140,
        to_pm1: bool = True,
        max_items: Optional[int] = None,
    ):
        super().__init__()
        split = str(split).lower().strip()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.root_path = os.path.expanduser(os.path.expandvars(str(root_path)))
        self.split = split
        self.n_inp = int(n_inp)
        self.n_query = int(n_query)
        self.to_pm1 = bool(to_pm1)

        split_dir = os.path.join(self.root_path, f"era5_temp2m_16x_{split}")
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Missing split directory: {split_dir}. "
                f"Download ERA5 data from Dupont et al. (2021)."
            )

        self.files = sorted([
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if f.endswith(".npz")
        ])

        if max_items is not None and int(max_items) > 0:
            self.files = self.files[:int(max_items)]

        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files in {split_dir}")

        # Precompute the grid coordinates (same for all samples).
        sample = np.load(self.files[0])
        lat = sample["latitude"]   # (46,) degrees, 90 to -90
        lon = sample["longitude"]  # (90,) degrees, 0 to 356
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        self.coords = _latlon_to_sphere(
            lat_grid.reshape(-1), lon_grid.reshape(-1)
        )  # [V, 3]
        self.V = int(self.coords.shape[0])  # 46 * 90 = 4140

        # Cell size in radians (rough angular extent per grid cell on the sphere).
        lat_step = np.deg2rad(abs(float(lat[1] - lat[0])))
        lon_step = np.deg2rad(abs(float(lon[1] - lon[0])))
        self.cell = np.array([lon_step, lon_step, lat_step], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[int(idx)])
        temp = data["temperature"].astype(np.float32).reshape(-1)  # [V]

        temp = (temp - T_MIN) / (T_MAX - T_MIN)
        temp = np.clip(temp, 0.0, 1.0)

        if self.to_pm1:
            temp = temp * 2.0 - 1.0
            value_kind = "temp_pm1"
        else:
            value_kind = "temp_01"

        rng = np.random.RandomState(int(idx) + 13 * len(self.files))

        n_inp = min(self.n_inp, self.V)
        inp_idx = rng.choice(self.V, size=n_inp, replace=(n_inp > self.V))

        n_query = min(self.n_query, self.V)
        if n_query >= self.V:
            qry_idx = np.arange(self.V, dtype=np.int64)
        else:
            qry_idx = rng.choice(self.V, size=n_query, replace=False)

        inp_coord = self.coords[inp_idx]  # [n_inp, 3]
        inp_value = temp[inp_idx].reshape(-1, 1)  # [n_inp, 1]

        gt_coord = self.coords[qry_idx]  # [n_query, 3]
        gt = temp[qry_idx].reshape(-1, 1)  # [n_query, 1]
        gt_cell = np.broadcast_to(
            self.cell.reshape(1, 3), (gt_coord.shape[0], 3)
        ).copy()

        return {
            "inp": {
                "coord": torch.from_numpy(inp_coord),
                "value": torch.from_numpy(inp_value),
            },
            "gt_coord": torch.from_numpy(gt_coord),
            "gt_cell": torch.from_numpy(gt_cell),
            "gt": torch.from_numpy(gt),
            "value_kind": value_kind,
        }
