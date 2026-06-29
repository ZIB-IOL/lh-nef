"""
ShapeNet16 voxel occupancy dataset (coord/value interface).

Each shape is a 3D occupancy grid [D,H,W] in {0,1}. We expose coord/value pairs:
  inp: {'coord':[N_inp,3], 'value':[N_inp,1]}; gt_coord:[N_query,3]; gt:[N_query,1].

On-disk layout (preprocessed):
  root_path/{index_train.txt, index_val.txt (optional), index_test.txt, npz/<relpath>.npz}
Each .npz must contain key 'occ' (uint8/bool [D,H,W]).

Coordinates are voxel centers in [-1,1]^3. Occupancy is mapped to {-1,+1} by default
to match L1/MSE losses used elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import register


def _read_index(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    lines = [ln for ln in lines if ln and (not ln.startswith("#"))]
    return lines


def _split_train_val(files: list[str], *, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")
    rng = np.random.RandomState(int(seed))
    idx = np.arange(len(files))
    rng.shuffle(idx)
    n_val = int(round(float(val_fraction) * len(files)))
    n_val = max(1, min(n_val, len(files) - 1))
    val = [files[int(i)] for i in idx[:n_val]]
    tr = [files[int(i)] for i in idx[n_val:]]
    return tr, val


def _coords_from_flat_indices_zyx(
    flat_idx: np.ndarray, *, shape_zyx: Tuple[int, int, int]
) -> np.ndarray:
    """Return voxel-center coordinates in [-1,1]^3 for flat indices over (z,y,x)."""
    D, H, W = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
    z, y, x = np.unravel_index(flat_idx.astype(np.int64), (D, H, W))
    cz = (z.astype(np.float32) + 0.5) / float(D) * 2.0 - 1.0
    cy = (y.astype(np.float32) + 0.5) / float(H) * 2.0 - 1.0
    cx = (x.astype(np.float32) + 0.5) / float(W) * 2.0 - 1.0
    return np.stack([cz, cy, cx], axis=-1).astype(np.float32)


def _cell_from_shape_zyx(shape_zyx: Tuple[int, int, int]) -> np.ndarray:
    D, H, W = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
    # cell width = 2 / res in coord-space [-1,1]
    return np.array([2.0 / float(D), 2.0 / float(H), 2.0 / float(W)], dtype=np.float32)


@dataclass(frozen=True)
class _SampleCfg:
    n: int
    balance: bool


def _sample_indices_balanced(
    occ_flat: np.ndarray, *, cfg: _SampleCfg, rng: np.random.RandomState
) -> np.ndarray:
    """Sample cfg.n flat indices into occ_flat [V] in {0,1}, optionally balancing pos/neg."""
    V = int(occ_flat.shape[0])
    n = int(cfg.n)
    if n <= 0:
        raise ValueError("n must be > 0")
    if n >= V:
        return np.arange(V, dtype=np.int64)
    if (not cfg.balance) or (occ_flat.min() == occ_flat.max()):
        return rng.randint(low=0, high=V, size=(n,), dtype=np.int64)

    pos = np.flatnonzero(occ_flat > 0)
    neg = np.flatnonzero(occ_flat <= 0)
    if pos.size == 0 or neg.size == 0:
        return rng.randint(low=0, high=V, size=(n,), dtype=np.int64)

    n_pos = n // 2
    n_neg = n - n_pos
    sel_pos = pos[rng.randint(low=0, high=pos.size, size=(n_pos,), dtype=np.int64)]
    sel_neg = neg[rng.randint(low=0, high=neg.size, size=(n_neg,), dtype=np.int64)]
    out = np.concatenate([sel_pos, sel_neg], axis=0)
    rng.shuffle(out)
    return out.astype(np.int64)


@register("shapenet16_vox_occ")
class ShapeNet16VoxOcc(Dataset):
    def __init__(
        self,
        *,
        split: str = "train",
        root_path: str = "load/shapenet16_vox_occ",
        npz_subdir: str = "npz",
        # If index_val.txt is missing we split train deterministically.
        val_fraction: float = 0.1,
        val_seed: int = 42,
        n_inp: int = 2048,
        n_query: int = 8192,
        balance_inp: bool = True,
        balance_query: bool = False,
        occ_to_pm1: bool = True,
        max_items: Optional[int] = None,
        return_label: bool = False,
    ):
        super().__init__()
        split = str(split).lower().strip()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'")

        self.root_path = os.path.expanduser(os.path.expandvars(str(root_path)))
        self.npz_subdir = str(npz_subdir)
        self.split = split
        self.return_label = bool(return_label)
        self.n_inp = int(n_inp)
        self.n_query = int(n_query)
        self.balance_inp = bool(balance_inp)
        self.balance_query = bool(balance_query)
        self.occ_to_pm1 = bool(occ_to_pm1)
        self.val_fraction = float(val_fraction)
        self.val_seed = int(val_seed)

        if self.n_inp <= 0 or self.n_query <= 0:
            raise ValueError("n_inp and n_query must be > 0")

        idx_train = os.path.join(self.root_path, "index_train.txt")
        idx_val = os.path.join(self.root_path, "index_val.txt")
        idx_test = os.path.join(self.root_path, "index_test.txt")

        if not os.path.isfile(idx_train):
            raise FileNotFoundError(f"Missing {idx_train}. Run preprocessing or provide dataset root.")
        if not os.path.isfile(idx_test):
            raise FileNotFoundError(f"Missing {idx_test}. Run preprocessing or provide dataset root.")

        train_files = _read_index(idx_train)
        test_files = _read_index(idx_test)
        if len(train_files) == 0 or len(test_files) == 0:
            raise RuntimeError("Empty index file(s).")

        if os.path.isfile(idx_val):
            val_files = _read_index(idx_val)
        else:
            train_files, val_files = _split_train_val(train_files, val_fraction=self.val_fraction, seed=self.val_seed)

        if split == "train":
            self.files = train_files
        elif split == "val":
            self.files = val_files
        else:
            self.files = test_files

        if max_items is not None and int(max_items) > 0:
            self.files = self.files[:int(max_items)]

        if len(self.files) == 0:
            raise RuntimeError(f"No files for split={split!r} under {self.root_path}")

        if self.return_label:
            all_files = train_files + (val_files if val_files else []) + test_files
            synsets = sorted(set(f.split("/")[0] for f in all_files))
            self.synset_to_idx = {s: i for i, s in enumerate(synsets)}
            self.num_classes = len(synsets)
        else:
            self.synset_to_idx = {}
            self.num_classes = 0

    def __len__(self):
        return len(self.files)

    def _load_occ(self, rel: str) -> np.ndarray:
        path = os.path.join(self.root_path, self.npz_subdir, rel)
        if not path.endswith(".npz"):
            path = path + ".npz"
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing sample npz: {path}")
        data = np.load(path, allow_pickle=False)
        if "occ" not in data:
            raise KeyError(f"{path} missing key 'occ'")
        occ = data["occ"]
        if occ.ndim != 3:
            raise ValueError(f"{path} occ must be 3D [D,H,W], got {occ.shape}")
        if occ.dtype != np.uint8:
            occ = occ.astype(np.uint8, copy=False)
        occ = (occ > 0).astype(np.uint8, copy=False)
        return occ

    def __getitem__(self, idx: int):
        rel = self.files[int(idx)]
        occ = self._load_occ(rel)  # [D,H,W] in {0,1}
        D, H, W = int(occ.shape[0]), int(occ.shape[1]), int(occ.shape[2])
        V = int(D * H * W)
        occ_flat = occ.reshape(-1)

        # Deterministic per-item RNG (reproducible across workers).
        rng = np.random.RandomState(int(idx) + 17 * len(self.files))

        inp_idx = _sample_indices_balanced(
            occ_flat, cfg=_SampleCfg(n=self.n_inp, balance=self.balance_inp), rng=rng
        )
        qry_idx = _sample_indices_balanced(
            occ_flat, cfg=_SampleCfg(n=self.n_query, balance=self.balance_query), rng=rng
        )

        inp_coord = _coords_from_flat_indices_zyx(inp_idx, shape_zyx=(D, H, W))
        gt_coord = _coords_from_flat_indices_zyx(qry_idx, shape_zyx=(D, H, W))

        inp_occ = occ_flat[inp_idx].astype(np.float32).reshape(-1, 1)
        gt_occ = occ_flat[qry_idx].astype(np.float32).reshape(-1, 1)

        if self.occ_to_pm1:
            inp_value = inp_occ * 2.0 - 1.0
            gt = gt_occ * 2.0 - 1.0
            occ_value_kind = "occ_pm1"
        else:
            inp_value = inp_occ
            gt = gt_occ
            occ_value_kind = "occ_01"

        cell = _cell_from_shape_zyx((D, H, W))
        gt_cell = np.broadcast_to(cell.reshape(1, 3), (gt_coord.shape[0], 3)).copy()

        inp_coord_t = torch.from_numpy(inp_coord)
        inp_value_t = torch.from_numpy(inp_value)
        gt_coord_t = torch.from_numpy(gt_coord)
        gt_cell_t = torch.from_numpy(gt_cell)
        gt_t = torch.from_numpy(gt)

        out = {
            "inp": {"coord": inp_coord_t, "value": inp_value_t},
            "inp_coord": inp_coord_t,
            "inp_value": inp_value_t,
            "gt_coord": gt_coord_t,
            "gt_cell": gt_cell_t,
            "gt": gt_t,
            # Not a dense grid; queries are randomly sampled.
            "gt_is_grid": torch.tensor(0, dtype=torch.int64),
            "gt_grid_shape": torch.tensor([], dtype=torch.int64),
            # Tag triggers IoU metrics when starting with "occ" (see CLAUDE.md).
            "value_kind": occ_value_kind,
            "occ_shape_zyx": torch.tensor([D, H, W], dtype=torch.int64),
        }
        if self.return_label:
            synset = rel.split("/")[0]
            out["label"] = torch.tensor(self.synset_to_idx[synset], dtype=torch.int64)
        return out

