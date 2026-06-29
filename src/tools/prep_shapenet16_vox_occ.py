"""
Preprocess ShapeNet16 voxel occupancy grids into a simple .npz + index format
consumable by `datasets.shapenet16_vox_occ.ShapeNet16VoxOcc`.
Setup:
  - ShapeNet-Part (16 classes) provides train/test splits (and segmentation point clouds).
  - ShapeNet provides voxelizations (.binvox) for ShapeNetCore models.

Inputs
  1) ShapeNet-Part segmentation benchmark:
     - contains train/test split jsons listing ShapeNetCore model paths
  2) ShapeNet voxelizations (binvox) for the corresponding models

This script:
  - reads the ShapeNet-Part split lists
  - resolves each model to a .binvox file under a voxel root
  - parses .binvox (run-length encoding)
  - optionally downsamples to a smaller resolution (e.g. 32 or 64)
  - writes:
      out_root/npz/<synset>/<modelId>.npz  (key: 'occ' in {0,1}, shape [D,H,W] as (z,y,x))
      out_root/index_{split}.txt          (relative paths without '.npz' suffix)

Notes:
  - binvox dims are stored as (x,y,z). We output arrays as (z,y,x) to match the repo's
    3D coord conventions elsewhere (see `src/tools/visualize_coord_group_regions.py`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_binvox_as_bool_zyx(path: Path) -> np.ndarray:
    """
    Read a .binvox file into a boolean occupancy array shaped [D,H,W] as (z,y,x).

    Implementation follows the common binvox spec:
      - header lines until "data"
      - then RLE pairs (value:uint8, count:uint8) repeated
      - voxel order is x-fastest, then z, then y (xzy)
        (widely used in reference binvox readers)

    We reshape to [x, z, y] then transpose to [z, y, x].
    """
    with open(path, "rb") as f:
        line = f.readline().decode("ascii", errors="ignore").strip()
        if not line.startswith("#binvox"):
            raise ValueError(f"{path} is not a binvox file (missing #binvox header). Got: {line!r}")

        dims = None
        # translate/scale are not needed for our [-1,1] normalized voxel-center convention
        while True:
            line = f.readline().decode("ascii", errors="ignore")
            if line == "":
                raise ValueError(f"{path} ended before 'data' marker.")
            line = line.strip()
            if line.startswith("dim"):
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError(f"{path} bad dim line: {line!r}")
                dims = (int(parts[1]), int(parts[2]), int(parts[3]))  # (x,y,z)
            if line.startswith("data"):
                break

        if dims is None:
            raise ValueError(f"{path} missing dim header.")
        dx, dy, dz = dims
        nvox = int(dx * dy * dz)

        # RLE stream is pairs of uint8: (value, count)
        raw = np.frombuffer(f.read(), dtype=np.uint8)
        if raw.size % 2 != 0:
            raise ValueError(f"{path} RLE data has odd length {raw.size}.")
        values = raw[0::2]
        counts = raw[1::2].astype(np.int64)
        expanded = np.repeat(values, counts)
        if expanded.size != nvox:
            raise ValueError(f"{path} decoded size {expanded.size} != expected {nvox} from dims={dims}.")
        occ = (expanded > 0)

        # binvox reference order is xzy: index = x + dx * (z + dz * y)
        occ_xzy = occ.reshape((dy, dz, dx)).transpose(2, 0, 1)  # (x,y,z) -> (x, y, z)? careful
        # The above is intentionally explicit: start from (y,z,x) then permute to (x,y,z)
        # Now convert to (z,y,x)
        occ_zyx = np.transpose(occ_xzy, (2, 1, 0))  # (x,y,z) -> (z,y,x)
        return occ_zyx.astype(np.uint8)


def downsample_maxpool_zyx(occ_zyx: np.ndarray, out_res: int) -> np.ndarray:
    D, H, W = occ_zyx.shape
    out_res = int(out_res)
    if out_res <= 0:
        raise ValueError("out_res must be > 0")
    if (D, H, W) == (out_res, out_res, out_res):
        return occ_zyx
    if not (D == H == W):
        raise ValueError(f"Expected cubic voxels, got {occ_zyx.shape}")
    if D % out_res != 0:
        raise ValueError(f"Cannot downsample {D} -> {out_res} evenly.")
    f = D // out_res
    x = occ_zyx.reshape(out_res, f, out_res, f, out_res, f)
    x = x.max(axis=(1, 3, 5))
    return x.astype(np.uint8)


def _find_split_files(shapenetpart_root: Path) -> Dict[str, Path]:
    """
    ShapeNetPart commonly provides JSON lists under train_test_split/.
    We accept a few common names.
    """
    cand = {
        "train": [
            shapenetpart_root / "train_test_split" / "shuffled_train_file_list.json",
            shapenetpart_root / "train_test_split" / "train_file_list.json",
        ],
        "val": [
            shapenetpart_root / "train_test_split" / "shuffled_val_file_list.json",
            shapenetpart_root / "train_test_split" / "val_file_list.json",
        ],
        "test": [
            shapenetpart_root / "train_test_split" / "shuffled_test_file_list.json",
            shapenetpart_root / "train_test_split" / "test_file_list.json",
        ],
    }
    out: Dict[str, Path] = {}
    for split, paths in cand.items():
        for p in paths:
            if p.is_file():
                out[split] = p
                break
    if "train" not in out or "test" not in out:
        raise FileNotFoundError(
            f"Could not find required split jsons under {shapenetpart_root}/train_test_split/. "
            f"Found: {list(out.keys())}"
        )
    return out


def _parse_shapenetpart_relpath(item: str) -> Tuple[str, str]:
    """
    ShapeNetPart split lists typically look like:
      "02691156/1a04e3eab45ca15dd86060f189eb133"
    or sometimes include leading path segments; we robustly take last two segments.
    Returns: (synset, model_id)
    """
    parts = [p for p in str(item).split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Unexpected split entry: {item!r}")
    synset, model_id = parts[-2], parts[-1]
    return synset, model_id


def _resolve_binvox(vox_root: Path, synset: str, model_id: str, binvox_name: str) -> Path:
    # Common ShapeNet layout: <synset>/<model_id>/models/<binvox_name>
    p1 = vox_root / synset / model_id / "models" / binvox_name
    if p1.is_file():
        return p1
    # Some dumps omit "models/"
    p2 = vox_root / synset / model_id / binvox_name
    if p2.is_file():
        return p2
    raise FileNotFoundError(f"Could not find binvox for {synset}/{model_id}. Tried {p1} and {p2}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapenetpart_root", type=str, required=True, help="Root of ShapeNetPart benchmark folder.")
    ap.add_argument("--shapenet_vox_root", type=str, required=True, help="Root of ShapeNet voxelization folder containing .binvox.")
    ap.add_argument("--out_root", type=str, required=True, help="Output root (writes index_*.txt and npz/).")
    ap.add_argument("--binvox_name", type=str, default="model.binvox", help="Filename of binvox within each model folder.")
    ap.add_argument("--out_res", type=int, default=32, help="Output voxel resolution (cubic). If matches input, no downsample.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz files.")
    args = ap.parse_args()

    shapenetpart_root = Path(os.path.expanduser(os.path.expandvars(args.shapenetpart_root))).resolve()
    shapenet_vox_root = Path(os.path.expanduser(os.path.expandvars(args.shapenet_vox_root))).resolve()
    out_root = Path(os.path.expanduser(os.path.expandvars(args.out_root))).resolve()

    split_jsons = _find_split_files(shapenetpart_root)
    _ensure_dir(out_root / "npz")

    for split, jpath in split_jsons.items():
        items = _read_json(jpath)
        if not isinstance(items, list):
            raise ValueError(f"{jpath} expected list, got {type(items)}")

        out_index: List[str] = []
        n_skipped = 0
        for it in items:
            synset, model_id = _parse_shapenetpart_relpath(it)
            try:
                binvox = _resolve_binvox(shapenet_vox_root, synset, model_id, str(args.binvox_name))
            except FileNotFoundError:
                n_skipped += 1
                print(f"  WARN: missing binvox for {synset}/{model_id}, skipping")
                continue

            occ_zyx = read_binvox_as_bool_zyx(binvox)
            occ_zyx = downsample_maxpool_zyx(occ_zyx, int(args.out_res))

            rel = f"{synset}/{model_id}"
            out_npz = out_root / "npz" / f"{rel}.npz"
            _ensure_dir(out_npz.parent)
            if out_npz.exists() and (not bool(args.overwrite)):
                out_index.append(rel)
                continue

            np.savez_compressed(out_npz, occ=occ_zyx.astype(np.uint8))
            out_index.append(rel)

        # Write index file
        idx_path = out_root / f"index_{split}.txt"
        with open(idx_path, "w", encoding="utf-8") as f:
            for rel in out_index:
                f.write(rel + "\n")

        print(f"[{split}] wrote {len(out_index)} items to {idx_path} (skipped {n_skipped} missing)")


if __name__ == "__main__":
    main()

