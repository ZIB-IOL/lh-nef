"""Bake ImageNet (train/val/test) into FFCV .beton files at 256x256.

The split parameters mirror the imagenet 256^2 cfg exactly:
  train: split='train', train_val_split='train', train_val_fraction=0.99, seed=0
  val:   split='train', train_val_split='val',   train_val_fraction=0.99, seed=0
  test:  split='val'   (official ImageNet val/ dir, 50K images)
"""
from __future__ import annotations

# IMPORTANT: import cv2 *before* PIL/torch so it claims the conda-forge libjpeg
# (which has the jpeg12 symbols libtiff expects). If PIL grabs a system libjpeg
# first, opencv's libtiff fails to load with "undefined symbol jpeg12_write_raw_data".
import cv2  # noqa: F401

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

REPO_SRC = Path(__file__).resolve().parents[1]  # src/
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import datasets  # noqa: F401  (registers imagenet_labeled via __init__)
from datasets.imagenet_labeled import ImageNetLabeled

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField


class ImageNet256BakeSource(Dataset):
    """Mirrors the PyTorch pipeline up to (but not including) ToTensor+Normalize.
    Returns (uint8 256x256x3 ndarray, int label)."""

    def __init__(self, **inet_kwargs):
        self.ds = ImageNetLabeled(return_label=True, **inet_kwargs)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        sample = self.ds[i]
        # ImageNetLabeled already does: convert('RGB'), center-square crop,
        # and BICUBIC upsample to 256 if the square side < min_side_floor.
        img = sample['image']
        label = int(sample['label'])
        # Wrapper's LANCZOS resize to 256x256:
        img = img.resize((256, 256), Image.LANCZOS)
        arr = np.array(img, dtype=np.uint8)
        if arr.shape != (256, 256, 3):
            raise RuntimeError(f"unexpected shape {arr.shape} (sample {i})")
        return arr, label


SPLIT_CFG_TEMPLATE = {
    'train': dict(split='train', train_val_split='train',
                  train_val_fraction=0.99, train_val_seed=0),
    'val':   dict(split='train', train_val_split='val',
                  train_val_fraction=0.99, train_val_seed=0),
    'test':  dict(split='val'),
}


def bake(split_name: str, out_path: str, root_path: str, num_workers: int,
         min_side_floor: int = 256, overwrite: bool = False):
    if os.path.exists(out_path) and not overwrite:
        print(f"[skip] {out_path} already exists (--overwrite to force)")
        return
    if os.path.exists(out_path) and overwrite:
        os.remove(out_path)

    kwargs = dict(SPLIT_CFG_TEMPLATE[split_name])
    kwargs.update(root_path=root_path, min_side_floor=min_side_floor)
    src = ImageNet256BakeSource(**kwargs)
    n = len(src)
    print(f"[bake] split={split_name}  n={n:,}  -> {out_path}")

    t0 = time.time()
    writer = DatasetWriter(out_path, {
        'image': RGBImageField(write_mode='raw', max_resolution=None),
        'label': IntField(),
    }, num_workers=num_workers)
    writer.from_indexed_dataset(src)
    dt = time.time() - t0
    sz = os.path.getsize(out_path) / 1e9
    rate = n / max(dt, 1e-6)
    print(f"[bake] {split_name}: DONE in {dt/60:.1f} min  size={sz:.2f} GB  rate={rate:.1f} img/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-root', default='/scratch/htc/aurbano/datasets/imagenet256_ffcv')
    ap.add_argument('--root-path', default='/scratch/llm/ais2t/datasets/pytorch/imagenet',
                    help='ImageNet root containing train/ and val/ subdirs')
    ap.add_argument('--num-workers', type=int, default=16)
    ap.add_argument('--splits', nargs='+', default=['test', 'val', 'train'],
                    help='order: test (50K, fast) -> val (~12K) -> train (~1.27M)')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--min-side-floor', type=int, default=256)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        if split not in SPLIT_CFG_TEMPLATE:
            raise ValueError(f"unknown split: {split}")
        out_path = str(out_root / f'{split}.beton')
        bake(split, out_path, root_path=args.root_path,
             num_workers=args.num_workers,
             min_side_floor=args.min_side_floor,
             overwrite=args.overwrite)


if __name__ == '__main__':
    main()
