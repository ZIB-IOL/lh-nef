"""
Create a folder of "real" images (PNG) for FID evaluation.

Why: FID implementations typically take two *directories* of images: generated vs real.
This script helps you build the "real" directory once and reuse it across sweeps.

Supported modes:
- Copy/resize from an existing image folder (works for anything, incl. CelebA-HQ).
- Export CIFAR-10 from a local torchvision download (no network; assumes already present).
- Export the *exact* split used by an LH-NeF config (prevents train/val/test leakage).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from PIL import Image

import sys

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_CIFAR10_BASE_FOLDER = "cifar-10-batches-py"


def _iter_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def _save_png(im: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(out_path), format="PNG")


def _resize_to_square(im: Image.Image, size: int) -> Image.Image:
    if size is None:
        return im
    if im.size[0] == size and im.size[1] == size:
        return im
    # Bicubic is standard for resizing natural images.
    return im.resize((int(size), int(size)), resample=Image.BICUBIC)


def _resolve_path(p: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(p))).resolve()


def _resolve_cifar10_root(root: str) -> Path:
    """
    Resolve a user-provided CIFAR10 root into what torchvision expects.

    torchvision.datasets.CIFAR10 expects:
      <root>/cifar-10-batches-py/...

    Users sometimes pass the base folder itself; handle that case.
    """
    root_p = _resolve_path(root)
    if root_p.name == _CIFAR10_BASE_FOLDER:
        return root_p.parent
    # Common alternative: <root>/CIFAR10/cifar-10-batches-py
    if (root_p / "CIFAR10" / _CIFAR10_BASE_FOLDER).exists():
        return root_p / "CIFAR10"
    # Common alternative: <root>/cifar10/cifar-10-batches-py
    if (root_p / "cifar10" / _CIFAR10_BASE_FOLDER).exists():
        return root_p / "cifar10"
    return root_p


def export_from_folder(*, src_dir: str, out_dir: str, size: int | None, limit: int | None) -> int:
    src = Path(os.path.expanduser(os.path.expandvars(src_dir))).resolve()
    out = Path(os.path.expanduser(os.path.expandvars(out_dir))).resolve()
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for p in _iter_images(src):
        if limit is not None and n >= int(limit):
            break
        im = Image.open(str(p)).convert("RGB")
        im = _resize_to_square(im, size)
        _save_png(im, out / f"{n:06d}.png")
        n += 1
    return n


def export_cifar10(*, root: str, split: str, out_dir: str, limit: int | None, download: bool) -> int:
    # Lazy import so the script can still run without torchvision when using folder mode.
    from torchvision.datasets import CIFAR10  # type: ignore

    split = str(split).lower().strip()
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")

    root_p = _resolve_cifar10_root(root)
    if download:
        root_p.mkdir(parents=True, exist_ok=True)
    else:
        expected = root_p / _CIFAR10_BASE_FOLDER
        if not expected.exists():
            raise FileNotFoundError(
                "CIFAR-10 not found at the provided --root.\n\n"
                f"Expected to find: {expected}\n"
                "Fixes:\n"
                f"- Point --root to the directory that contains `{_CIFAR10_BASE_FOLDER}`\n"
                "- Or rerun with `--download` to fetch it via torchvision\n"
            )

    ds = CIFAR10(root=str(root_p), train=(split == "train"), download=bool(download))
    out = Path(os.path.expanduser(os.path.expandvars(out_dir))).resolve()
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for i in range(len(ds)):
        if limit is not None and n >= int(limit):
            break
        im, _y = ds[i]
        # CIFAR is already 32x32 RGB PIL.
        _save_png(im.convert("RGB"), out / f"{n:06d}.png")
        n += 1
    return n


def export_from_lhnef_cfg(*, cfg_path: str, split: str, out_dir: str, size: int | None, limit: int | None) -> int:
    """
    Export a "real images" directory from the EXACT dataset split defined in an LH-NeF YAML config.

    This is the safest way to avoid leakage: if your run's CIFAR10 "train" is actually
    (50k - val_size) due to a deterministic val split, this will export exactly that subset.
    """
    # Make repo importable (so `import datasets` works when running as a script).
    repo_root = Path(__file__).resolve().parents[2]
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from omegaconf import OmegaConf  # type: ignore

    import datasets as lhnef_datasets  # type: ignore

    cfg_p = Path(os.path.expanduser(os.path.expandvars(cfg_path))).resolve()
    cfg = OmegaConf.load(str(cfg_p))

    split = str(split).lower().strip()
    if split not in ("train", "val", "test"):
        raise ValueError("split must be one of: train/val/test")

    ds_spec = cfg.get("datasets", {}).get(split, None)
    if ds_spec is None:
        raise KeyError(f"cfg has no datasets.{split}")

    # Many image configs wrap the underlying dataset in wrapper_cae(_coord_value).
    # For FID, we want the RAW images from the underlying dataset split, without wrapper
    # randomness/augmentations. So, if args.dataset exists, unwrap it.
    args = ds_spec.get("args", {}) or {}
    base_spec = args.get("dataset", None)
    if base_spec is None:
        # No wrapper; use the dataset spec directly.
        base_spec = {"name": ds_spec.get("name"), "args": args}

    ds = lhnef_datasets.make(base_spec)

    out = Path(os.path.expanduser(os.path.expandvars(out_dir))).resolve()
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for i in range(len(ds)):
        if limit is not None and n >= int(limit):
            break
        item = ds[i]
        if isinstance(item, dict):
            im = item.get("image", None)
        else:
            im = item
        if not isinstance(im, Image.Image):
            raise TypeError(
                f"Expected dataset to yield PIL.Image (or dict{{'image': PIL}}), got {type(im)} at idx={i} "
                f"from {base_spec.get('name')}"
            )
        im = im.convert("RGB")
        im = _resize_to_square(im, size)
        _save_png(im, out / f"{n:06d}.png")
        n += 1
    return n


def export_from_lhnef_ckpt(*, ckpt_path: str, split: str, out_dir: str, size: int | None, limit: int | None) -> int:
    """
    Export a "real images" directory using the dataset spec saved inside an LH-NeF checkpoint.

    This is the most reliable way to match a specific run: it reads ckpt['cfg']['datasets'][split]
    (which includes val_seed/val_fraction/etc), so the exported subset matches exactly.
    """
    # Make repo importable (so `import datasets` works when running as a script).
    repo_root = Path(__file__).resolve().parents[2]
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import torch
    import datasets as lhnef_datasets  # type: ignore

    ckpt_p = Path(os.path.expanduser(os.path.expandvars(ckpt_path))).resolve()
    ckpt = torch.load(str(ckpt_p), map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg", {}) or {}

    split = str(split).lower().strip()
    if split not in ("train", "val", "test"):
        raise ValueError("split must be one of: train/val/test")

    ds_spec = (cfg.get("datasets", {}) or {}).get(split, None)
    if ds_spec is None:
        raise KeyError(f"checkpoint cfg has no datasets.{split}")

    args = ds_spec.get("args", {}) or {}
    base_spec = args.get("dataset", None)
    if base_spec is None:
        base_spec = {"name": ds_spec.get("name"), "args": args}

    ds = lhnef_datasets.make(base_spec)

    out = Path(os.path.expanduser(os.path.expandvars(out_dir))).resolve()
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for i in range(len(ds)):
        if limit is not None and n >= int(limit):
            break
        item = ds[i]
        if isinstance(item, dict):
            im = item.get("image", None)
        else:
            im = item
        if not isinstance(im, Image.Image):
            raise TypeError(
                f"Expected dataset to yield PIL.Image (or dict{{'image': PIL}}), got {type(im)} at idx={i} "
                f"from {base_spec.get('name')}"
            )
        im = im.convert("RGB")
        im = _resize_to_square(im, size)
        _save_png(im, out / f"{n:06d}.png")
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    # NOTE: With argparse + subcommands, "global" args typically must appear *before* the subcommand.
    # Users naturally type: `script subcmd --out_dir ...`. To support that, we also add these args
    # to each subparser below. Keep them optional here for backward compatibility.
    p.add_argument("--out_dir", required=False, help="Output directory for real PNGs.")
    p.add_argument("--limit", type=int, default=None, help="Optional limit on number of images to export.")

    sub = p.add_subparsers(dest="mode", required=True)

    p_folder = sub.add_parser("folder", help="Copy/resize from an existing image folder.")
    p_folder.add_argument("--src_dir", required=True, help="Folder containing real images (any nesting).")
    p_folder.add_argument("--size", type=int, default=None, help="Optional resize to SxS (e.g. 32 or 64).")
    p_folder.add_argument("--out_dir", required=True, help="Output directory for real PNGs.")
    p_folder.add_argument("--limit", type=int, default=None, help="Optional limit on number of images to export.")

    p_cifar = sub.add_parser("cifar10", help="Export CIFAR-10 from a local torchvision dataset folder.")
    p_cifar.add_argument(
        "--root",
        required=True,
        help=f"Path where torchvision expects `{_CIFAR10_BASE_FOLDER}` (or pass --download).",
    )
    p_cifar.add_argument("--split", default="train", choices=["train", "test"])
    p_cifar.add_argument("--download", action="store_true", help="Download CIFAR-10 into --root if missing.")
    p_cifar.add_argument("--out_dir", required=True, help="Output directory for real PNGs.")
    p_cifar.add_argument("--limit", type=int, default=None, help="Optional limit on number of images to export.")

    p_cfg = sub.add_parser("lhnef_cfg", help="Export the exact split defined by an LH-NeF YAML config (no leakage).")
    p_cfg.add_argument("--cfg", required=True, help="Path to the YAML config used for the run.")
    p_cfg.add_argument("--split", default="train", choices=["train", "val", "test"])
    p_cfg.add_argument("--size", type=int, default=None, help="Optional resize to SxS (e.g. 32 or 64).")
    p_cfg.add_argument("--out_dir", required=True, help="Output directory for real PNGs.")
    p_cfg.add_argument("--limit", type=int, default=None, help="Optional limit on number of images to export.")

    p_ckpt = sub.add_parser("lhnef_ckpt", help="Export the exact split defined by an LH-NeF checkpoint (safest).")
    p_ckpt.add_argument("--ckpt", required=True, help="Path to an LH-NeF checkpoint (e.g. stage-1 best-model.pth).")
    p_ckpt.add_argument("--split", default="train", choices=["train", "val", "test"])
    p_ckpt.add_argument("--size", type=int, default=None, help="Optional resize to SxS (e.g. 32 or 64).")
    p_ckpt.add_argument("--out_dir", required=True, help="Output directory for real PNGs.")
    p_ckpt.add_argument("--limit", type=int, default=None, help="Optional limit on number of images to export.")

    args = p.parse_args()

    if getattr(args, "out_dir", None) is None:
        raise SystemExit("error: --out_dir is required")

    if args.mode == "folder":
        n = export_from_folder(src_dir=args.src_dir, out_dir=args.out_dir, size=args.size, limit=args.limit)
    elif args.mode == "cifar10":
        n = export_cifar10(root=args.root, split=args.split, out_dir=args.out_dir, limit=args.limit, download=args.download)
    elif args.mode == "lhnef_cfg":
        # lhnef_cfg
        n = export_from_lhnef_cfg(cfg_path=args.cfg, split=args.split, out_dir=args.out_dir, size=args.size, limit=args.limit)
    else:
        # lhnef_ckpt
        n = export_from_lhnef_ckpt(ckpt_path=args.ckpt, split=args.split, out_dir=args.out_dir, size=args.size, limit=args.limit)

    print(f"[make_real_image_dir] wrote {n} images to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()

