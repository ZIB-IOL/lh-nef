import os, sys, platform, importlib.util, hashlib
from PIL import Image
from torch.utils.data import Dataset
import numpy as np  # for optional train/val split inside the train directory

from datasets import register


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP')


def _resolve_imagenet_root(root_path, class_list=None, dataset_name=None, permanent_root=None):
    """If root_path is 'auto' or None, resolve via the cluster caching module; else passthrough."""
    if root_path and str(root_path).lower() != 'auto':
        return root_path

    ds_name = (
        str(dataset_name).strip()
        if dataset_name is not None and str(dataset_name).strip()
        else os.environ.get("LHNEF_IMAGENET_DATASET_NAME", os.environ.get("INFD_IMAGENET_DATASET_NAME", "")).strip()
    )

    perm_root = (
        str(permanent_root).strip()
        if permanent_root is not None and str(permanent_root).strip()
        else os.environ.get("LHNEF_IMAGENET_PERMANENT_ROOT", os.environ.get("INFD_IMAGENET_PERMANENT_ROOT", "")).strip()
    )
    if not perm_root:
        perm_root = None

    # If a class subset is given, pass it to the caching layer via env vars and a stable alias.
    if class_list is not None:
        try:
            cls_file = os.path.abspath(os.path.expanduser(os.path.expandvars(class_list)))
            with open(cls_file, 'r') as f:
                wanted = sorted({ln.strip() for ln in f if ln.strip()})
            if len(wanted) > 0:
                sig = hashlib.sha1("\n".join(wanted).encode("utf-8")).hexdigest()[:8]
                alias = f"imagenet-in100-{sig}"
                os.environ.setdefault("Z1_VISION_SUBSET_FILE", cls_file)
                os.environ.setdefault("Z1_IMAGENET_CLASS_LIST", cls_file)
                os.environ.setdefault("Z1_VISION_DATASET_ALIAS", alias)
        except Exception:
            pass

    import getpass
    if 'gcp' not in platform.uname().node:
        sys.path.append('/software/ais2t/bin/z1-dataset-caching')
        spec = importlib.util.spec_from_file_location(
            'Caching', '/software/ais2t/bin/z1-dataset-caching/Caching/__init__.py'
        )
    else:
        username = getpass.getuser()
        sys.path.append(f'/scratch/htc/{username}/bin/z1-dataset-caching')
        spec = importlib.util.spec_from_file_location(
            'Caching', f'/scratch/htc/{username}/bin/z1-dataset-caching/Caching/__init__.py'
        )
    Caching = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Caching)
    try:
        if perm_root is not None:
            return Caching.get_dataset_root(ds_name, permanent_path=perm_root)
    except TypeError:
        # Older caching modules may not support permanent_path kwarg.
        pass
    return Caching.get_dataset_root(ds_name)


@register('imagenet_labeled')
class ImageNetLabeled(Dataset):
    """
    ImageNet (train/val) with optional class subset; center square-crops to satisfy WrapperCAE.
    Supports a reproducible train/val split inside the train directory via
    (train_val_split, train_val_fraction, train_val_seed).
    """
    def __init__(
        self,
        split: str = "train",
        root_path: str = "auto",
        class_list=None,
        return_label: bool = True,
        dataset_name: str | None = None,
        permanent_root: str | None = None,
        min_side_floor: int | None = None,
        train_val_split: str | None = None,
        train_val_fraction: float | None = None,
        train_val_seed: int = 0,
        subset_fraction: float | None = None,
        subset_seed: int = 0,
    ):
        if split not in {'train', 'val'}:
            raise ValueError("split must be 'train' or 'val'")

        if train_val_split is not None and split != 'train':
            raise ValueError("train_val_split can only be used when split='train'")

        root = _resolve_imagenet_root(
            root_path,
            class_list=class_list,
            dataset_name=dataset_name,
            permanent_root=permanent_root,
        )
        self.root_split = os.path.join(root, split)

        all_classes = [d for d in os.listdir(self.root_split)
                       if os.path.isdir(os.path.join(self.root_split, d))]
        all_classes = sorted(all_classes)

        # optional IN-100 restriction
        if class_list is not None:
            with open(class_list, 'r') as f:
                wanted = {ln.strip() for ln in f if ln.strip()}
            classes = [c for c in all_classes if c in wanted]
        else:
            classes = all_classes

        if len(classes) == 0:
            raise RuntimeError(f"No classes found under {self.root_split}")

        self.class_to_idx = {c: i for i, c in enumerate(sorted(classes))}

        files = []
        for c in classes:
            cdir = os.path.join(self.root_split, c)
            for fn in os.listdir(cdir):
                if fn.endswith(IMAGE_EXTS):
                    files.append((os.path.join(cdir, fn), self.class_to_idx[c]))
        if len(files) == 0:
            raise RuntimeError(f"No images found under {self.root_split}")

        # Optional reproducible train/val split *inside* the train directory
        if train_val_split is not None:
            split_kind = str(train_val_split).lower()
            if split_kind not in ('train', 'val'):
                raise ValueError(f"train_val_split must be 'train' or 'val', got {train_val_split!r}")
            if train_val_fraction is None:
                raise ValueError("train_val_fraction must be set when using train_val_split")
            frac = float(train_val_fraction)
            if not (0.0 < frac < 1.0):
                raise ValueError(f"train_val_fraction must be in (0, 1), got {frac}")

            rng = np.random.RandomState(int(train_val_seed))
            indices = np.arange(len(files))
            rng.shuffle(indices)

            n_train = int(round(frac * len(files)))
            if n_train <= 0 or n_train >= len(files):
                raise RuntimeError(
                    f"train_val_fraction={frac} and len(files)={len(files)} produced an empty split"
                )

            if split_kind == 'train':
                sel = indices[:n_train]
            else:
                sel = indices[n_train:]

            files = [files[i] for i in sel]
            if len(files) == 0:
                raise RuntimeError(f"Split '{split_kind}' resulted in 0 images under {self.root_split}")

        self.files = files

        # Optional subsampling of the already-split partition.
        self.indices = np.arange(len(self.files))
        if subset_fraction is not None:
            frac_sub = float(subset_fraction)
            if not (0.0 < frac_sub <= 1.0):
                raise ValueError(f"subset_fraction must be in (0, 1], got {frac_sub}")
            if frac_sub < 1.0:
                rng_sub = np.random.RandomState(int(subset_seed))
                n_keep = max(1, int(round(frac_sub * len(self.indices))))
                sel = rng_sub.choice(self.indices, size=n_keep, replace=False)
                self.indices = np.array(sel, dtype=int)

        self.return_label = bool(return_label)
        self.min_side_floor = min_side_floor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        path, label = self.files[real_idx]
        img = Image.open(path).convert('RGB')

        # center square-crop to satisfy WrapperCAE
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

        # If original side < floor, upsample so WrapperCAE's GT sampling stays valid.
        if self.min_side_floor is not None and side < int(self.min_side_floor):
            tgt = int(self.min_side_floor)
            img = img.resize((tgt, tgt), Image.BICUBIC)

        if self.return_label:
            return {"image": img, "label": int(label)}
        return img


