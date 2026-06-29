import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from datasets import register


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP')


@register('celebahq')
class CelebAHQ(Dataset):
    """
    CelebA-HQ image dataset loader (no labels).

    Layouts supported:
      A) root_path/{train,val}/...
      B) root_path/images/... (or via img_folder); split deterministically using val_fraction + seed.

    Returns a PIL.Image (RGB), square-cropped (WrapperCAE assumes square input).
    """

    def __init__(
        self,
        split: str = 'train',
        root_path: str = 'load/celebahq',
        img_folder: str | None = None,
        val_fraction: float = 0.1,
        test_fraction: float = 0.0,
        seed: int = 0,
        min_side_floor: int | None = None,
    ):
        if split not in {'train', 'val', 'test'}:
            raise ValueError("split must be 'train', 'val', or 'test'")

        root_path = os.path.expanduser(os.path.expandvars(str(root_path)))

        # Prefer root_path/{train,val,test} if present; else (root_path/img_folder) or root_path.
        split_dir = os.path.join(root_path, split)
        if os.path.isdir(split_dir):
            img_root = split_dir
            files = self._scan_images(img_root)
        else:
            img_root = os.path.join(root_path, img_folder) if img_folder is not None else root_path
            files_all = self._scan_images(img_root)
            if len(files_all) == 0:
                raise RuntimeError(f"No images found under {img_root}")

            frac = float(val_fraction)
            if not (0.0 < frac < 1.0):
                raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")

            tfrac = float(test_fraction)
            if not (0.0 <= tfrac < 1.0):
                raise ValueError(f"test_fraction must be in [0,1), got {test_fraction}")
            if (frac + tfrac) >= 1.0:
                raise ValueError(f"val_fraction + test_fraction must be < 1, got {frac + tfrac}")

            rng = np.random.RandomState(int(seed))
            idx = np.arange(len(files_all))
            rng.shuffle(idx)
            n_val = int(round(frac * len(idx)))
            n_test = int(round(tfrac * len(idx)))
            if n_val == 0 and frac > 0:
                n_val = 1
            if n_test == 0 and tfrac > 0:
                n_test = 1
            n_val = min(n_val, len(idx) - 1)
            n_test = min(n_test, len(idx) - 1 - n_val)

            if split == 'val':
                sel = idx[:n_val]
            elif split == 'test':
                sel = idx[n_val:n_val + n_test]
            else:
                sel = idx[n_val + n_test:]
            files = [files_all[int(i)] for i in sel]

        if len(files) == 0:
            raise RuntimeError(f"No images found for split='{split}' under {img_root}")

        self.files = files
        self.min_side_floor = None if min_side_floor is None else int(min_side_floor)

    @staticmethod
    def _scan_images(root: str):
        out = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(IMAGE_EXTS):
                    out.append(os.path.join(dirpath, fn))
        out = sorted(out)
        return out

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[int(idx)]).convert('RGB')

        # Center square-crop (WrapperCAE assumes square input).
        w, h = img.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        img = img.crop((left, top, left + side, top + side))

        if self.min_side_floor is not None and side < int(self.min_side_floor):
            tgt = int(self.min_side_floor)
            img = img.resize((tgt, tgt), Image.BICUBIC)

        return img

