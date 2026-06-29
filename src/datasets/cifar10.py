import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import torch

from datasets import register


@register('cifar10')
class CIFAR10Dataset(Dataset):

    def __init__(self, split='train', root_path='load/cifar10', download=True,
                 val_size: int | None = None, val_fraction: float | None = 0.1,
                 val_seed: int = 42, return_label: bool = False):
        try:
            from torchvision.datasets import CIFAR10
        except Exception as e:
            raise ImportError('torchvision is required to use CIFAR10Dataset') from e

        allowed = {'train', 'val', 'test'}
        if split not in allowed:
            raise ValueError(f"CIFAR10Dataset: unsupported split '{split}'. Allowed: {sorted(allowed)}")

        if split in {'train', 'val'}:
            self.ds = CIFAR10(root=root_path, train=True, download=download)
            num_train = len(self.ds)
            if val_fraction is not None and 0 < float(val_fraction) < 1:
                vs = int(round(num_train * float(val_fraction)))
            else:
                vs = int(val_size) if val_size is not None else 5000
            vs = max(1, min(vs, num_train - 1))
            rng = np.random.RandomState(int(val_seed))
            perm = rng.permutation(num_train)
            val_idx = perm[:vs]
            train_idx = perm[vs:]
            if split == 'train':
                self.indices = train_idx.tolist()
            else:
                self.indices = val_idx.tolist()
        else:
            self.ds = CIFAR10(root=root_path, train=False, download=download)
            self.indices = None
        
        self.return_label = bool(return_label)

    def __len__(self):
        return len(self.ds) if self.indices is None else len(self.indices)

    def __getitem__(self, idx):
        real_idx = idx if self.indices is None else self.indices[idx]
        img, label = self.ds[real_idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if self.return_label:
            return {"image": img, "label": int(label)}
        return img


