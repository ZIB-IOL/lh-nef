"""
Dataset wrappers:
  - ``wrapper_cae_coord_value``: image dataset -> coord/value tokens (inp={coord:[N,2], value:[N,3]}).
  - ``wrapper_coord_value``: passthrough for datasets that already emit the coord/value contract.
"""

from __future__ import annotations

import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import datasets
from datasets import register
from utils.geometry import make_coord_grid


class BaseWrapperCAE:

    def __init__(self, dataset, resize_inp, ret_gt=True, resize_gt_lb=None, resize_gt_ub=None,
                 final_crop_gt=None, p_whole=0.0, p_max=0.0, inp_aug=None,
                 **_legacy_kwargs):
        # `**_legacy_kwargs` silently absorbs deprecated dataset args from old saved cfg.yaml files.
        self.dataset = datasets.make(dataset)
        self.resize_inp = resize_inp
        self.ret_gt = ret_gt
        self.resize_gt_lb = resize_gt_lb
        self.resize_gt_ub = resize_gt_ub
        self.final_crop_gt = final_crop_gt
        self.p_whole = p_whole
        self.p_max = p_max
        self.inp_aug = inp_aug or {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
        self.inp_aug_transform = self._build_inp_aug_transform()

    def _build_inp_aug_transform(self):
        """Augmentations applied to the encoder input image before conversion to tensor."""
        cfg = (self.inp_aug or {})
        if not bool(cfg.get('enable', False)):
            return None
        kind = str(cfg.get('kind', 'cifar10')).lower().strip()
        if kind == 'cifar10':
            pad = int(cfg.get('pad', 4))
            crop = int(cfg.get('crop', 32))
            flip_p = float(cfg.get('flip_p', 0.5))
            ops = [
                transforms.RandomCrop(crop, padding=pad, padding_mode='constant'),
                transforms.RandomHorizontalFlip(p=flip_p),
            ]
            return transforms.Compose(ops)

        if kind == 'imagenet_randaugment':
            out = int(cfg.get('size', self.resize_inp))
            scale = cfg.get('scale', [0.08, 1.0])
            ratio = cfg.get('ratio', [3 / 4, 4 / 3])
            n_ops = int(cfg.get('num_ops', 2))
            mag = int(cfg.get('magnitude', 5))
            ops = [
                transforms.RandomResizedCrop(out, scale=tuple(scale), ratio=tuple(ratio), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=float(cfg.get('flip_p', 0.5))),
            ]
            if hasattr(transforms, 'RandAugment'):
                ops.append(transforms.RandAugment(num_ops=n_ops, magnitude=mag))
            return transforms.Compose(ops)

        raise ValueError(f"Unknown inp_aug.kind={kind!r}")

    def process(self, sample):
        # Accept either raw PIL.Image or a dict {'image': PIL.Image, 'label': int}
        if isinstance(sample, dict):
            img = sample['image']
            label = int(sample.get('label', -1))
        else:
            img = sample
            label = None

        assert img.size[0] == img.size[1]
        ret = {}

        inp_img = img
        if self.inp_aug_transform is not None:
            inp_img = self.inp_aug_transform(inp_img)
        inp = inp_img.resize((self.resize_inp, self.resize_inp), Image.LANCZOS)
        inp = self.transform(inp)
        ret.update({'inp': inp})
        if label is not None:
            ret['label'] = torch.tensor(label, dtype=torch.long)

        if self.ret_gt:
            if self.resize_gt_lb is None:
                gt = self.transform(img)
            else:
                if random.random() < self.p_whole:
                    r = self.final_crop_gt
                elif random.random() < self.p_max:
                    r = min(img.size[0], self.resize_gt_ub)
                else:
                    r = random.randint(self.resize_gt_lb, min(img.size[0], self.resize_gt_ub))
                gt = img.resize((r, r), Image.LANCZOS)
                gt = self.transform(gt)

            p = self.final_crop_gt
            ii = random.randint(0, gt.shape[-2] - p)
            jj = random.randint(0, gt.shape[-1] - p)
            gt_patch = gt[:, ii: ii + p, jj: jj + p]

            x0, y0 = ii / gt.shape[-2], jj / gt.shape[-1]  # assume range [0, 1]
            x1, y1 = (ii + p) / gt.shape[-2], (jj + p) / gt.shape[-1]
            coord = make_coord_grid((p, p), range=[[x0, x1], [y0, y1]])
            coord = 2 * coord - 1  # convert to range [-1, 1]
            cell = torch.tensor([2 / gt.shape[-2], 2 / gt.shape[-1]], dtype=torch.float32)
            cell = cell.view(1, 1, 2).expand(p, p, -1)
            ret.update({
                'gt': gt_patch,       # 3 p p
                'gt_coord': coord,    # p p 2
                'gt_cell': cell,      # p p 2
            })

        return ret


@register('wrapper_cae_coord_value')
class WrapperCAECoordValue(BaseWrapperCAE, Dataset):
    """
    Image dataset -> coord/value tokens.
    Output: inp={'coord':[N,2],'value':[N,3]}; gt_coord:[Q,2]; gt_cell:[Q,2]; gt:[Q,3].
    """

    def __init__(self, *args, n_inp: int | None = None, gt_n_query: int | None = None,
                 gt_replace: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_inp = None if n_inp is None or int(n_inp) <= 0 else int(n_inp)
        self.gt_n_query = None if gt_n_query is None else int(gt_n_query)
        self.gt_replace = bool(gt_replace)
        # If True, always return full-grid GT queries (useful for train-PSNR proxy evaluation).
        self.force_full_gt = False

    def set_force_full_gt(self, enabled: bool = True):
        self.force_full_gt = bool(enabled)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # process() returns image-shaped: inp [3,H,W], gt [3,Hq,Wq], gt_coord [Hq,Wq,2], gt_cell [Hq,Wq,2]
        out = self.process(self.dataset[idx])

        inp = out["inp"]
        gt = out["gt"]
        gt_coord = out["gt_coord"]
        gt_cell = out["gt_cell"]

        if not torch.is_tensor(inp) or inp.ndim != 3:
            raise ValueError(f"Expected inp as [3,H,W] tensor, got {type(inp)} shape={getattr(inp,'shape',None)}")
        if not torch.is_tensor(gt) or gt.ndim != 3:
            raise ValueError(f"Expected gt as [3,Hq,Wq] tensor, got {type(gt)} shape={getattr(gt,'shape',None)}")

        H, W = int(inp.shape[1]), int(inp.shape[2])
        coord = make_coord_grid((H, W), device=inp.device)  # [H,W,2] (y,x) in [-1,1]
        inp_coord = coord.reshape(-1, 2).to(dtype=torch.float32)
        inp_value = inp.permute(1, 2, 0).reshape(-1, int(inp.shape[0])).contiguous().to(dtype=torch.float32)  # [N,3]

        if self.n_inp is not None and self.n_inp < inp_coord.shape[0]:
            sel = torch.randperm(inp_coord.shape[0], device=inp_coord.device)[:self.n_inp]
            inp_coord = inp_coord[sel]
            inp_value = inp_value[sel]

        Hq, Wq = int(gt.shape[1]), int(gt.shape[2])
        gt_flat = gt.permute(1, 2, 0).reshape(-1, int(gt.shape[0])).contiguous().to(dtype=torch.float32)  # [Q,3]
        gt_coord_flat = gt_coord.reshape(-1, 2).contiguous().to(dtype=torch.float32)
        gt_cell_flat = gt_cell.reshape(-1, 2).contiguous().to(dtype=torch.float32)

        # If we subsample queries, we can't treat them as a dense grid for PSNR; mark gt_is_grid=0.
        if (not self.force_full_gt) and self.gt_n_query is not None and self.gt_n_query > 0:
            Q = int(gt_flat.shape[0])
            nq = int(self.gt_n_query)
            if nq < Q:
                if self.gt_replace:
                    sel = torch.randint(low=0, high=Q, size=(nq,), device=gt_flat.device)
                else:
                    sel = torch.randperm(Q, device=gt_flat.device)[:nq]
                gt_flat = gt_flat.index_select(0, sel)
                gt_coord_flat = gt_coord_flat.index_select(0, sel)
                gt_cell_flat = gt_cell_flat.index_select(0, sel)
                out["gt_is_grid"] = torch.tensor(0, dtype=torch.int64)
                out["gt_grid_shape"] = torch.tensor([Hq, Wq], dtype=torch.int64)
            else:
                out["gt_is_grid"] = torch.tensor(1, dtype=torch.int64)
                out["gt_grid_shape"] = torch.tensor([Hq, Wq], dtype=torch.int64)
        else:
            out["gt_is_grid"] = torch.tensor(1, dtype=torch.int64)
            out["gt_grid_shape"] = torch.tensor([Hq, Wq], dtype=torch.int64)

        out["inp"] = {"coord": inp_coord, "value": inp_value}
        out["inp_coord"] = inp_coord
        out["inp_value"] = inp_value
        out["gt"] = gt_flat
        out["gt_coord"] = gt_coord_flat
        out["gt_cell"] = gt_cell_flat
        return out


@register("wrapper_coord_value")
class WrapperCoordValue:
    def __init__(self, dataset, ret_gt: bool = True, **_legacy_kwargs):
        # `**_legacy_kwargs` silently absorbs deprecated args from old saved cfg.yaml files.
        self.dataset = datasets.make(dataset)
        self.ret_gt = bool(ret_gt)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
