"""FFCV-backed ImageNet 256x256 dataloader, in coord-value format.

Equivalent to ``wrapper_cae_coord_value`` at the 256^2 configuration:
  resize_inp = resize_gt_lb = resize_gt_ub = final_crop_gt = 256,
  p_whole = p_max = 0.

Requires a baked .beton produced by ``src/tools/bake_imagenet_ffcv.py``.).
"""
from __future__ import annotations

# Import cv2 first to claim the conda-forge libjpeg with jpeg12 symbols
# (otherwise PIL's libjpeg loads first and libtiff fails on import).
import cv2  # noqa: F401

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import register
from utils.geometry import make_coord_grid


# (uint8 / 255 - 0.5) / 0.5  == (uint8 - 127.5) / 127.5
_NORM_MEAN = np.array([127.5, 127.5, 127.5])
_NORM_STD = np.array([127.5, 127.5, 127.5])


def _build_ffcv_pipelines(device: torch.device | None = None) -> dict:
    """FFCV pipelines that produce a float32 image in [-1, 1] CHW + integer label.
    """
    from ffcv.fields.decoders import SimpleRGBImageDecoder, IntDecoder
    from ffcv.transforms import ToTensor, ToTorchImage, NormalizeImage, ToDevice, Squeeze

    image_pipe = [
        SimpleRGBImageDecoder(),
        NormalizeImage(_NORM_MEAN, _NORM_STD, np.float32),
        ToTensor(),
        ToTorchImage(),
    ]
    label_pipe = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
    ]
    if device is not None:
        image_pipe.append(ToDevice(device, non_blocking=True))
        label_pipe.append(ToDevice(device, non_blocking=True))
    return {'image': image_pipe, 'label': label_pipe}


class FFCVBatchAdapter:
    """Wrap an FFCV ``Loader`` to yield dict batches in the trainer's coord-value contract."""

    def __init__(self, ffcv_loader, coord_flat: torch.Tensor, cell_flat: torch.Tensor,
                 gt_n_query: int | None = None, gt_replace: bool = False):
        self.ffcv_loader = ffcv_loader
        self.coord_flat = coord_flat   # [N=65536, 2] in [-1, 1], CPU
        self.cell_flat = cell_flat     # [N=65536, 2], CPU
        self.gt_n_query = int(gt_n_query) if gt_n_query else None
        self.gt_replace = bool(gt_replace)
        self._coord_dev = None
        self._cell_dev = None

    def __len__(self):
        return len(self.ffcv_loader)

    @property
    def batch_size(self):
        return self.ffcv_loader.batch_size

    def __iter__(self):
        for img, label in self.ffcv_loader:
            B = int(img.shape[0])
            # Lazy-cache coord/cell on the same device as the batch.
            if self._coord_dev is None or self._coord_dev.device != img.device:
                self._coord_dev = self.coord_flat.to(img.device)
                self._cell_dev = self.cell_flat.to(img.device)
            # [B, 3, 256, 256] -> [B, 65536, 3]
            value = img.permute(0, 2, 3, 1).reshape(B, -1, 3).contiguous()
            coord = self._coord_dev.unsqueeze(0).expand(B, -1, -1)
            cell = self._cell_dev.unsqueeze(0).expand(B, -1, -1)

            N = value.shape[1]
            if self.gt_n_query is not None and 0 < self.gt_n_query < N:
                # Per-sample random subsampling (independent across the batch).
                if self.gt_replace:
                    sel = torch.randint(0, N, (B, self.gt_n_query), device=img.device)
                else:
                    sel = torch.argsort(torch.rand(B, N, device=img.device), dim=1)[:, :self.gt_n_query]
                sel3 = sel.unsqueeze(-1).expand(-1, -1, 3)
                sel2 = sel.unsqueeze(-1).expand(-1, -1, 2)
                gt = torch.gather(value, 1, sel3)
                gt_coord = torch.gather(coord, 1, sel2)
                gt_cell = torch.gather(cell, 1, sel2)
                gt_is_grid = torch.zeros(B, dtype=torch.int64, device=img.device)
            else:
                gt = value
                gt_coord = coord
                gt_cell = cell
                gt_is_grid = torch.ones(B, dtype=torch.int64, device=img.device)

            yield {
                'inp': {'coord': coord, 'value': value},
                'inp_coord': coord,
                'inp_value': value,
                'gt': gt,
                'gt_coord': gt_coord,
                'gt_cell': gt_cell,
                'gt_is_grid': gt_is_grid,
                'gt_grid_shape': torch.tensor([[256, 256]] * B, dtype=torch.int64, device=img.device),
                'label': label,
            }


class FFCVSamplerStub:
    """Sampler-shaped object the trainer can call .set_epoch on. FFCV's RANDOM
    traversal already reshuffles each iter() so this is a no-op."""
    def set_epoch(self, epoch: int) -> None:  # noqa: D401
        return None


@register('wrapper_ffcv_imagenet256')
class WrapperFFCVImageNet256(Dataset):
    """FFCV-backed drop-in for ``wrapper_cae_coord_value`` on ImageNet 256^2.

    The trainer's ``make_datasets`` detects ``is_ffcv=True`` and calls
    ``make_loader`` directly, bypassing PyTorch's DataLoader wrapping.
    """

    is_ffcv = True

    def __init__(self, beton_path: str,
                 gt_n_query: int | None = None,
                 gt_replace: bool = False,
                 subset_n: int | None = None,
                 subset_seed: int = 0,
                 subset_select: str = "random",
                 subset_pool_n: int = 4000,
                 **_legacy_kwargs):
        # `**_legacy_kwargs` silently absorbs deprecated/unused dataset args
        # so cfgs that bundle e.g. dataset spec for the old wrapper can be
        # swapped in-place without errors.
        from ffcv.reader import Reader

        self.beton_path = str(beton_path)
        if not str(beton_path):
            raise ValueError("wrapper_ffcv_imagenet256: beton_path is required")
        if gt_n_query is not None and int(gt_n_query) <= 0:
            gt_n_query = None
        self.gt_n_query = int(gt_n_query) if gt_n_query else None
        self.gt_replace = bool(gt_replace)

        reader = Reader(self.beton_path)
        self._n = int(reader.num_samples)
        del reader

        self.subset_seed = int(subset_seed)
        self.subset_select = str(subset_select).lower()
        self.subset_pool_n = int(subset_pool_n)
        self._subset_indices = None
        if subset_n is not None and int(subset_n) > 0:
            n = min(int(subset_n), self._n)
            if self.subset_select == "hifreq":
                self._subset_indices = self._rank_hifreq_subset(n)
            else:
                rng = np.random.default_rng(self.subset_seed)
                self._subset_indices = np.sort(
                    rng.choice(self._n, size=n, replace=False)).astype(np.int64)

        # Deterministic coord grid + cell at 256x256. Held on CPU; replicated
        # to the batch device lazily in the adapter.
        coord_2d = make_coord_grid((256, 256))            # [256, 256, 2]
        self.coord_flat = coord_2d.reshape(-1, 2).contiguous().float()  # [65536, 2]
        cell_one = torch.tensor([2.0 / 256, 2.0 / 256], dtype=torch.float32)
        self.cell_flat = cell_one.view(1, 2).expand(256 * 256, 2).contiguous()

    def _rank_hifreq_subset(self, n: int) -> np.ndarray:
        """Pick the `n` highest-frequency images from a seeded candidate pool.
        """
        from pathlib import Path

        pool_n = min(self.subset_pool_n, self._n)
        cache = (Path(self.beton_path).parent
                 / f".hifreq_rank_seed{self.subset_seed}_pool{pool_n}.npz")
        if cache.exists():
            d = np.load(cache)
            pool, scores = d["pool"], d["scores"]
        else:
            rng = np.random.default_rng(self.subset_seed)
            pool = np.sort(rng.choice(self._n, size=pool_n, replace=False)).astype(np.int64)
            scores = self._hifreq_scores_for(pool)
            try:
                np.savez(cache, pool=pool, scores=scores)
            except OSError:
                pass  # read-only fs: just skip caching

        order = np.argsort(scores)[::-1]          # high-freq first
        top = pool[order[:n]]
        print(f"[hifreq] picked top-{n}/{pool_n} by HF energy; "
              f"score range [{scores[order[0]]:.4f}, {scores[order[n-1]]:.4f}]; "
              f"indices={np.sort(top).tolist()}")
        return np.sort(top).astype(np.int64)

    def _hifreq_scores_for(self, indices: np.ndarray) -> np.ndarray:
        """Fraction of luminance spectral power above half-Nyquist, per image."""
        from ffcv.loader import Loader, OrderOption

        loader = Loader(self.beton_path, batch_size=128, num_workers=8,
                        order=OrderOption.SEQUENTIAL, drop_last=False,
                        indices=np.asarray(indices, dtype=np.int64),
                        pipelines=_build_ffcv_pipelines(device=None))
        # Radial high-pass mask on the 256x256 grid (built once).
        fy = np.fft.fftshift(np.fft.fftfreq(256))[:, None]
        fx = np.fft.fftshift(np.fft.fftfreq(256))[None, :]
        hi = (np.sqrt(fy ** 2 + fx ** 2) / 0.5) > 0.5      # >half-Nyquist
        scores = []
        for img, _ in loader:
            x = img.numpy()                                # [B,3,256,256] in [-1,1]
            lum = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
            F = np.fft.fftshift(np.fft.fft2(lum, axes=(-2, -1)), axes=(-2, -1))
            power = F.real ** 2 + F.imag ** 2              # [B,256,256]
            total = power.sum(axis=(-2, -1)) + 1e-12
            high = (power * hi[None]).sum(axis=(-2, -1))
            scores.append((high / total).astype(np.float64))
        return np.concatenate(scores)

    def __len__(self):
        # Reflect the active subset so trainer logs / epoch math see the true count.
        if self._subset_indices is not None:
            return int(self._subset_indices.size)
        return self._n

    def __getitem__(self, i):
        """Slow single-sample decode for the trainer's sanity-check log line.
        Not used in training; ``make_loader`` returns the fast iterator."""
        from ffcv.loader import Loader, OrderOption

        # No ToDevice here: batch_size=1 with Squeeze+ToDevice triggers an FFCV IndexError
        loader = Loader(self.beton_path,
                        batch_size=1, num_workers=1,
                        order=OrderOption.SEQUENTIAL, drop_last=False,
                        indices=np.array([int(i)], dtype=np.int64),
                        pipelines=_build_ffcv_pipelines(device=None))
        img, label = next(iter(loader))
        img = img[0]  # [3, 256, 256]
        value = img.permute(1, 2, 0).reshape(-1, 3).contiguous()  # [65536, 3]
        return {
            'inp': {'coord': self.coord_flat, 'value': value},
            'gt': value,
            'gt_coord': self.coord_flat,
            'gt_cell': self.cell_flat,
            'gt_is_grid': torch.tensor(1, dtype=torch.int64),
            'gt_grid_shape': torch.tensor([256, 256], dtype=torch.int64),
            'label': torch.tensor(int(label.item()), dtype=torch.long),
        }

    def make_loader(self, *, batch_size: int, num_workers: int, distributed: bool,
                    shuffle: bool, drop_last: bool, device: torch.device,
                    world_size: int = 1):
        """Build the FFCV-backed batch iterator. Returns (loader, sampler_stub).

        Replaces ``BaseTrainer.make_distributed_loader`` for this dataset.
        """
        from ffcv.loader import Loader, OrderOption

        if distributed:
            if batch_size % world_size != 0:
                raise ValueError(
                    f"FFCV: batch_size={batch_size} not divisible by world_size={world_size}"
                )
            per_rank_bs = batch_size // world_size
        else:
            per_rank_bs = batch_size

        order = OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL

        loader_kwargs = dict(
            batch_size=per_rank_bs,
            num_workers=num_workers,
            order=order,
            os_cache=True,
            drop_last=drop_last,
            distributed=bool(distributed),
            pipelines=_build_ffcv_pipelines(device),
        )
        if self._subset_indices is not None:
            # Restrict the epoch to a fixed deterministic subset
            loader_kwargs['indices'] = self._subset_indices
            loader_kwargs['seed'] = int(self.subset_seed)
        ffcv_loader = Loader(self.beton_path, **loader_kwargs)

        adapter = FFCVBatchAdapter(
            ffcv_loader,
            coord_flat=self.coord_flat,
            cell_flat=self.cell_flat,
            gt_n_query=self.gt_n_query,
            gt_replace=self.gt_replace,
        )
        return adapter, FFCVSamplerStub()

    def make_subset_loader(self, indices, *, batch_size: int, num_workers: int = 0,
                           device: torch.device | None = None):
        """Build an FFCV loader over a fixed subset of sample indices.

        Used by the trainer's side paths (``train_psnr_proxy``, ``eval_subset``,
        ``visualize_subset``) instead of ``Subset(ds, idx) + DataLoader(...)``,
        which would invoke ``__getitem__`` one sample at a time and rebuild a
        Loader on each call (too slow on the 237 GB train.beton).
        """
        from ffcv.loader import Loader, OrderOption

        idx_arr = np.asarray(list(indices), dtype=np.int64)
        if idx_arr.size == 0:
            raise ValueError("make_subset_loader: empty indices")

        # Clamp to >=1: numba.set_num_threads (called by FFCV's Loader init)
        # rejects 0. The trainer's subset paths typically pass num_workers=0.
        nw = max(1, int(num_workers))
        ffcv_loader = Loader(
            self.beton_path,
            batch_size=int(batch_size),
            num_workers=nw,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            distributed=False,
            indices=idx_arr,
            pipelines=_build_ffcv_pipelines(device),
        )
        return FFCVBatchAdapter(
            ffcv_loader,
            coord_flat=self.coord_flat,
            cell_flat=self.cell_flat,
            gt_n_query=self.gt_n_query,
            gt_replace=self.gt_replace,
        )
