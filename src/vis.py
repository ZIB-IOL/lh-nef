from __future__ import annotations

import io
import pathlib
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from utils.geometry import make_coord_cell_grid


@torch.no_grad()
def visualize_reconstructions(model, loader, device, save_dir: pathlib.Path, vis_res: int, max_samples: int = 10, use_amp: bool = True, amp_dtype: torch.dtype = torch.float16):
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    collected = 0
    for batch in loader:
        # Handles nested batches (e.g. inp as dict{'coord','value'}).
        def _to_device(x):
            if torch.is_tensor(x):
                return x.to(device, non_blocking=True)
            if isinstance(x, (list, tuple)):
                return [_to_device(_v) for _v in x]
            if isinstance(x, dict):
                return {kk: _to_device(vv) for kk, vv in x.items()}
            return x
        data = _to_device(batch)

        if not isinstance(data, dict):
            return None, None

        inp = data.get('inp', None)
        gt = data.get('gt', None)

        # Batch size
        B = None
        if torch.is_tensor(inp):
            B = int(inp.shape[0])
        elif torch.is_tensor(gt):
            B = int(gt.shape[0])
        if B is None:
            return None, None

        # Pick reference image: image-shaped gt -> image-shaped inp -> reshape flat gt grid.
        ref_img = None  # [B,3,H,W]
        if torch.is_tensor(gt) and gt.ndim == 4 and int(gt.shape[1]) == 3:
            ref_img = gt
        elif torch.is_tensor(inp) and inp.ndim == 4 and int(inp.shape[1]) == 3:
            ref_img = inp
        else:
            gt_is_grid = data.get('gt_is_grid', None)
            gt_grid_shape = data.get('gt_grid_shape', None)
            if torch.is_tensor(gt) and gt.ndim == 3 and int(gt.shape[-1]) == 3:
                is_grid = False
                if torch.is_tensor(gt_is_grid):
                    try:
                        is_grid = int(gt_is_grid.reshape(-1)[0].item()) == 1
                    except Exception:
                        is_grid = False
                Hq = Wq = None
                if torch.is_tensor(gt_grid_shape):
                    try:
                        if gt_grid_shape.ndim == 1 and int(gt_grid_shape.numel()) == 2:
                            Hq, Wq = int(gt_grid_shape[0].item()), int(gt_grid_shape[1].item())
                        elif gt_grid_shape.ndim == 2 and int(gt_grid_shape.shape[-1]) == 2:
                            Hq, Wq = int(gt_grid_shape[0, 0].item()), int(gt_grid_shape[0, 1].item())
                    except Exception:
                        Hq = Wq = None
                if is_grid and (Hq is not None) and (Wq is not None) and int(Hq * Wq) == int(gt.shape[1]):
                    ref_img = gt.view(B, Hq, Wq, 3).permute(0, 3, 1, 2).contiguous()

        if ref_img is None:
            return None, None

        coord, cell = make_coord_cell_grid((vis_res, vis_res), device=device, bs=B)
        data_pred = {'inp': inp, 'gt_coord': coord, 'gt_cell': cell}
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            pred = model(data_pred, mode='pred')

        def _to_img(pred_any):
            if torch.is_tensor(pred_any):
                if pred_any.ndim == 4 and int(pred_any.shape[1]) == 3:
                    return pred_any
                if pred_any.ndim == 3 and int(pred_any.shape[-1]) == 3 and int(pred_any.shape[1]) == int(vis_res * vis_res):
                    return pred_any.view(B, vis_res, vis_res, 3).permute(0, 3, 1, 2).contiguous()
            return None

        # Support either a single tensor prediction or a dict of named predictions.
        pred_imgs = []
        pred_order_names = None
        if torch.is_tensor(pred):
            pred_img = _to_img(pred)
            if pred_img is None:
                return None, None
            pred_imgs = [("pred", pred_img)]
            pred_order_names = ["pred"]
        elif isinstance(pred, dict):
            # Stable, semantic column ordering when those keys are present.
            order = []
            if "pred_recon" in pred:
                order.append("pred_recon")
            if "pred_base" in pred:
                order.append("pred_base")
            if "pred_base_trainonly" in pred:
                order.append("pred_base_trainonly")
            for k in sorted(pred.keys()):
                if k not in order:
                    order.append(k)
            pred_order_names = list(order)
            for k in order:
                img = _to_img(pred.get(k, None))
                if img is not None:
                    pred_imgs.append((k, img))
            if len(pred_imgs) == 0:
                return None, None
        else:
            return None, None

        if int(ref_img.shape[-1]) != int(vis_res) or int(ref_img.shape[-2]) != int(vis_res):
            ref_img = F.interpolate(ref_img, size=(vis_res, vis_res), mode='bilinear', align_corners=False)

        # Column names only; per-subset PSNR was misleading (use train/val PSNR logs instead).
        try:
            if pred_order_names is None:
                pred_order_names = [n for n, _ in pred_imgs]
            msg = "[visualize] columns: GT | " + " | ".join([str(n) for n in pred_order_names])
            print(msg)
        except Exception:
            pass

        for i in range(B):
            tiles.append(ref_img[i].clamp(-1, 1))
            for _name, _img in pred_imgs:
                tiles.append(_img[i].clamp(-1, 1))
            collected += 1
            if collected >= max_samples:
                break
        if collected >= max_samples:
            break
    if len(tiles) == 0:
        return None, None
    nrow = 1 + int(len(pred_imgs))  # Columns: GT | pred1 | pred2 | ...
    grid = make_grid(torch.stack(tiles, dim=0), nrow=nrow, normalize=True, value_range=(-1, 1))
    img_path = save_dir / 'val_recon.png'
    save_image(grid, img_path)
    return img_path, grid


def _render_voxel_to_tensor(
    occ: np.ndarray,
    title: str = "",
    img_size: int = 256,
    elev: float = 25.0,
    azim: float = -60.0,
) -> torch.Tensor:
    """Render a boolean voxel grid [D,H,W] to a [3,img_size,img_size] RGB tensor via matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 4), dpi=img_size // 4)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(occ.astype(bool), facecolors="steelblue", edgecolor="k", linewidth=0.05, alpha=0.7)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9, pad=-5)
    fig.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=img_size // 4, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)

    from PIL import Image
    img = Image.open(buf).convert("RGB").resize((img_size, img_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]


def _unwrap_to_inner_dataset(ds):
    """Walk through Subset / wrapper layers to find the inner dataset."""
    if hasattr(ds, 'dataset'):
        return _unwrap_to_inner_dataset(ds.dataset)
    return ds


def _load_gt_occ_grid(loader, batch_global_idx: int, vis_res: int) -> Optional[np.ndarray]:
    """Try to load the full GT occupancy grid for a sample from the underlying dataset."""
    try:
        ds = loader.dataset
        inner = _unwrap_to_inner_dataset(ds)
        if not (hasattr(inner, '_load_occ') and hasattr(inner, 'files')):
            return None
        # Resolve actual index through Subset layers.
        idx = batch_global_idx
        cur = loader.dataset
        while hasattr(cur, 'indices'):
            idx = cur.indices[idx]
            cur = cur.dataset
        occ = inner._load_occ(inner.files[idx])
        if occ.shape != (vis_res, vis_res, vis_res):
            f = occ.shape[0] // vis_res
            if f > 1 and occ.shape[0] % vis_res == 0:
                occ = occ.reshape(vis_res, f, vis_res, f, vis_res, f).max(axis=(1, 3, 5))
        return occ.astype(bool)
    except Exception:
        return None


def _gt_from_sparse_samples(
    gt_vals: torch.Tensor, gt_coords: torch.Tensor, vis_res: int, threshold: float
) -> np.ndarray:
    """Reconstruct approximate GT voxel grid from sparse coord/value samples."""
    gt_vox = np.zeros((vis_res, vis_res, vis_res), dtype=bool)
    occ_mask = gt_vals[:, 0] > threshold
    if occ_mask.any():
        cz = ((gt_coords[occ_mask, 0] + 1.0) / 2.0 * vis_res).long().clamp(0, vis_res - 1)
        cy = ((gt_coords[occ_mask, 1] + 1.0) / 2.0 * vis_res).long().clamp(0, vis_res - 1)
        cx = ((gt_coords[occ_mask, 2] + 1.0) / 2.0 * vis_res).long().clamp(0, vis_res - 1)
        gt_vox[cz.numpy(), cy.numpy(), cx.numpy()] = True
    return gt_vox


@torch.no_grad()
def visualize_reconstructions_3d_occ(
    model,
    loader,
    device,
    save_dir: pathlib.Path,
    vis_res: int = 32,
    max_samples: int = 4,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    occ_threshold: float = 0.0,
    img_size: int = 256,
) -> Tuple[Optional[pathlib.Path], Optional[torch.Tensor]]:
    """Visualize GT vs predicted 3D occupancy grids as side-by-side voxel renders."""
    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    # Dense 3D query grid [1, V, 3] where V = vis_res^3.
    coord, cell = make_coord_cell_grid((vis_res, vis_res, vis_res), device=device, bs=1)
    coord_flat = coord.reshape(1, -1, 3)
    cell_flat = cell.reshape(1, -1, 3)

    tiles = []
    collected = 0
    global_sample_idx = 0

    for batch in loader:
        def _to_device(x):
            if torch.is_tensor(x):
                return x.to(device, non_blocking=True)
            if isinstance(x, (list, tuple)):
                return [_to_device(v) for v in x]
            if isinstance(x, dict):
                return {k: _to_device(v) for k, v in x.items()}
            return x
        data = _to_device(batch)
        if not isinstance(data, dict):
            continue

        inp = data.get("inp", None)
        gt = data.get("gt", None)
        if gt is None or not torch.is_tensor(gt):
            continue
        B = int(gt.shape[0])

        for i in range(B):
            if collected >= max_samples:
                break

            inp_i = {k: v[i:i+1] for k, v in inp.items()} if isinstance(inp, dict) else inp[i:i+1]
            data_pred = {"inp": inp_i, "gt_coord": coord_flat, "gt_cell": cell_flat}
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred_out = model(data_pred, mode="pred")

            if isinstance(pred_out, dict):
                pred_t = pred_out.get("pred_recon", None)
                if pred_t is None:
                    for v in pred_out.values():
                        if torch.is_tensor(v):
                            pred_t = v
                            break
            else:
                pred_t = pred_out
            if pred_t is None:
                global_sample_idx += 1
                continue

            pred_occ = (pred_t[0].reshape(vis_res, vis_res, vis_res, -1)[..., 0] > occ_threshold).cpu().numpy()

            # GT: load full grid from dataset, else fall back to sparse reconstruction.
            gt_vox = _load_gt_occ_grid(loader, global_sample_idx, vis_res)
            if gt_vox is None:
                gt_vox = _gt_from_sparse_samples(
                    data["gt"][i].cpu(), data["gt_coord"][i].cpu(), vis_res, occ_threshold
                )

            gt_tile = _render_voxel_to_tensor(gt_vox, title="GT", img_size=img_size)
            pred_tile = _render_voxel_to_tensor(pred_occ, title="Pred", img_size=img_size)
            tiles.append(gt_tile)
            tiles.append(pred_tile)
            collected += 1
            global_sample_idx += 1

        if collected >= max_samples:
            break

    if len(tiles) == 0:
        return None, None

    # Rows of [GT, Pred] pairs.
    grid = make_grid(torch.stack(tiles, dim=0), nrow=2, padding=4, pad_value=1.0)
    img_path = save_dir / "val_recon_3d.png"
    save_image(grid, img_path)
    return img_path, grid


@torch.no_grad()
def visualize_reconstructions_temperature(
    model,
    loader,
    device,
    save_dir: pathlib.Path,
    max_samples: int = 4,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    img_size: int = 256,
) -> Tuple[Optional[pathlib.Path], Optional[torch.Tensor]]:
    """Visualize GT vs predicted temperature fields on lat-lon grids.

    Activated when batch value_kind starts with 'temp'. Renders each sample as a
    2D lat x lon heatmap with GT and pred side by side.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    model.eval()
    save_dir.mkdir(parents=True, exist_ok=True)

    tiles = []
    collected = 0

    for batch in loader:
        def _to_device(x):
            if torch.is_tensor(x):
                return x.to(device, non_blocking=True)
            if isinstance(x, (list, tuple)):
                return [_to_device(v) for v in x]
            if isinstance(x, dict):
                return {k: _to_device(v) for k, v in x.items()}
            return x

        data = _to_device(batch)
        if not isinstance(data, dict):
            continue

        vk = data.get("value_kind", None)
        if vk is None:
            return None, None
        if isinstance(vk, (list, tuple)):
            vk = vk[0]
        if not str(vk).startswith("temp"):
            return None, None

        inp = data.get("inp", None)
        gt = data.get("gt", None)
        gt_coord = data.get("gt_coord", None)
        if gt is None or gt_coord is None:
            continue
        B = int(gt.shape[0])

        for i in range(B):
            if collected >= max_samples:
                break

            inp_i = {k: v[i:i+1] for k, v in inp.items()} if isinstance(inp, dict) else inp[i:i+1]
            data_pred = {
                "inp": inp_i,
                "gt_coord": gt_coord[i:i+1],
                "gt_cell": data["gt_cell"][i:i+1],
            }
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pred_out = model(data_pred, mode="pred")

            if isinstance(pred_out, dict):
                pred_t = pred_out.get("pred_recon", None)
                if pred_t is None:
                    for v in pred_out.values():
                        if torch.is_tensor(v):
                            pred_t = v
                            break
            else:
                pred_t = pred_out
            if pred_t is None:
                continue

            gt_vals = gt[i].cpu().numpy().reshape(-1)       # [N]
            pred_vals = pred_t[0].cpu().numpy().reshape(-1)  # [N]
            coords = gt_coord[i].cpu().numpy()               # [N, 3] sphere coords

            # Recover lat/lon from sphere embedding for plotting.
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            lat_rad = np.arcsin(np.clip(z, -1, 1))
            lon_rad = np.arctan2(y, x)

            gt_vals_np = gt_vals
            pred_vals_np = pred_vals

            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            lat_rad = np.arcsin(np.clip(z, -1, 1))
            lon_rad = np.arctan2(y, x)

            # Reshape into lat x lon grid for pcolormesh when possible.
            n_lat = len(np.unique(np.round(lat_rad, 4)))
            n_lon = len(np.unique(np.round(lon_rad, 4)))
            is_grid = (n_lat * n_lon == len(gt_vals_np))

            lat_1d = np.sort(np.unique(np.round(lat_rad, 4)))[::-1]
            lon_1d = np.sort(np.unique(np.round(lon_rad, 4)))

            if is_grid:
                gt_grid = gt_vals_np.reshape(n_lat, n_lon)
                pred_grid = pred_vals_np.reshape(n_lat, n_lon)
                lon_mesh, lat_mesh = np.meshgrid(lon_1d, lat_1d)
            else:
                gt_grid = None

            vmin = float(min(gt_vals_np.min(), pred_vals_np.min()))
            vmax = float(max(gt_vals_np.max(), pred_vals_np.max()))
            norm = Normalize(vmin=vmin, vmax=vmax)

            fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                                     subplot_kw={"projection": "mollweide"})

            for ax, vals_grid, title in [(axes[0], gt_grid, "GT"),
                                          (axes[1], pred_grid if is_grid else None, "Pred")]:
                if vals_grid is not None:
                    ax.pcolormesh(lon_mesh, lat_mesh, vals_grid,
                                  cmap="RdBu_r", norm=norm, shading="gouraud")
                else:
                    ax.scatter(lon_rad, lat_rad, c=gt_vals_np if title == "GT" else pred_vals_np,
                               cmap="RdBu_r", s=15, edgecolors="none", norm=norm)
                ax.set_title(title, fontsize=13, fontweight="bold")
                ax.grid(True, alpha=0.2)

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=250, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            from PIL import Image
            pil_img = Image.open(buf).convert("RGB")
            arr = np.array(pil_img).astype(np.float32) / 255.0
            tile = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
            tiles.append(tile)
            collected += 1

        if collected >= max_samples:
            break

    if len(tiles) == 0:
        return None, None

    grid = make_grid(torch.stack(tiles, dim=0), nrow=1, padding=4, pad_value=1.0)
    img_path = save_dir / "val_recon_temperature.png"
    save_image(grid, img_path)
    return img_path, grid
