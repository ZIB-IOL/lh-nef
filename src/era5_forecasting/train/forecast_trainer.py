"""Trainer for ERA5 temporal forecasting on HiP latents.

Loss is computed in function space (as in ENF for fairness): decode predicted latents
through the frozen stage-1 renderer and compare against GT temperature fields.
Eval reports T_t-MSE (backbone recon) and T_{t+1}-MSE (forecasting).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import models
from trainers.base_trainer import BaseTrainer
from trainers import register


def _build_era5_query_grid(era5_root: str, split: str = "train", device: torch.device = torch.device("cpu")):
    """Build the full 46x90 ERA5 query coordinate + cell grid.

    Returns (coord, cell) each [1, 4140, 3] on device.
    """
    split_dir = os.path.join(era5_root, f"era5_temp2m_16x_{split}")
    if not os.path.isdir(split_dir):
        split_dir = os.path.join(era5_root, split)
    files = sorted([f for f in os.listdir(split_dir) if f.endswith(".npz")])
    if not files:
        raise FileNotFoundError(f"No .npz files in {split_dir}")

    sample = np.load(os.path.join(split_dir, files[0]))
    lat = sample["latitude"]   # (46,) degrees
    lon = sample["longitude"]  # (90,) degrees

    # (lat, lon) degrees → 3D unit-sphere coords (cos θ cos φ, cos θ sin φ, sin θ).
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    lat_rad = np.deg2rad(lat_grid.reshape(-1).astype(np.float64))
    lon_rad = np.deg2rad(lon_grid.reshape(-1).astype(np.float64))
    coords = np.stack([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ], axis=-1).astype(np.float32)  # [4140, 3]

    lat_step = np.deg2rad(abs(float(lat[1] - lat[0])))
    lon_step = np.deg2rad(abs(float(lon[1] - lon[0])))
    cell_val = np.array([lon_step, lon_step, lat_step], dtype=np.float32)
    cell = np.broadcast_to(cell_val.reshape(1, 3), (coords.shape[0], 3)).copy()

    coord_t = torch.from_numpy(coords).unsqueeze(0).to(device)   # [1, 4140, 3]
    cell_t = torch.from_numpy(cell).unsqueeze(0).to(device)       # [1, 4140, 3]
    return coord_t, cell_t


def _load_era5_gt_temperatures(era5_root: str, split: str = "train") -> torch.Tensor:
    """Load all ground truth temperature fields for a split.

    Returns [N, V] float32 tensor in normalized space matching stage-1 training
    (i.e. [0,1] then optionally mapped to [-1,1]).
    """
    from datasets.era5_temperature import T_MIN, T_MAX

    split_dir = os.path.join(era5_root, f"era5_temp2m_16x_{split}")
    if not os.path.isdir(split_dir):
        split_dir = os.path.join(era5_root, split)
    files = sorted([f for f in os.listdir(split_dir) if f.endswith(".npz")])

    temps = []
    for fn in files:
        data = np.load(os.path.join(split_dir, fn))
        temp = data["temperature"].astype(np.float32).reshape(-1)
        # Match ERA5 dataset normalization: map raw kelvin to [0,1] via T_MIN/T_MAX.
        temp = (temp - T_MIN) / (T_MAX - T_MIN)
        temp = np.clip(temp, 0.0, 1.0)
        temps.append(torch.from_numpy(temp))

    return torch.stack(temps, dim=0)  # [N, V]


@register("era5_forecast_trainer")
class ERA5ForecastTrainer(BaseTrainer):

    def run(self):
        try:
            import era5_forecasting.data  # noqa: F401
            import era5_forecasting.models  # noqa: F401
        except Exception:
            pass

        if self.cfg.random_seed is not None:
            self.seed_everything(self.cfg.random_seed, rank_shift=True)

        self.make_datasets()
        # BaseTrainer.__init__ runs auto_schedule before datasets exist; redo it now.
        self._maybe_auto_schedule()
        self._load_stage1_and_grid()

        resume_file = os.path.join(self.cfg._env.save_dir, "last-model.pth")
        if os.path.isfile(resume_file):
            import copy
            from omegaconf import OmegaConf
            ckpt = torch.load(resume_file, map_location="cpu", weights_only=False)
            model_spec = copy.deepcopy(OmegaConf.to_container(self.cfg.model, resolve=True))
            model_spec["sd"] = ckpt["model"]["sd"]
            self.make_model(model_spec)
            self.make_optimizers()
            opt_dict = ckpt.get("optimizers", {})
            for k, v in opt_dict.items():
                if k in self.optimizers:
                    self.optimizers[k].load_state_dict(v["sd"])
            self.log(f"Resumed from {resume_file}.")
        else:
            self.make_model()
            self.make_optimizers()

        self.run_training()

        best_file = os.path.join(self.cfg._env.save_dir, "best-model.pth")
        if os.path.isfile(best_file):
            self.log(f"[forecast] evaluating best-model on val: {best_file}")
            ckpt = torch.load(best_file, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model"]["sd"], strict=False)
            self.evaluate()

        if self.enable_tb:
            self.writer.close()
        if self.enable_wandb:
            import wandb
            wandb.finish()

    def make_datasets(self):
        super().make_datasets()

    def make_optimizers(self):
        import utils
        for name, spec in self.cfg.optimizers.items():
            if spec is not None:
                self.optimizers = {name: utils.make_optimizer(self.model.parameters(), spec)}
                return
        raise ValueError("No optimizer spec found in config")

    def make_model(self, model_spec=None):
        if model_spec is None:
            cfg = self.cfg
            model_spec = dict(cfg.model)

            manifest_path = cfg.get("manifest_path", None)
            if manifest_path is not None:
                manifest_path = os.path.expanduser(os.path.expandvars(str(manifest_path)))
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                split_key = list(manifest["splits"].keys())[0]
                shape = manifest["splits"][split_key]["shape"]
                args = model_spec.get("args", {})
                for key, mkey in [("num_groups", "G"), ("tokens_per_group", "K"), ("token_dim", "C")]:
                    if args.get(key) == "from_manifest":
                        args[key] = int(shape[mkey])
                        if self.is_master:
                            self.log(f"[forecast] {key} = {args[key]} (from manifest)")

        if "sd" in model_spec:
            super().make_model(model_spec)
        else:
            model = models.make(model_spec)
            self.log(f'Model: #params={sum(p.numel() for p in model.parameters())} ({sum(p.numel() for p in model.parameters()):,})')
            self.model = model.to(self.device)
            self.model_ddp = model

    def _load_stage1_and_grid(self):
        """Load frozen stage-1 model, ERA5 query grid, and ground truth temperatures."""
        cfg = self.cfg
        stage1_ckpt = cfg.get("stage1_ckpt", None)
        if stage1_ckpt is None:
            self.log("[forecast] No stage1_ckpt specified — function-space eval disabled.")
            self._stage1 = None
            return

        stage1_ckpt = os.path.expanduser(os.path.expandvars(str(stage1_ckpt)))
        self.log(f"[forecast] Loading stage-1 model from {stage1_ckpt}")
        ckpt = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
        spec = ckpt["model"]
        stage1 = models.make(spec, load_sd=True)
        stage1.eval().to(self.device)
        for p in stage1.parameters():
            p.requires_grad_(False)
        self._stage1 = stage1

        # to_pm1 controls whether stage-1 maps temperature to [-1,1] or keeps [0,1].
        s1_cfg = ckpt.get("cfg", {}) or {}
        s1_train = (s1_cfg.get("datasets", {}) or {}).get("train", {}) or {}
        s1_args = ((s1_train.get("args", {}) or {}).get("dataset", {}) or {}).get("args", {}) or {}
        self._to_pm1 = bool(s1_args.get("to_pm1", True))

        # G/K/C must match the stage-1 latent shape recorded in the manifest.
        manifest_path = cfg.get("manifest_path", None)
        manifest_path = os.path.expanduser(os.path.expandvars(str(manifest_path)))
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        split_key = list(manifest["splits"].keys())[0]
        split_meta = manifest["splits"][split_key]
        shape = split_meta["shape"]
        self._G = int(shape["G"])
        self._K = int(shape["K"])
        self._C = int(shape["C"])
        self._block_key = str(split_meta.get("block_key", "block0"))

        # Always use train-split stats to denormalize before decoding (no leakage).
        train_meta = manifest["splits"].get("train", split_meta)
        mean_t = torch.tensor(train_meta["mean"], dtype=torch.float32)
        std_t = torch.tensor(train_meta["std"], dtype=torch.float32)
        L = self._G * self._K
        if mean_t.ndim == 1 and mean_t.shape[0] == L * self._C:
            mean_t = mean_t.reshape(1, L, self._C)
            std_t = std_t.reshape(1, L, self._C)
        elif mean_t.shape[0] == self._C:
            mean_t = mean_t.reshape(1, 1, self._C)
            std_t = std_t.reshape(1, 1, self._C)
        norm_scale = float(cfg.get("datasets", {}).get("train", {}).get("args", {}).get("norm_scale", 1.0))
        self._latent_mean = mean_t.to(self.device)
        self._latent_std = (std_t * norm_scale).clamp_min(1e-6).to(self.device)

        shard0_path = split_meta["shards"][0]
        shard0_path = os.path.expanduser(os.path.expandvars(shard0_path))
        shard0 = torch.load(shard0_path, map_location="cpu", weights_only=False)
        p_token = shard0["p"].to(dtype=torch.float32, device=self.device)  # [L, d]
        centers = p_token[:: self._K]  # [G, d]
        group_scales = shard0["group_scales"].to(dtype=torch.float32, device=self.device)  # [G, d]
        self._centers = centers
        self._group_scales = group_scales

        era5_root = str(cfg.get("era5_root", ""))
        era5_root = os.path.expanduser(os.path.expandvars(era5_root))
        self._query_coord, self._query_cell = _build_era5_query_grid(
            era5_root, split="train", device=self.device
        )
        self.log(f"[forecast] ERA5 query grid: {self._query_coord.shape}")

        self._gt_temps = {}
        for split_name in ["train", "val", "test"]:
            try:
                gt = _load_era5_gt_temperatures(era5_root, split=split_name)
                if self._to_pm1:
                    gt = gt * 2.0 - 1.0  # match renderer output range
                self._gt_temps[split_name] = gt
                self.log(f"[forecast] GT temperatures loaded for {split_name}: {gt.shape}")
            except Exception:
                pass

    def _denormalize_latent(self, c: torch.Tensor) -> torch.Tensor:
        return c * self._latent_std + self._latent_mean

    def _decode_latent(self, c_norm: torch.Tensor) -> torch.Tensor:
        """Decode normalized latent [B, L, C] -> temperature field [B, V, 1] via the frozen stage-1 renderer.

        Renderer params are frozen but gradients still flow back to the input latent.
        """
        c_raw = self._denormalize_latent(c_norm)
        B = c_raw.shape[0]
        G, K, C = self._G, self._K, self._C

        enc_blocks = {self._block_key: c_raw.view(B, G, K, C).contiguous()}
        enc_regions = {
            "routing_space": "coord",
            "coord_dim": 3,
            "blocks": {
                self._block_key: {
                    "centers": self._centers.unsqueeze(0).expand(B, -1, -1),
                    "scales": self._group_scales.unsqueeze(0).expand(B, -1, -1),
                }
            }
        }

        coord = self._query_coord.expand(B, -1, -1)
        cell = self._query_cell.expand(B, -1, -1)

        pred = self._stage1.renderer(
            z_dec=torch.zeros((B, 1, 1, 1), device=self.device),
            coord=coord,
            cell=cell,
            enc_blocks=enc_blocks,
            enc_regions=enc_regions,
        )
        return pred  # [B, V, 1]

    def train_step(self, data: dict, bp: bool = True) -> dict:
        c_t = data["c"].to(self.device)         # [B, L, C] normalized
        c_next = data["c_next"].to(self.device)  # [B, L, C] normalized
        p = data["p"].to(self.device)            # [L, d] or [B, L, d]

        pred_delta = self.model_ddp(c_t, p)  # [B, L, C]
        pred_abs = c_t + pred_delta
        target_delta = c_next - c_t

        if self._stage1 is not None:
            # Function-space loss in temperature space.
            pred_temp = self._decode_latent(pred_abs)  # [B, V, 1]
            target_temp = self._decode_latent(c_next)  # [B, V, 1]
            loss = F.mse_loss(pred_temp, target_temp)
        else:
            loss = F.mse_loss(pred_delta, target_delta)

        if bp:
            for opt in self.optimizers.values():
                opt.zero_grad()
            loss.backward()
            for opt in self.optimizers.values():
                opt.step()

        ret = {"loss": float(loss.detach().item())}
        with torch.no_grad():
            ret["delta_mse"] = float(F.mse_loss(pred_delta, target_delta).item())
            ret["latent_mse"] = float(F.mse_loss(pred_abs, c_next).item())

        return ret

    def evaluate(self, split: str = "val"):
        import utils
        self._eval_split = split
        self.model_ddp.eval()

        loader_key = split
        if loader_key not in self.loaders:
            self.log(f"[forecast] No '{split}' loader — skipping eval.")
            return {}

        loader = self.loaders[loader_key]
        ave_scalars = {}

        with torch.no_grad():
            for data in loader:
                data = {k: v.cuda() if torch.is_tensor(v) else v for k, v in data.items()}
                ret = self.eval_step(data)
                bs = data["c"].shape[0]
                for k, v in ret.items():
                    if ave_scalars.get(k) is None:
                        ave_scalars[k] = utils.Averager()
                    ave_scalars[k].add(v, n=bs)

        log_parts = []
        result = {}
        for k, avg in ave_scalars.items():
            val = avg.item()
            result[k] = val
            log_parts.append(f"{k}={val:.6e}")

        if self.is_master:
            prefix = f"val" if split == "val" else f"test"
            self.log(f"{prefix}: " + " ".join(log_parts))

            if self.enable_wandb:
                import wandb
                wandb.log({f"{prefix}/{k}": v for k, v in result.items()}, step=self.iter)
            if self.enable_tb:
                for k, v in result.items():
                    self.writer.add_scalar(f"{prefix}/{k}", v, self.iter)

        self.model_ddp.train()
        return ave_scalars

    def eval_step(self, data: dict) -> dict:
        c_t = data["c"].to(self.device)  # [B, L, C] normalized
        c_next = data["c_next"].to(self.device)
        p = data["p"].to(self.device)  # [L, d] or [B, L, d]

        with torch.no_grad():
            pred_delta = self.model_ddp(c_t, p)
            pred_abs = c_t + pred_delta

            ret = {
                "loss": float(F.mse_loss(pred_delta, c_next - c_t).item()),
                "latent_mse": float(F.mse_loss(pred_abs, c_next).item()),
            }

            # Table 6 metrics: Tt_mse (recon) and Tt1_mse (forecasting).
            if self._stage1 is not None:
                pred_temp = self._decode_latent(pred_abs)   # [B, V, 1]
                recon_temp = self._decode_latent(c_t)       # [B, V, 1]

                idx_t = data.get("idx_t", None)
                idx_t1 = data.get("idx_t1", None)
                split = getattr(self, '_eval_split', 'val')
                gt_all = self._gt_temps.get(split, None)

                if gt_all is not None and idx_t is not None and idx_t1 is not None:
                    gt_t = gt_all[idx_t.cpu()].to(self.device).unsqueeze(-1)
                    gt_t1 = gt_all[idx_t1.cpu()].to(self.device).unsqueeze(-1)
                    ret["Tt_mse"] = float(F.mse_loss(recon_temp, gt_t).item())
                    ret["Tt1_mse"] = float(F.mse_loss(pred_temp, gt_t1).item())

        return ret
