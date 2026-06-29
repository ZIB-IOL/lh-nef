"""HiP encoder adapter ingesting coord/value token sets."""

import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models import register

logger = logging.getLogger(__name__)

import sys
from pathlib import Path

hip_path = Path(__file__).parent.parent / "hip"
sys.path.insert(0, str(hip_path))
from hip import HiP, HIP_VARIANTS  # type: ignore


class HiPCoordValueEncoder(nn.Module):
    def __init__(
        self,
        *,
        coord_dim: int = 2,
        value_dim: int = 3,
        # Dummy latent grid for LHNeFBase compatibility
        z_channels: int = 1,
        z_hw: Tuple[int, int] = (1, 1),
        hip_variant: Optional[str] = "CIFAR10_H64S_64_9B",
        num_groups: Optional[Sequence[int]] = None,
        num_self_attends_per_block: Optional[Sequence[int]] = None,
        z_index_dim: Optional[Sequence[int]] = None,
        num_z_channels: Optional[Sequence[int]] = None,
        num_embedding_channels: Optional[int] = None,
        pos_encoding: str = "fourier",
        fourier_bands: int = 32,
        fourier_max_freq: float = 10.0,
        coord_grouping: str = "kdtree",  # none|voxel|kdtree|healpix|s2
        coord_bits: int = 8,
        coord_range: Tuple[float, float] = (-1.0, 1.0),
        normalize_sphere_centers: bool = False,
        hip_run_decoder: bool = False,
        dropout: float = 0.0,
        sdpa_backend: str = "auto",
        # Which encoder block to use for rendering. None=all encoder blocks;
        # "last" or -1=block processor_idx-1; int>=0=that specific block index.
        render_block_index: Optional[int] = None,
        # Group-region cache
        hip_cache_group_regions: bool = False,
    ):
        super().__init__()
        self.coord_dim = int(coord_dim)
        self.value_dim = int(value_dim)
        self.z_channels = int(z_channels)
        self.z_hw = (int(z_hw[0]), int(z_hw[1]))
        self.pos_encoding = str(pos_encoding)
        self.fourier_bands = int(fourier_bands)
        self.fourier_max_freq = float(fourier_max_freq)
        self.coord_grouping = str(coord_grouping or "none")
        self.coord_bits = int(coord_bits)
        self.coord_range = (float(coord_range[0]), float(coord_range[1]))
        self.normalize_sphere_centers = bool(normalize_sphere_centers)
        self.hip_run_decoder = bool(hip_run_decoder)
        self.dropout = float(dropout)
        self.sdpa_backend = str(sdpa_backend)

        self._render_block_index_raw = render_block_index

        if self.z_channels <= 0:
            raise ValueError("z_channels must be >= 1")
        if self.z_hw[0] <= 0 or self.z_hw[1] <= 0:
            raise ValueError("z_hw must be positive")

        # Side-effect outputs for renderer routing
        self.last_enc_blocks = None
        self.encoder_group_regions = None
        self.last_pyramid = None

        if hip_variant is not None:
            if hip_variant not in HIP_VARIANTS:
                raise ValueError(f"Unknown hip_variant: {hip_variant}. Choose from {list(HIP_VARIANTS.keys())}")
            hip_config = HIP_VARIANTS[hip_variant].copy()
            variant_embed = hip_config.pop("num_embedding_channels", None)
            if num_embedding_channels is None:
                num_embedding_channels = variant_embed
        else:
            if any(x is None for x in [num_groups, num_self_attends_per_block, z_index_dim, num_z_channels]):
                raise ValueError("Must provide all HiP config params when hip_variant is None")
            num_blocks = len(num_groups)
            hip_config = {
                "num_groups": num_groups,
                "num_self_attends_per_block": num_self_attends_per_block,
                "z_index_dim": z_index_dim,
                "num_z_channels": num_z_channels,
                "num_cross_attend_heads": (1,) * num_blocks,
                "num_self_attend_heads": (4,) * num_blocks,
                "cross_attend_widening_factor": (1,) * num_blocks,
                "self_attend_widening_factor": (4,) * num_blocks,
            }

        if num_embedding_channels is None:
            raise ValueError("num_embedding_channels must be set (either via hip_variant or explicitly).")
        self.num_embedding_channels = int(num_embedding_channels)

        self.value_proj = nn.Linear(self.value_dim, self.num_embedding_channels)

        # HiP backbone (encoder-only by default)
        self.hip = HiP(
            num_embedding_channels=self.num_embedding_channels,
            dropout=self.dropout,
            pos_encoding=self.pos_encoding,
            fourier_bands=self.fourier_bands,
            fourier_max_freq=self.fourier_max_freq,
            coord_grouping=self.coord_grouping,
            coord_bits=self.coord_bits,
            coord_range=self.coord_range,
            normalize_sphere_centers=self.normalize_sphere_centers,
            build_decoder=bool(self.hip_run_decoder),
            build_reconstruction_head=False,
            **hip_config,
        )

        # Apply cache toggle after construction.
        self.hip_cache_group_regions = bool(hip_cache_group_regions)
        if self.hip_cache_group_regions:
            self.hip.set_cache_group_regions(True)
            print(f"[HiPCoordValueEncoder] group-region caching enabled "
                  f"(assumes constant input coords across forward passes)")

        processor_idx = int(self.hip.processor_idx)
        if self._render_block_index_raw is None:
            self.render_blocks = list(range(processor_idx))
        elif self._render_block_index_raw == "last" or self._render_block_index_raw == -1:
            # Last encoder block (before the single-group bottleneck).
            self.render_blocks = [processor_idx - 1]
        else:
            idx = int(self._render_block_index_raw)
            if idx < 0 or idx >= processor_idx:
                raise ValueError(f"render_block_index={idx} out of range [0, {processor_idx-1}]")
            self.render_blocks = [idx]
        
        logger.info(f"HiPCoordValueEncoder: render_blocks={self.render_blocks} (processor_idx={processor_idx})")

        # Materialize lazy per-modality params inside HiP (Embedder/PositionEncoder).
        self._warmup()

    def _warmup(self):
        try:
            with torch.no_grad():
                B = 1
                N = 8
                coords = torch.zeros(B, N, self.coord_dim, dtype=torch.float32)
                vals = torch.zeros(B, N, self.value_dim, dtype=torch.float32)
                _ = self.forward({"coord": coords, "value": vals})
        except Exception as e:
            raise RuntimeError(f"HiPCoordValueEncoder warmup failed: {e}") from e

    def forward(self, x):
        """Accepts dict{'coord','value'} or (coord,value); returns dummy z grid [B,zc,zh,zw]."""
        if isinstance(x, (tuple, list)) and len(x) == 2:
            coord, value = x[0], x[1]
        elif isinstance(x, dict):
            coord, value = x.get("coord", None), x.get("value", None)
        else:
            raise ValueError("encoder_hip_coord_value expects x as dict{'coord','value'} or (coord,value) tuple.")

        if (not torch.is_tensor(coord)) or (not torch.is_tensor(value)):
            raise ValueError("coord and value must be tensors.")
        if coord.ndim != 3:
            raise ValueError(f"coord must be [B,N,d], got {tuple(coord.shape)}")
        if value.ndim != 3:
            raise ValueError(f"value must be [B,N,v], got {tuple(value.shape)}")
        if int(coord.shape[-1]) != int(self.coord_dim):
            raise ValueError(f"coord_dim mismatch: expected {self.coord_dim}, got {int(coord.shape[-1])}")
        if int(value.shape[-1]) != int(self.value_dim):
            raise ValueError(f"value_dim mismatch: expected {self.value_dim}, got {int(value.shape[-1])}")
        if coord.shape[:2] != value.shape[:2]:
            raise ValueError(f"coord/value N mismatch: coord {tuple(coord.shape)} vs value {tuple(value.shape)}")

        B = int(coord.shape[0])
        device = value.device
        dtype = value.dtype

        # Under bf16 autocast we let value/coord projections run in bf16
        if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16:
            v = self.value_proj(value)  # bf16 via autocast, kept in bf16 downstream
            hip_input = {"data": v}
            hip_coords = {"data": coord}
        else:
            v = self.value_proj(value.to(dtype=torch.float32)).to(dtype=dtype)
            hip_input = {"data": v}
            hip_coords = {"data": coord.to(dtype=torch.float32)}

        hip_out = self.hip(
            hip_input,
            coords=hip_coords,
            return_latents=False,
            return_bottleneck=False,
            return_pyramid=False,
            return_reconstruction=False,
            return_block_latents=True,
            block_indices=list(range(int(self.hip.processor_idx))),
            run_decoder=bool(self.hip_run_decoder),
            return_group_regions=True,
        )

        all_enc_blocks = hip_out.get("block_latents", None)
        all_group_regions = hip_out.get("group_regions", None)

        if all_enc_blocks is not None:
            self.last_enc_blocks = {
                f"block{i}": all_enc_blocks[f"block{i}"]
                for i in self.render_blocks
                if f"block{i}" in all_enc_blocks
            }
        else:
            self.last_enc_blocks = None

        if all_group_regions is not None:
            # group_regions: {"routing_space", "coord_dim", "blocks": {...}}. Filter blocks.
            filtered_blocks = {
                f"block{i}": all_group_regions["blocks"][f"block{i}"]
                for i in self.render_blocks
                if f"block{i}" in all_group_regions.get("blocks", {})
            }
            self.encoder_group_regions = {
                "routing_space": all_group_regions.get("routing_space", "coord"),
                "coord_dim": all_group_regions.get("coord_dim"),
                "blocks": filtered_blocks,
            }
        else:
            self.encoder_group_regions = None

        self.last_pyramid = None

        # Dummy latent grid (renderer ignores it, but LHNeFBase expects a 4D tensor).
        zh, zw = self.z_hw
        z = torch.zeros((B, self.z_channels, zh, zw), device=device, dtype=dtype)
        return z


@register("encoder_hip_coord_value")
def make_hip_coord_value_encoder(**kwargs):
    return HiPCoordValueEncoder(**kwargs)

