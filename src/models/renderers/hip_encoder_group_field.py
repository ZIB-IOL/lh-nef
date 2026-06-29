import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils.geometry import convert_posenc

logger = logging.getLogger(__name__)


class _SDPACrossAttn(nn.Module):
    """Per-query cross-attention over a small KV set using SDPA.

    Shapes:
      q:  [B, Q, D]
      kv: [B, Q, K, C]
    Returns:
      out: [B, Q, D]
    """

    def __init__(
        self,
        d_model: int,
        kv_dim: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        sdpa_backend: str = "auto",  # auto|flash|math|mem_efficient
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = int(d_model)
        self.kv_dim = int(kv_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.dropout = float(dropout)
        self.sdpa_backend = str(sdpa_backend)

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=True)
        self.k_proj = nn.Linear(self.kv_dim, self.d_model, bias=True)
        self.v_proj = nn.Linear(self.kv_dim, self.d_model, bias=True)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=True)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Qn, D = q.shape
        assert D == self.d_model
        # kv can be either:
        #  - shared across queries: [B, K, C]
        #  - per-query selected:    [B, Q, K, C]
        if kv.ndim == 3:
            K = kv.shape[1]
            kv_q = False
        elif kv.ndim == 4:
            assert kv.shape[0] == B and kv.shape[1] == Qn
            K = kv.shape[2]
            kv_q = True
        else:
            raise ValueError(f"kv must be [B,K,C] or [B,Q,K,C], got {tuple(kv.shape)}")

        if not kv_q:
            # Standard cross-attn: same KV for all queries in the batch
            kh = self.k_proj(kv).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, K, Hd]
            vh = self.v_proj(kv).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, K, Hd]
            qh = self.q_proj(q).view(B, Qn, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, Q, Hd]
        else:
            # Per-query KV (expensive fallback)
            qh = self.q_proj(q).view(B * Qn, 1, self.n_heads, self.head_dim).transpose(1, 2)  # [B*Q, H, 1, Hd]
            kh = self.k_proj(kv).view(B * Qn, K, self.n_heads, self.head_dim).transpose(1, 2)  # [B*Q, H, K, Hd]
            vh = self.v_proj(kv).view(B * Qn, K, self.n_heads, self.head_dim).transpose(1, 2)  # [B*Q, H, K, Hd]

        # SDPA backend selection. Flash generally requires fp16/bf16; fp32 "auto" falls back to math.
        backend = self.sdpa_backend
        ctx = None
        if backend != "auto":
            try:
                if backend == "flash":
                    ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
                elif backend == "math":
                    ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
                elif backend in ("mem_efficient", "mem-efficient", "mem"):
                    ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
                else:
                    raise ValueError(f"Unknown sdpa_backend={backend!r}")
            except Exception:
                ctx = None

        out = _sdpa(qh, kh, vh, dropout_p=self.dropout if self.training else 0.0, backend=self.sdpa_backend)
        if not kv_q:
            out = out.transpose(1, 2).contiguous().view(B, Qn, self.d_model)  # [B,Q,D]
        else:
            out = out.transpose(1, 2).contiguous().view(B, Qn, self.d_model)  # [B,Q,D] since q_len=1 per chunk
        return self.out_proj(out)


def _sdpa(qh: torch.Tensor, kh: torch.Tensor, vh: torch.Tensor, *, dropout_p: float, backend: str) -> torch.Tensor:
    """SDPA wrapper with backend selection and a math-backend fallback.

    Stability note: on some CUDA/PyTorch combos SDPA crashes with
    "CUDA error: invalid configuration argument" for very small head_dim
    (e.g. 8/4/2). We proactively force the math backend when head_dim < 16,
    and retry once with math if SDPA throws.
    """
    try:
        head_dim = int(qh.shape[-1])
    except Exception:
        head_dim = 0

    force_math = bool(head_dim > 0 and head_dim < 16)
    eff_backend = "math" if force_math else str(backend)

    def _ctx_for(b: str):
        if b == "auto":
            return None
        try:
            if b == "flash":
                return torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False)
            if b == "math":
                return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
            if b in ("mem_efficient", "mem-efficient", "mem"):
                return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False)
            raise ValueError(f"Unknown sdpa_backend={b!r}")
        except Exception:
            return None

    def _call(ctx):
        if ctx is None:
            return F.scaled_dot_product_attention(qh, kh, vh, attn_mask=None, dropout_p=dropout_p, is_causal=False)
        with ctx:
            return F.scaled_dot_product_attention(qh, kh, vh, attn_mask=None, dropout_p=dropout_p, is_causal=False)

    ctx = _ctx_for(eff_backend)
    try:
        return _call(ctx)
    except RuntimeError:
        # Retry once with math backend.
        ctx2 = _ctx_for("math")
        return _call(ctx2)


@register("renderer_hip_encoder_group_field")
class HiPEncoderGroupFieldRenderer(nn.Module):
    """
    GR-Renderer
    """

    def __init__(
        self,
        net=None,
        *,
        # Which encoder blocks to use (indices < processor_idx). Required at init-time so attention
        # modules and output net are built before optimizer/DDP wrap. If None, configure_from_encoder.
        use_blocks: Optional[List[int]] = None,
        # KV channel dim (C) for each selected block, in the same order as use_blocks.
        block_kv_dims: Optional[List[int]] = None,
        # Base input resolution used by encoder grouping (for coord->pixel routing).
        base_resolution: Optional[int] = None,
        coord_pe_dim: Optional[int] = 32,
        coord_pe_w_max: Optional[float] = 10.0,
        cell_pe_dim: Optional[int] = 16,
        cell_pe_w_max: Optional[float] = 10.0,
        d_model: int = 256,
        n_heads: int = 4,
        dropout: float = 0.0,
        # Query coordinate dimensionality (d). For image-grid NeFs, d=2 (y,x).
        coord_dim: int = 2,
        # 'conv2d': reshape to [B,C,Hq,Wq] and run `net` (Conv2d).
        # 'point_mlp': apply an MLP pointwise on [B,Q,D] and return [B,Q,out_dim].
        head_kind: str = "point_mlp",
        out_dim: int = 3,
        point_hidden: int = 256,
        point_depth: int = 2,
        # 'nearest': nearest base-grid pixel -> 1 group per block.
        # 'bilinear': 4 neighbouring pixels, attend then bilinearly blend per query.
        # 'knn': (coord-space only) k nearest groups; generalizes bilinear beyond grids.
        routing_mode: str = "knn",
        knn_k: int = 4,
        knn_p: float = 1.0,
        knn_eps: float = 1e-6,
        # kNN weighting (coord routing):
        # 'power':    w ~ (dist+eps)^(-p)
        # 'gaussian': w ~ exp(-dist^2 / (2*sigma^2))  (learnable sigma optional)
        knn_weighting: str = "gaussian",
        knn_gaussian_sigma: float = 0.25,
        knn_gaussian_sigma_learnable: bool = True,
        # Per-block k schedule for routing_mode='knn':
        #   k_l = min(knn_k, G_l); if G_l <= knn_small_groups_threshold: k_l = knn_small_groups_k.
        # Keeps fine blocks local while making coarse blocks (few groups) cheaper.
        knn_cap_small_groups: bool = False,
        knn_small_groups_threshold: int = 8,
        knn_small_groups_k: int = 1,
        # Multi-scale fusion across blocks:
        # 'concat' or 'gated_sum' (per-query weighted sum, keeps channel dim = d_model).
        fusion: str = "gated_sum",
        # For fusion='gated_sum', fuse on-the-fly to avoid the [B,Q,L,D] stack.
        # Equivalent because gate weights depend only on q (not per-block outputs).
        gated_sum_stream: bool = True,
        # LIIF-style relative geometry features (rel_coord, rel_cell in grid units),
        # appended to the query embedding input.
        use_liif_geometry: bool = False,
        rel_pe_dim: Optional[int] = None,
        rel_pe_w_max: Optional[float] = 10.0,
        # FiLM modulation of per-block attention outputs using (rel_coord, rel_cell).
        value_film: bool = False,
        value_film_pe_dim: Optional[int] = 16,
        value_film_pe_w_max: float = 10.0,
        value_film_hidden: int = 256,
        sdpa_backend: str = "auto",
        **kwargs,
    ):
        super().__init__()
        self.use_blocks = None if use_blocks is None else [int(x) for x in use_blocks]
        self.block_kv_dims = None if block_kv_dims is None else [int(x) for x in block_kv_dims]
        self.base_resolution = None if base_resolution is None else int(base_resolution)

        self.coord_pe_dim = coord_pe_dim
        self.coord_pe_w_max = coord_pe_w_max
        self.cell_pe_dim = cell_pe_dim
        self.cell_pe_w_max = cell_pe_w_max
        self.coord_dim = int(coord_dim)
        if self.coord_dim <= 0:
            raise ValueError("coord_dim must be >= 1")
        self.routing_mode = str(routing_mode or "nearest").lower().strip()
        if self.routing_mode not in ("nearest", "bilinear", "knn"):
            raise ValueError("routing_mode must be 'nearest', 'bilinear', or 'knn'.")
        self.knn_k = int(knn_k)
        self.knn_p = float(knn_p)
        self.knn_eps = float(knn_eps)
        if self.knn_k <= 0:
            raise ValueError("knn_k must be >= 1")
        if self.knn_eps <= 0:
            raise ValueError("knn_eps must be > 0")

        self.knn_weighting = str(knn_weighting or "power").lower().strip()
        if self.knn_weighting not in ("power", "gaussian"):
            raise ValueError("knn_weighting must be 'power' or 'gaussian'")
        self.knn_gaussian_sigma = float(knn_gaussian_sigma)
        if self.knn_gaussian_sigma <= 0:
            raise ValueError("knn_gaussian_sigma must be > 0")
        self.knn_gaussian_sigma_learnable = bool(knn_gaussian_sigma_learnable)
        self._knn_log_sigma = None
        if self.knn_gaussian_sigma_learnable:
            self._knn_log_sigma = nn.Parameter(torch.log(torch.tensor(float(self.knn_gaussian_sigma))))
        self.knn_cap_small_groups = bool(knn_cap_small_groups)
        self.knn_small_groups_threshold = int(knn_small_groups_threshold)
        self.knn_small_groups_k = int(knn_small_groups_k)
        if self.knn_small_groups_threshold <= 0:
            raise ValueError("knn_small_groups_threshold must be >= 1")
        if self.knn_small_groups_k <= 0:
            raise ValueError("knn_small_groups_k must be >= 1")
        self.fusion = str(fusion or "concat").lower().strip()
        if self.fusion in ("gated_sum", "gated-sum", "gated"):
            self.fusion = "gated_sum"
        if self.fusion not in ("concat", "gated_sum"):
            raise ValueError("fusion must be 'concat' or 'gated_sum'.")
        self.gated_sum_stream = bool(gated_sum_stream)
        self.use_liif_geometry = bool(use_liif_geometry)
        self.rel_pe_dim = rel_pe_dim
        self.rel_pe_w_max = rel_pe_w_max

        self.value_film = bool(value_film)
        self.value_film_pe_dim = value_film_pe_dim
        self.value_film_pe_w_max = float(value_film_pe_w_max)
        self.value_film_hidden = int(value_film_hidden)
        self._value_film = None  # built in _build_modules (depends on d_model/coord_dim)

        if self.coord_pe_dim is not None:
            assert int(self.coord_pe_dim) % 2 == 0
        if self.cell_pe_dim is not None:
            assert int(self.cell_pe_dim) % 2 == 0
        if self.rel_pe_dim is not None:
            assert int(self.rel_pe_dim) % 2 == 0

        coord_dim_total = self.coord_dim if self.coord_pe_dim is None else self.coord_dim * int(self.coord_pe_dim)
        cell_dim_total = self.coord_dim if self.cell_pe_dim is None else self.coord_dim * int(self.cell_pe_dim)
        rel_coord_dim_total = self.coord_dim if self.rel_pe_dim is None else self.coord_dim * int(self.rel_pe_dim)
        rel_cell_dim_total = self.coord_dim if self.rel_pe_dim is None else self.coord_dim * int(self.rel_pe_dim)
        self.query_in_dim = int(coord_dim_total + cell_dim_total)
        if self.use_liif_geometry:
            self.query_in_dim += int(rel_coord_dim_total + rel_cell_dim_total)

        self.d_model = int(d_model)
        self.query_mlp = nn.Sequential(
            nn.Linear(self.query_in_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.sdpa_backend = str(sdpa_backend)

        # `net` is only required when head_kind == 'conv2d'.
        self._net_spec = dict(net) if net is not None else None
        self._configured = False
        self._attn = nn.ModuleDict()
        self._gate = None  # built in _build_modules when fusion='gated_sum'
        self.net = None

        self.head_kind = str(head_kind or "conv2d").lower().strip()
        if self.head_kind in ("point", "pointwise", "mlp", "point_mlp", "point-mlp"):
            self.head_kind = "point_mlp"
        if self.head_kind not in ("conv2d", "point_mlp"):
            raise ValueError("head_kind must be 'conv2d' or 'point_mlp'.")
        self.out_dim = int(out_dim)
        if self.out_dim <= 0:
            raise ValueError("out_dim must be >= 1")
        self.point_hidden = int(point_hidden)
        self.point_depth = int(point_depth)
        self._point_head = None

        if (self.use_blocks is not None) and (self.block_kv_dims is not None) and (self.base_resolution is not None):
            self._build_modules()

    def _build_modules(self):
        if self._configured:
            return
        if self.use_blocks is None or self.block_kv_dims is None or self.base_resolution is None:
            raise RuntimeError("Renderer not configured: missing use_blocks/block_kv_dims/base_resolution.")
        if len(self.use_blocks) != len(self.block_kv_dims):
            raise ValueError("use_blocks and block_kv_dims must have same length.")

        # Per-block attention modules (built at construction time so DDP wraps them correctly).
        for bi, kv_dim in zip(self.use_blocks, self.block_kv_dims):
            self._attn[f"block{int(bi)}"] = _SDPACrossAttn(
                d_model=self.d_model,
                kv_dim=int(kv_dim),
                n_heads=self.n_heads,
                dropout=self.dropout,
                sdpa_backend=self.sdpa_backend,
            )

        if self.fusion == "gated_sum":
            L = int(len(self.use_blocks))
            self._gate = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.SiLU(),
                nn.Linear(self.d_model, L),
            )

        in_channels = int(self.d_model) if self.fusion == "gated_sum" else int(len(self.use_blocks) * self.d_model)

        # FiLM MLP produces (gamma, beta) in R^{2*d_model} from rel geometry.
        # Last layer is zero-init so default modulation is identity.
        if self.value_film:
            pe = self.value_film_pe_dim
            if pe is None:
                rel_in = int(2 * self.coord_dim)  # rel_coord + rel_cell
            else:
                if int(pe) % 2 != 0:
                    raise ValueError("value_film_pe_dim must be even")
                rel_in = int(2 * self.coord_dim * int(pe))
            hid = int(self.value_film_hidden)
            self._value_film = nn.Sequential(
                nn.Linear(rel_in, hid),
                nn.SiLU(),
                nn.Linear(hid, 2 * int(self.d_model)),
            )
            with torch.no_grad():
                last = self._value_film[-1]
                if isinstance(last, nn.Linear):
                    nn.init.zeros_(last.weight)
                    if last.bias is not None:
                        nn.init.zeros_(last.bias)

        if self.head_kind == "conv2d":
            if self._net_spec is None:
                raise ValueError("head_kind='conv2d' requires a `net` spec at construction time.")
            net_spec = dict(self._net_spec)
            net_spec = {**net_spec, "args": dict(net_spec.get("args", {}))}
            net_spec["args"]["in_channels"] = int(in_channels)
            self.net = models.make(net_spec)
        else:
            # Pointwise MLP head: [B,Q,in_channels] -> [B,Q,out_dim].
            depth = int(self.point_depth)
            hidden = int(self.point_hidden)
            if depth <= 1:
                self._point_head = nn.Sequential(nn.Linear(in_channels, int(self.out_dim)))
            else:
                layers = [nn.Linear(in_channels, hidden), nn.SiLU()]
                for _ in range(depth - 2):
                    layers += [nn.Linear(hidden, hidden), nn.SiLU()]
                layers += [nn.Linear(hidden, int(self.out_dim))]
                self._point_head = nn.Sequential(*layers)
        self._configured = True

    def get_last_layer_weight(self):
        if self.head_kind == "conv2d":
            if self.net is None:
                raise RuntimeError("Renderer net is not built yet.")
            return self.net.get_last_layer_weight()
        if self._point_head is None:
            raise RuntimeError("Point head is not built yet.")
        last = None
        for m in self._point_head.modules():
            if isinstance(m, nn.Linear):
                last = m
        if last is None:
            raise RuntimeError("Point head has no Linear layer.")
        return last.weight

    def configure_from_encoder(self, encoder):
        """Auto-configure from a HiP encoder. Called at model construction time (pre-DDP)."""
        if self._configured:
            return

        hip = getattr(encoder, "hip", None) or getattr(encoder, "hip_model", None)
        if hip is None:
            raise RuntimeError("configure_from_encoder expected encoder.hip to exist.")

        processor_idx = int(getattr(hip, "processor_idx"))

        encoder_render_blocks = getattr(encoder, "render_blocks", None)
        if encoder_render_blocks is not None:
            use_blocks = list(encoder_render_blocks)
        else:
            use_blocks = list(range(processor_idx))

        num_z_channels = list(getattr(hip, "num_z_channels", []))
        if len(num_z_channels) == 0:
            raise RuntimeError("HiP must expose num_z_channels for auto configuration.")
        kv_dims = [int(num_z_channels[i]) for i in use_blocks]

        num_groups = list(getattr(hip, "num_groups", []))
        z_index_dim = list(getattr(hip, "z_index_dim", []))

        base_res = int(getattr(encoder, "resolution", None) or getattr(encoder, "res", None) or 0)
        if base_res <= 0:
            # For coord/value encoders base_resolution is unused; keep dummy positive for checks.
            base_res = int(getattr(encoder, "resolution", 1) or 1)

        self.use_blocks = use_blocks
        self.block_kv_dims = kv_dims
        self.base_resolution = base_res

        logger.info("=" * 60)
        logger.info("HiPEncoderGroupFieldRenderer: Block configuration")
        logger.info(f"  processor_idx (bottleneck): {processor_idx}")
        logger.info(f"  Rendering with blocks: {use_blocks}")
        total_floats = 0
        for i, bi in enumerate(use_blocks):
            G = int(num_groups[bi]) if bi < len(num_groups) else "?"
            K = int(z_index_dim[bi]) if bi < len(z_index_dim) else "?"
            C = int(kv_dims[i])
            if isinstance(G, int) and isinstance(K, int):
                floats = G * K * C
                total_floats += floats
                logger.info(f"    block{bi}: G={G:4d}, K={K:3d}, C={C:3d}  →  {floats:,} floats/image")
            else:
                logger.info(f"    block{bi}: G={G}, K={K}, C={C}")
        if total_floats > 0:
            logger.info(f"  Total floats/image for rendering: {total_floats:,}")
        logger.info("=" * 60)

        self._build_modules()

    def _encode_query(
        self,
        coord: torch.Tensor,
        cell: torch.Tensor,
        *,
        rel_coord: Optional[torch.Tensor] = None,
        rel_cell: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cc = coord
        if self.coord_pe_dim is not None:
            cc = convert_posenc(cc, int(self.coord_pe_dim), float(self.coord_pe_w_max))
        ce = cell
        if self.cell_pe_dim is not None:
            ce = convert_posenc(ce, int(self.cell_pe_dim), float(self.cell_pe_w_max))
        qin_parts = [cc, ce]
        if self.use_liif_geometry:
            if rel_coord is None or rel_cell is None:
                if self.base_resolution is None:
                    raise RuntimeError("use_liif_geometry requires base_resolution (configure_from_encoder) unless rel_coord/rel_cell are provided.")
                if int(self.coord_dim) != 2:
                    raise RuntimeError("Raster LIIF fallback assumes coord_dim=2; pass rel_coord/rel_cell for coord routing.")
                H0 = W0 = int(self.base_resolution)
                if H0 <= 1:
                    raise RuntimeError("use_liif_geometry requires base_resolution >= 2.")
                # Nearest pixel center under the same mapping as routing (align_corners=True style).
                y = coord[..., 0]
                x = coord[..., 1]
                c = ((x + 1) * 0.5 * (W0 - 1)).round().clamp(0, W0 - 1)
                r = ((y + 1) * 0.5 * (H0 - 1)).round().clamp(0, H0 - 1)
                y0 = (2.0 * r / float(H0 - 1)) - 1.0
                x0 = (2.0 * c / float(W0 - 1)) - 1.0
                rel_coord = torch.stack([y - y0, x - x0], dim=-1)
                rel_coord = rel_coord * rel_coord.new_tensor([float(H0), float(W0)])
                rel_cell = cell.clone()
                rel_cell = rel_cell * rel_cell.new_tensor([float(H0), float(W0)])
            if self.rel_pe_dim is not None:
                rel_coord = convert_posenc(rel_coord, int(self.rel_pe_dim), float(self.rel_pe_w_max))
                rel_cell = convert_posenc(rel_cell, int(self.rel_pe_dim), float(self.rel_pe_w_max))
            qin_parts.extend([rel_coord, rel_cell])
        qin = torch.cat(qin_parts, dim=-1)  # [B, ..., Din]
        q = self.query_mlp(qin)  # [B, ..., D]
        return q

    @staticmethod
    def _coord_to_pixel_index(coord: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # coord [...,2] in [-1,1], convention is (y,x) since make_coord_grid uses indexing='ij'
        # (see src/utils/geometry.py::make_coord_grid). Nearest pixel center (hard routing).
        y = coord[..., 0]
        x = coord[..., 1]
        c = ((x + 1) * 0.5 * (W - 1)).round().clamp(0, W - 1).to(torch.long)
        r = ((y + 1) * 0.5 * (H - 1)).round().clamp(0, H - 1).to(torch.long)
        return r * W + c

    @staticmethod
    def _coord_to_bilinear_pixel_indices_and_weights(coord: torch.Tensor, H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear neighbors on the base grid.

        Returns:
          idx: [...,4] pixel indices (r*W+c) in raster order
          w:   [...,4] bilinear weights that sum to 1
        Corner order: (y0,x0), (y0,x1), (y1,x0), (y1,x1)
        """
        y = coord[..., 0]
        x = coord[..., 1]
        xf = (x + 1) * 0.5 * (W - 1)
        yf = (y + 1) * 0.5 * (H - 1)
        x0 = xf.floor()
        y0 = yf.floor()
        x1 = x0 + 1
        y1 = y0 + 1
        x0i = x0.clamp(0, W - 1).to(torch.long)
        x1i = x1.clamp(0, W - 1).to(torch.long)
        y0i = y0.clamp(0, H - 1).to(torch.long)
        y1i = y1.clamp(0, H - 1).to(torch.long)
        wx1 = (xf - x0).clamp(0.0, 1.0)
        wy1 = (yf - y0).clamp(0.0, 1.0)
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1
        w00 = wy0 * wx0
        w01 = wy0 * wx1
        w10 = wy1 * wx0
        w11 = wy1 * wx1
        idx00 = y0i * W + x0i
        idx01 = y0i * W + x1i
        idx10 = y1i * W + x0i
        idx11 = y1i * W + x1i
        idx = torch.stack([idx00, idx01, idx10, idx11], dim=-1)
        w = torch.stack([w00, w01, w10, w11], dim=-1).to(dtype=coord.dtype)
        ws = w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        w = w / ws
        return idx, w

    def _select_groups(self, pix_idx: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
        # pix_idx: [B,Q] int64 in [0,N0)
        # ends: [G] int64 increasing, last == N0
        # returns g_idx: [B,Q] in [0,G)
        return torch.bucketize(pix_idx, ends, right=False)

    @staticmethod
    def _gather_group_feat(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, G, d]
        idx: [B, Q] long
        ->   [B, Q, d]
        """
        B, G, d = x.shape
        if idx.ndim != 2 or idx.shape[0] != B:
            raise ValueError(f"idx must be [B,Q], got {tuple(idx.shape)} for B={B}")
        return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))

    def _coord_knn_groups_and_weights(
        self,
        coord_flat: torch.Tensor,
        centers: torch.Tensor,
        *,
        k: int,
        p: float,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        coord_flat: [B,Q,d], centers: [B,G,d]
        Returns idx [B,Q,k] long, w [B,Q,k] with sum_k w = 1.
        """
        B, Qn, d = coord_flat.shape
        if d != centers.shape[-1]:
            raise ValueError(f"coord dim mismatch: coord has {d}, centers has {centers.shape[-1]}")
        G = int(centers.shape[1])
        kk = int(min(int(k), int(G)))
        dist2 = (coord_flat.unsqueeze(2) - centers.unsqueeze(1)).pow(2).sum(dim=-1)  # [B,Q,G]
        vals, idx = torch.topk(dist2, k=kk, dim=2, largest=False)
        if self.knn_weighting == "gaussian":
            # If sigma is learnable, keep it as a tensor so gradients can flow.
            if self._knn_log_sigma is not None:
                sigma = torch.exp(self._knn_log_sigma).clamp_min(1e-4)  # scalar tensor
                denom = 2.0 * (sigma * sigma) + float(eps)
                w = torch.exp(-vals / denom)
            else:
                sigma_f = float(self.knn_gaussian_sigma)
                w = torch.exp(-vals / (2.0 * (sigma_f**2) + float(eps)))
        else:
            dist = torch.sqrt(vals + float(eps))
            w = (dist + float(eps)).pow(-float(p))
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(float(eps))
        return idx.to(torch.long), w.to(dtype=coord_flat.dtype)

    def _encode_rel_geom(self, rel_coord: torch.Tensor, rel_cell: torch.Tensor) -> torch.Tensor:
        """rel_coord, rel_cell: [B,Q,coord_dim] -> [B,Q,Din] for value FiLM."""
        rc = rel_coord
        rl = rel_cell
        pe = self.value_film_pe_dim
        if pe is not None:
            rc = convert_posenc(rc, int(pe), float(self.value_film_pe_w_max))
            rl = convert_posenc(rl, int(pe), float(self.value_film_pe_w_max))
        return torch.cat([rc, rl], dim=-1)

    def forward(
        self,
        z_dec,
        coord,
        cell,
        *,
        enc_blocks: Optional[Dict] = None,
        enc_regions: Optional[List[Dict]] = None,
        return_features: bool = False,
        **kwargs,
    ):
        # z_dec unused; signature compatibility.
        if enc_blocks is None or enc_regions is None:
            raise ValueError("renderer_hip_encoder_group_field requires enc_blocks and enc_regions kwargs.")

        if not self._configured:
            raise RuntimeError(
                "renderer_hip_encoder_group_field is not configured. "
                "Expected configure_from_encoder() to be called during model init."
            )

        use = list(self.use_blocks)

        # enc_regions accepts:
        #  - list[dict] of regions (raster order)
        #  - {"token_order": ..., "inv_perm": [...], "regions": [...]}
        #  - {"routing_space": "coord", "blocks": {"block0": {"centers", "scales"}, ...}}
        routing_space = "raster"
        token_order = "raster"
        inv_perm = None
        regions = enc_regions
        regions_blocks = None
        if isinstance(enc_regions, dict):
            if str(enc_regions.get("routing_space", "")).lower().strip() == "coord" and ("blocks" in enc_regions):
                routing_space = "coord"
                regions_blocks = enc_regions.get("blocks", {})
            else:
                token_order = str(enc_regions.get("token_order", "raster")).lower().strip()
                regions = enc_regions.get("regions", [])
                if token_order in ("morton", "z", "zorder", "z-order"):
                    token_order = "morton"
                    inv_list = enc_regions.get("inv_perm", None)
                    if not isinstance(inv_list, list) or len(inv_list) == 0:
                        raise ValueError("enc_regions token_order='morton' requires 'inv_perm' list.")
                    inv_perm = torch.tensor(inv_list, device=coord.device, dtype=torch.long)
        if routing_space == "raster" and self.routing_mode == "knn":
            raise ValueError("routing_mode='knn' is only supported when enc_regions specifies coord routing (routing_space='coord').")

        # Query shape
        B = coord.shape[0]
        if int(coord.shape[-1]) != int(self.coord_dim):
            raise ValueError(f"coord last-dim must be coord_dim={self.coord_dim}, got {int(coord.shape[-1])}")
        if int(cell.shape[-1]) != int(self.coord_dim):
            raise ValueError(f"cell last-dim must be coord_dim={self.coord_dim}, got {int(cell.shape[-1])}")
        qshape = coord.shape[1:-1]
        Qn = int(torch.tensor(qshape).prod().item()) if len(qshape) > 0 else 1

        # Encode query (optionally with LIIF-style relative geometry).
        rel_coord = None
        rel_cell = None
        coord_flat = coord.view(B, Qn, coord.shape[-1])
        cell_flat = cell.view(B, Qn, cell.shape[-1])
        if routing_space == "coord" and self.use_liif_geometry:
            if regions_blocks is None or len(regions_blocks) == 0:
                raise ValueError("coord routing with use_liif_geometry requires enc_regions['blocks'] with centers/scales.")
            base_bi = int(min(use))
            base_key = f"block{base_bi}"
            rb = regions_blocks.get(base_key, None)
            if rb is None:
                raise KeyError(f"enc_regions['blocks'] missing {base_key}. Available: {list(regions_blocks.keys())}")
            centers0 = rb.get("centers", None)
            scales0 = rb.get("scales", None)
            if centers0 is None or scales0 is None:
                raise ValueError(f"{base_key} must provide 'centers' and 'scales' for coord LIIF geometry.")
            # nearest group anchor at the finest used level
            dist2 = (coord_flat.unsqueeze(2) - centers0.unsqueeze(1)).pow(2).sum(dim=-1)  # [B,Q,G]
            g0 = torch.argmin(dist2, dim=-1).to(torch.long)  # [B,Q]
            ctr = self._gather_group_feat(centers0, g0)       # [B,Q,d]
            scl = self._gather_group_feat(scales0, g0).clamp_min(float(self.knn_eps))  # [B,Q,d]
            rel_coord = ((coord_flat - ctr) / scl).view(B, *qshape, -1)
            rel_cell = (cell_flat / scl).view(B, *qshape, -1)

        q = self._encode_query(coord, cell, rel_coord=rel_coord, rel_cell=rel_cell).view(B, Qn, self.d_model)  # [B,Q,D]

        # Raster routing precomputations
        pix_idx = None
        pix_idx4 = None
        w4 = None
        if routing_space == "raster":
            H0 = W0 = int(self.base_resolution)
            N0 = H0 * W0
            pix_idx = self._coord_to_pixel_index(coord, H0, W0).view(B, Qn)  # [B,Q] in raster order
            if token_order == "morton":
                if inv_perm is None or int(inv_perm.numel()) != int(N0):
                    raise ValueError(f"Invalid inv_perm for morton routing: got {None if inv_perm is None else int(inv_perm.numel())}, expected {N0}")
                pix_idx = inv_perm[pix_idx]
            if self.routing_mode == "bilinear":
                idx4, w4_ = self._coord_to_bilinear_pixel_indices_and_weights(coord, H0, W0)  # [B,...,4]
                pix_idx4 = idx4.view(B, Qn, 4)
                w4 = w4_.view(B, Qn, 4).to(dtype=q.dtype)
                if token_order == "morton":
                    pix_idx4 = inv_perm[pix_idx4]

        outs = []
        device = coord.device

        # For gated-sum fusion, the gate depends only on q, so we can compute weights once and
        # optionally fuse each block output on-the-fly to reduce peak memory.
        w_gated = None
        fused_gated = None
        if self.fusion == "gated_sum":
            if self._gate is None:
                raise RuntimeError("fusion='gated_sum' but gate module is not built.")
            logits = self._gate(q)  # [B,Q,L]
            w_gated = torch.softmax(logits, dim=-1).to(dtype=q.dtype)  # [B,Q,L]
            if self.gated_sum_stream:
                fused_gated = torch.zeros_like(q)  # [B,Q,D]

        if routing_space == "coord":
            if regions_blocks is None:
                raise ValueError("coord routing requires enc_regions['blocks'].")
            region_iter = [int(bi) for bi in use]
        else:
            region_iter = regions

        out_l = 0  # index within used blocks order (matches gate's L dimension)
        for r in region_iter:
            if routing_space == "coord":
                bi = int(r)
                key = f"block{bi}"
                rb = regions_blocks.get(key, None)
                if rb is None:
                    raise KeyError(f"enc_regions['blocks'] missing {key}. Available: {list(regions_blocks.keys())}")
            else:
                bi = int(r["block_idx"])
                if bi not in use:
                    continue
                key = f"block{bi}"
            if key not in enc_blocks:
                raise KeyError(f"enc_blocks missing {key}. Available: {list(enc_blocks.keys())}")
            z = enc_blocks[key]  # [B,G,K,C]
            if z.ndim != 4:
                raise ValueError(f"enc_blocks[{key}] must be [B,G,K,C], got {tuple(z.shape)}")
            if int(z.shape[3]) != int(self._attn[key].kv_dim):
                raise ValueError(f"{key} kv_dim mismatch: renderer expects {self._attn[key].kv_dim}, got {int(z.shape[3])}")
            # Select groups for this block.
            ends = None
            g_idx = None
            g_idx4 = None
            w_multi = None
            if routing_space == "coord":
                centers = rb.get("centers", None)
                if centers is None:
                    raise ValueError(f"{key} missing centers for coord routing.")
                G = int(z.shape[1])
                if int(centers.shape[1]) != G:
                    raise ValueError(f"{key} centers G mismatch: centers has {int(centers.shape[1])}, enc_blocks has {G}")
                if self.routing_mode in ("knn", "bilinear"):
                    if self.routing_mode == "bilinear":
                        kk = int(4)
                    else:
                        kk = int(self.knn_k)
                        # Per-block k schedule for coarse blocks
                        if self.knn_cap_small_groups and int(G) <= int(self.knn_small_groups_threshold):
                            kk = int(self.knn_small_groups_k)
                    g_idx4, w_multi = self._coord_knn_groups_and_weights(
                        coord_flat, centers, k=kk, p=self.knn_p, eps=self.knn_eps
                    )  # [B,Q,k], [B,Q,k]
                    g_idx = None
                else:
                    dist2 = (coord_flat.unsqueeze(2) - centers.unsqueeze(1)).pow(2).sum(dim=-1)  # [B,Q,G]
                    g_idx = torch.argmin(dist2, dim=-1).to(torch.long)  # [B,Q]
                    g_idx4 = None
            else:
                ends = torch.tensor(r["ends"], device=device, dtype=torch.long)
                if self.routing_mode == "bilinear":
                    assert pix_idx4 is not None and w4 is not None
                    g_idx4 = self._select_groups(pix_idx4.reshape(B, Qn * 4), ends).view(B, Qn, 4)  # [B,Q,4]
                    w_multi = w4
                    g_idx = None
                else:
                    assert pix_idx is not None
                    g_idx = self._select_groups(pix_idx, ends)  # [B,Q]
                    g_idx4 = None

            # All queries assigned to group g attend to the SAME KV set z[:,g,:,:] (shared KV).
            # Avoids [B,Q,K,*] KV replication that causes OOM on large batches.
            G = int(z.shape[1])
            if ends is not None and ends.numel() != G:
                raise ValueError(f"Region ends length {ends.numel()} != G {G} for {key}")
            out_block = torch.zeros_like(q)  # [B,Q,D]

            # Pre-allocate the two arange tensors used inside the per-group loop
            ar_Q_full = torch.arange(Qn, device=device)  # [Qn]
            ar_B = torch.arange(B, device=device)        # [B]

            for g in range(G):
                if g_idx4 is not None:
                    assert w_multi is not None
                    m4 = (g_idx4 == g)  # [B,Q,k]
                    w_sum = (m4.to(dtype=w_multi.dtype) * w_multi).sum(dim=-1)  # [B,Q]
                    mask = w_sum > 0
                else:
                    assert g_idx is not None
                    w_sum = None
                    mask = (g_idx == g)  # [B,Q] bool
                counts = mask.sum(dim=1)  # [B]
                max_qg = int(counts.max().item())
                if max_qg == 0:
                    continue

                # Bring selected query indices to the front per batch element.
                # Do NOT use argsort(mask): for ImageNet128 (Q=16384, G=256, multiple blocks) the
                # full [B,Q] permutation allocation blows up peak memory. topk on the bool mask is
                # equivalent (we still gate with `valid` below) and far cheaper.
                _, idx_sel = mask.to(torch.int32).topk(k=max_qg, dim=1, largest=True, sorted=False)  # [B,max_qg]

                # Gather queries into a padded tensor [B,max_qg,D].
                q_pad = q.gather(1, idx_sel.unsqueeze(-1).expand(-1, -1, self.d_model))

                # Zero padded positions (deterministic outputs).
                # Slice the pre-allocated arange — math-identical to
                # torch.arange(max_qg, device=device).unsqueeze(0).
                ar = ar_Q_full[:max_qg].unsqueeze(0)  # [1,max_qg]
                valid = ar < counts.unsqueeze(1)  # [B,max_qg] bool
                q_pad = q_pad * valid.unsqueeze(-1).to(dtype=q_pad.dtype)

                kv_shared = z[:, g]  # [B,K,Ckv]
                out_pad = self._attn[key](q_pad, kv_shared)  # [B,max_qg,D]
                if g_idx4 is not None:
                    assert w_sum is not None
                    w_sel = w_sum.gather(1, idx_sel)  # [B,max_qg]
                    w_sel = w_sel * valid.to(dtype=w_sel.dtype)
                    out_pad = out_pad * w_sel.unsqueeze(-1).to(dtype=out_pad.dtype)

                # Scatter only valid outputs back to [B,Q,D] without Python loops.
                # Use the pre-allocated arange — math-identical to
                # torch.arange(B, device=device).repeat_interleave(counts).
                b_ids = ar_B.repeat_interleave(counts)
                q_ids = idx_sel[valid]  # [sum(counts)]
                out_vals = out_pad[valid]  # [sum(counts), D]
                # Accumulate: in bilinear/kNN a query can receive contributions from multiple groups.
                # index_add_ on a flat view avoids the large temporaries of `out_block[b_ids, q_ids] = ...`
                # which stresses the CUDA caching allocator on ImageNet128.
                flat = out_block.view(B * Qn, self.d_model)
                lin = (b_ids * Qn + q_ids).to(torch.long)
                flat.index_add_(0, lin, out_vals)

            out = out_block
            # FiLM-modulate per-block outputs using relative pose to the nearest group center.
            if self.value_film and (self._value_film is not None) and (routing_space == "coord"):
                try:
                    centers = rb.get("centers", None)
                    scales = rb.get("scales", None)
                    if centers is not None and scales is not None:
                        if g_idx4 is not None:
                            g0 = g_idx4[..., 0]
                        else:
                            g0 = g_idx
                        if g0 is not None:
                            ctr = self._gather_group_feat(centers, g0)  # [B,Q,coord_dim]
                            scl = self._gather_group_feat(scales, g0).clamp_min(float(self.knn_eps))
                            rel_coord_f = (coord_flat - ctr) / scl
                            rel_cell_f = cell_flat / scl
                            din = self._encode_rel_geom(rel_coord_f, rel_cell_f)  # [B,Q,Din]
                            gb = self._value_film(din)  # [B,Q,2D]
                            gamma, beta = gb.chunk(2, dim=-1)
                            out = out * (1.0 + gamma) + beta
                except Exception:
                    pass
            if self.fusion == "gated_sum" and self.gated_sum_stream:
                assert fused_gated is not None and w_gated is not None
                wi = w_gated[:, :, out_l].unsqueeze(-1).to(dtype=out.dtype)  # [B,Q,1]
                fused_gated = fused_gated + out * wi
            else:
                outs.append(out)
            out_l += 1

        if (self.fusion != "gated_sum" or not self.gated_sum_stream) and len(outs) == 0:
            raise RuntimeError("No encoder blocks selected for rendering.")

        if self.fusion == "gated_sum":
            if self.gated_sum_stream:
                assert fused_gated is not None
                fused = fused_gated
            else:
                if self._gate is None:
                    raise RuntimeError("fusion='gated_sum' but gate module is not built.")
                hs = torch.stack(outs, dim=2)  # [B,Q,L,D]
                logits = self._gate(q)         # [B,Q,L]
                w = torch.softmax(logits, dim=-1).to(dtype=hs.dtype)
                fused = (hs * w.unsqueeze(-1)).sum(dim=2)  # [B,Q,D]
        else:
            fused = torch.cat(outs, dim=-1)  # [B,Q, len(blocks)*D]
        if self.head_kind == "conv2d":
            if len(qshape) != 2:
                raise ValueError(f"head_kind='conv2d' requires 2D query grids (qshape len=2), got qshape={tuple(qshape)}")
            # net expects [B,C,Hq,Wq].
            layout = fused.view(B, *qshape, fused.shape[-1]).permute(0, -1, *range(1, 1 + len(qshape))).contiguous()
            pred = self.net(layout)
            if return_features:
                return {"pred": pred, "fused": fused, "q": q}
            return pred
        # Pointwise head returns [B,Q,out_dim].
        if self._point_head is None:
            raise RuntimeError("head_kind='point_mlp' but point head is not built.")
        pred = self._point_head(fused)
        if return_features:
            return {"pred": pred, "fused": fused, "q": q}
        return pred

