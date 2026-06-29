from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torchvision
from tqdm import tqdm

import models
from trainers import register
from trainers.base_trainer import BaseTrainer
from utils.geometry import make_coord_cell_grid

from diffusion.edm import EDMSchedule, TokenEDM


def _read_json(path: str) -> Dict[str, Any]:
    p = Path(os.path.expanduser(os.path.expandvars(str(path)))).resolve()
    return json.loads(p.read_text())


def _expand_path(p: str) -> str:
    return os.path.expanduser(os.path.expandvars(str(p)))


@register("hip_token_sample_trainer")
class HipTokenSampleTrainer(BaseTrainer):
    """
    DDP-safe sampler: sample token latents from a DM, render with a frozen stage-1 LH-NeF.
    """

    def run(self):
        # BaseTrainer.run() normally sets self.iter; this trainer skips the training loop,
        # so initialize it manually for stable TB/W&B step counts.
        if not hasattr(self, "iter"):
            self.iter = 0

        if self.cfg.random_seed is not None:
            self.seed_everything(self.cfg.random_seed, rank_shift=True)

        self._sample()

        if self.enable_tb:
            self.writer.close()
        if self.enable_wandb:
            import wandb
            wandb.finish()

    def _load_stage1(self, ckpt_path: str, *, input_res: int):
        ckpt_path = _expand_path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        spec = ckpt["model"]
        net = models.make(spec, load_sd=True).to(self.device).eval()
        for p in net.parameters():
            p.requires_grad_(False)
        # Encoder routing metadata must be populated at the stage-1 input grid resolution.
        coord_in, _ = make_coord_cell_grid((int(input_res), int(input_res)), device=self.device, bs=1)
        coord_in = coord_in.reshape(1, -1, 2).to(dtype=torch.float32)
        value_in = torch.zeros((1, coord_in.shape[1], 3), device=self.device, dtype=torch.float32)
        _ = net({"inp": {"coord": coord_in, "value": value_in}}, mode="encode")
        return net

    def _load_dm(self, dm_ckpt: str, *, use_ema: bool):
        dm_ckpt = _expand_path(dm_ckpt)
        ckpt = torch.load(dm_ckpt, map_location="cpu", weights_only=False)
        spec = ckpt["model"]
        dm = models.make(spec, load_sd=True).to(self.device).eval()
        for p in dm.parameters():
            p.requires_grad_(False)
        if use_ema:
            ema_sd = ckpt.get("ema_sd", None)
            if isinstance(ema_sd, dict) and len(ema_sd) > 0:
                dm.load_state_dict(ema_sd, strict=False)
        return dm, ckpt

    @torch.no_grad()
    def _sample(self):
        cfg = self.cfg
        scfg = cfg.get("sample", None)
        if scfg is None:
            raise ValueError("sample_trainer requires cfg.sample.*")

        # Render reconstructions from real cached latents (skip diffusion). Useful as an
        # approximate upper bound on achievable image-space FID given the stage-1 latent space.
        use_real_latents = bool(getattr(scfg, "use_real_latents", False))

        dm_ckpt = getattr(scfg, "dm_ckpt", None)
        stage1_ckpt = getattr(scfg, "stage1_ckpt", None)
        latents_manifest = getattr(scfg, "latents_manifest", None)
        if stage1_ckpt is None or latents_manifest is None:
            raise ValueError("sample requires stage1_ckpt and latents_manifest")
        if (not use_real_latents) and dm_ckpt is None:
            raise ValueError("sample requires dm_ckpt unless sample.use_real_latents=true")

        latents_split = str(getattr(scfg, "latents_split", "train"))
        input_res = int(getattr(scfg, "input_res", 32))
        render_res = int(getattr(scfg, "render_res", 32))
        n_samples = int(getattr(scfg, "n_samples", 50000))
        batch_size = int(getattr(scfg, "batch_size", 32))
        steps = int(getattr(scfg, "steps", 18))
        sampler = str(getattr(scfg, "sampler", "edm_heun")).lower().strip()
        if sampler != "edm_heun":
            raise ValueError(f"sample.sampler must be 'edm_heun', got {sampler!r}")
        use_ema = bool(getattr(scfg, "use_ema", True))
        outdir = str(getattr(scfg, "outdir", "samples"))
        save_grid = bool(getattr(scfg, "save_grid", True))
        grid_n = int(getattr(scfg, "grid_n", 16))
        save_backend = str(getattr(scfg, "save_backend", "png")).lower().strip()  # png|save_image
        show_progress = bool(getattr(scfg, "progress", True))
        progress_every = int(getattr(scfg, "progress_every", 1))
        progress_every = max(1, progress_every)

        outdir_exp = _expand_path(outdir)
        if not os.path.isabs(outdir_exp):
            outdir_exp = os.path.join(cfg._env.save_dir, outdir_exp)
        os.makedirs(outdir_exp, exist_ok=True)
        if self.distributed:
            dist.barrier()

        manifest = _read_json(latents_manifest)
        split = manifest.get("splits", {}).get(latents_split, None)
        if split is None:
            raise KeyError(f"latents_split={latents_split!r} not found in manifest.")
        shape = split.get("shape", {}) or {}
        G = int(shape.get("G"))
        K = int(shape.get("K"))
        C = int(shape.get("C"))
        L = int(shape.get("L", G * K))
        L = G * K  # canonical

        # Stats layouts: [C] (channel) or [L,C] (per-token-per-channel).
        mean_t = torch.tensor(split.get("mean", []), dtype=torch.float32, device=self.device)
        std_t = torch.tensor(split.get("std", []), dtype=torch.float32, device=self.device)
        if mean_t.ndim == 1:
            if int(mean_t.numel()) == int(C):
                mean = mean_t.view(1, 1, C)
                std = std_t.view(1, 1, C)
            elif int(mean_t.numel()) == int(L * C):
                mean = mean_t.view(1, L, C)
                std = std_t.view(1, L, C)
            else:
                raise ValueError(f"Unsupported mean/std length: mean={int(mean_t.numel())} (expected C={C} or L*C={L*C})")
        elif mean_t.ndim == 2:
            if tuple(mean_t.shape) != (int(L), int(C)):
                raise ValueError(f"Unsupported mean/std shape: {tuple(mean_t.shape)} (expected {(L, C)})")
            mean = mean_t.view(1, L, C)
            std = std_t.view(1, L, C)
        else:
            raise ValueError(f"Unsupported mean/std ndim: {mean_t.ndim}")
        std = std.clamp_min(1e-6)
        norm_scale = float(getattr(scfg, "norm_scale", 1.0))

        shard0 = (split.get("shards") or [None])[0]
        if shard0 is None:
            raise ValueError("manifest split has no shards.")
        shard_payload = torch.load(str(shard0), map_location="cpu", weights_only=False)
        p_tokens = shard_payload["p"].to(dtype=torch.float32, device=self.device)  # [L,d]

        stage1 = self._load_stage1(stage1_ckpt, input_res=input_res)
        dm = None
        dm_ckpt_obj = None
        if not use_real_latents:
            dm, dm_ckpt_obj = self._load_dm(dm_ckpt, use_ema=use_ema)

        # Augment p with per-group lambda_g for HiPDiT slot-position / FiLM conditioning.
        use_slot = False
        if dm is not None:
            use_slot = bool(getattr(dm, "use_within_group_slot_pos", False))
        if use_slot:
            gs = shard_payload.get("group_scales", None)
            if gs is None:
                gs_list = split.get("group_scales", None)
                if gs_list is not None:
                    gs = torch.tensor(gs_list, dtype=torch.float32)
            if (not torch.is_tensor(gs)) or gs.ndim != 2:
                raise ValueError(
                    "HiPDiT(use_within_group_slot_pos=True) requires group_scales [G,d] "
                    "in the extraction outputs. Re-run extraction with the updated extractor."
                )
            gs = gs.to(dtype=torch.float32, device=self.device).clamp_min(1e-6)  # [G,d]
            if int(gs.shape[0]) != int(G):
                raise ValueError(f"group_scales has G={int(gs.shape[0])} but expected G={int(G)}")
            gs_rep = gs.repeat_interleave(int(K), dim=0).contiguous()  # [L,d]
            p_tokens = torch.cat([p_tokens, gs_rep], dim=-1).contiguous()  # [L,2d]

        ddpm = None
        if not use_real_latents:
            ckpt_dm_cfg = (dm_ckpt_obj.get("cfg", {}) or {}).get("dm", {}) or {}
            # Symmetric guard with dm_trainer: EDM requires the FourierTimeEmbedder.
            ckpt_model_args = (dm_ckpt_obj.get("model", {}) or {}).get("args", {}) or {}
            ti = str(ckpt_model_args.get("time_input", "discrete")).lower().strip()
            if ti != "edm_cnoise":
                raise ValueError(
                    f"EDM sampling requires the loaded model's time_input='edm_cnoise', got {ti!r}"
                )
            # sigma_data: never auto-measure at sample time. The ckpt was trained with a
            # specific value; we must use that value (otherwise preconditioning is wrong).
            sd_ckpt = ckpt_dm_cfg.get("sigma_data", None)
            sd_override = getattr(scfg, "sigma_data", None)
            if sd_override is not None:
                sigma_data = float(sd_override)
            elif sd_ckpt is not None and not (isinstance(sd_ckpt, str) and sd_ckpt.lower() == "auto"):
                sigma_data = float(sd_ckpt)
            else:
                raise ValueError(
                    "EDM sampling requires sigma_data in cfg.sample.sigma_data or the ckpt cfg "
                    "(value must be numeric, not 'auto')."
                )
            ddpm = TokenEDM(EDMSchedule(
                sigma_data=sigma_data,
                P_mean=float(getattr(scfg, "P_mean", None) or ckpt_dm_cfg.get("P_mean", -1.2)),
                P_std=float(getattr(scfg, "P_std", None) or ckpt_dm_cfg.get("P_std", 1.2)),
                sigma_min=float(getattr(scfg, "sigma_min", None) or ckpt_dm_cfg.get("sigma_min", 0.002)),
                sigma_max=float(getattr(scfg, "sigma_max", None) or ckpt_dm_cfg.get("sigma_max", 80.0)),
                rho=float(getattr(scfg, "rho", None) or ckpt_dm_cfg.get("rho", 7.0)),
            ))

        enc_regions = getattr(getattr(stage1, "encoder", None), "encoder_group_regions", None)
        if enc_regions is None:
            raise RuntimeError("Stage-1 encoder did not populate encoder_group_regions; cannot route tokens.")
        blocks = enc_regions.get("blocks", {}) or {}
        if len(blocks) != 1:
            raise ValueError("sample_trainer currently supports single-block enc_regions (render_block_index=-1 stage-1 configs).")
        block_key = next(iter(blocks.keys()))
        block_info0 = blocks[block_key]

        def _expand_batch(x, B: int):
            # Expand routing metadata from B=1 to desired B.
            if torch.is_tensor(x):
                if x.ndim >= 1 and int(x.shape[0]) == 1 and B > 1:
                    return x.expand(B, *x.shape[1:])
                return x
            if isinstance(x, dict):
                return {k: _expand_batch(v, B) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                t = [_expand_batch(v, B) for v in x]
                return type(x)(t)
            return x

        ws = int(self.world_size) if self.distributed else 1
        r = int(self.rank) if self.distributed else 0
        per_rank = int(math.ceil(float(n_samples) / float(ws)))
        start = r * per_rank
        end = min(n_samples, (r + 1) * per_rank)
        n_local = max(0, end - start)

        H = W = int(render_res)

        grid_imgs = []

        idx = 0
        pbar = None
        prev_global = 0
        if show_progress and self.is_master:
            pbar = tqdm(total=int(n_samples), desc="sampling", leave=True)

        def _progress_sync(local_done: int):
            nonlocal prev_global, pbar
            if not show_progress:
                return
            if not self.distributed:
                if pbar is not None:
                    pbar.update(int(local_done) - int(prev_global))
                    prev_global = int(local_done)
                return
            # All ranks must enter the all_reduce so rank0 sees global progress.
            t = torch.tensor(int(local_done), device=self.device, dtype=torch.int64)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            if pbar is not None:
                gdone = int(t.item())
                pbar.update(max(0, gdone - int(prev_global)))
                prev_global = gdone

        def _save_png_tensor(img01_bchw: torch.Tensor, out_path: str):
            # img01_bchw: [3,H,W] float in [0,1]
            if save_backend == "save_image":
                torchvision.utils.save_image(img01_bchw, out_path)
                return
            # torchvision.io.write_png on uint8 is typically faster than PIL-based save_image.
            from torchvision.io import write_png
            u8 = (img01_bchw.mul(255.0).round().clamp(0, 255)).to(dtype=torch.uint8, device="cpu")
            write_png(u8, out_path, compression_level=3)

        if use_real_latents:
            # Stream real (unnormalized) latents in deterministic shard order.
            shards = list(split.get("shards", []) or [])
            if not shards:
                raise ValueError("manifest split has no shards.")
            cum = 0
            for sp in shards:
                payload = torch.load(str(sp), map_location="cpu", weights_only=False)
                c_all = payload.get("c", None)
                if (not torch.is_tensor(c_all)) or c_all.ndim != 3:
                    raise ValueError(f"Shard {sp} missing c [N,L,C], got {type(c_all)} shape={getattr(c_all,'shape',None)}")
                n_sh = int(c_all.shape[0])
                sh_start = int(cum)
                sh_end = int(cum + n_sh)
                cum = sh_end

                a = max(int(start), sh_start)
                b = min(int(end), sh_end)
                if b <= a:
                    continue
                sl = c_all[int(a - sh_start) : int(b - sh_start)]  # [M,L,C] CPU

                off = 0
                while off < int(sl.shape[0]):
                    cur = min(int(batch_size), int(sl.shape[0]) - int(off))
                    c = sl[int(off) : int(off + cur)].to(device=self.device, dtype=torch.float32)  # [cur,L,C]
                    enc_blocks = c.view(cur, G, K, C).contiguous()

                    coord, cell = make_coord_cell_grid((H, W), device=self.device, bs=cur)
                    coord = coord.reshape(cur, -1, 2).to(dtype=torch.float32)
                    cell = cell.reshape(cur, -1, 2).to(dtype=torch.float32)
                    block_info = _expand_batch(block_info0, int(cur))
                    pred = stage1.renderer(
                        z_dec=torch.zeros((cur, 1, 1, 1), device=self.device),
                        coord=coord,
                        cell=cell,
                        enc_blocks={block_key: enc_blocks},
                        enc_regions={"routing_space": "coord", "coord_dim": enc_regions.get("coord_dim", 2), "blocks": {block_key: block_info}},
                    )
                    img = pred.view(cur, H, W, 3).permute(0, 3, 1, 2).contiguous()
                    img01 = (img * 0.5 + 0.5).clamp(0, 1)

                    for j in range(cur):
                        global_idx = a + off + j  # dataset-order index across concatenated shards
                        out_path = os.path.join(outdir_exp, f"{int(global_idx):06d}.png")
                        _save_png_tensor(img01[j], out_path)
                        if self.is_master and save_grid and len(grid_imgs) < grid_n:
                            grid_imgs.append(img01[j].detach().cpu())
                    idx += cur
                    off += cur
                    if (idx % progress_every) == 0 or idx >= n_local:
                        _progress_sync(idx)
                if idx >= n_local:
                    break
        else:
            while idx < n_local:
                cur = min(batch_size, n_local - idx)
                c_norm = ddpm.sample_heun(dm, p=p_tokens, shape=(cur, L, C), num_steps=steps, device=self.device)
                c = c_norm * (std * float(norm_scale)) + mean
                enc_blocks = c.view(cur, G, K, C).contiguous()

                coord, cell = make_coord_cell_grid((H, W), device=self.device, bs=cur)
                coord = coord.reshape(cur, -1, 2).to(dtype=torch.float32)
                cell = cell.reshape(cur, -1, 2).to(dtype=torch.float32)
                block_info = _expand_batch(block_info0, int(cur))
                pred = stage1.renderer(
                    z_dec=torch.zeros((cur, 1, 1, 1), device=self.device),
                    coord=coord,
                    cell=cell,
                    enc_blocks={block_key: enc_blocks},
                    enc_regions={"routing_space": "coord", "coord_dim": enc_regions.get("coord_dim", 2), "blocks": {block_key: block_info}},
                )
                img = pred.view(cur, H, W, 3).permute(0, 3, 1, 2).contiguous()
                img01 = (img * 0.5 + 0.5).clamp(0, 1)

                for j in range(cur):
                    global_idx = start + idx + j
                    out_path = os.path.join(outdir_exp, f"{global_idx:06d}.png")
                    _save_png_tensor(img01[j], out_path)
                    if self.is_master and save_grid and len(grid_imgs) < grid_n:
                        grid_imgs.append(img01[j].detach().cpu())
                idx += cur
                if (idx % progress_every) == 0 or idx >= n_local:
                    _progress_sync(idx)

        if self.distributed:
            dist.barrier()

        if pbar is not None:
            pbar.close()

        if self.is_master and save_grid and grid_imgs:
            grid = torchvision.utils.make_grid(torch.stack(grid_imgs, dim=0), nrow=int(math.sqrt(len(grid_imgs))) or 4)
            self.log_image("sample/grid", grid)

        # Rank0-only FID/IS/KID against a real-image folder.
        eval_cfg = getattr(scfg, "eval", None)
        if self.is_master and (eval_cfg is not None):
            enable = bool(getattr(eval_cfg, "enable", False))
            if enable:
                real_dir = getattr(eval_cfg, "real_dir", None)
                if real_dir is None:
                    raise ValueError("sample.eval.enable=true requires sample.eval.real_dir")

                real_dir_exp = _expand_path(real_dir)
                if not os.path.isabs(real_dir_exp):
                    real_dir_exp = os.path.join(cfg._env.save_dir, real_dir_exp)

                # Sanity: both dirs exist, both non-empty, and the resolutions match (FID assumes equal res).
                if not os.path.isdir(outdir_exp):
                    raise FileNotFoundError(f"Generated samples dir not found: {outdir_exp}")
                if not os.path.isdir(real_dir_exp):
                    raise FileNotFoundError(f"Real images dir not found: {real_dir_exp}")
                try:
                    gen_files = [f for f in os.listdir(outdir_exp) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                    real_files = [f for f in os.listdir(real_dir_exp) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                except Exception:
                    gen_files, real_files = [], []
                if len(gen_files) == 0:
                    raise RuntimeError(f"No images found in generated dir: {outdir_exp}")
                if len(real_files) == 0:
                    raise RuntimeError(f"No images found in real dir: {real_dir_exp}")
                try:
                    from PIL import Image
                    g0 = Image.open(os.path.join(outdir_exp, gen_files[0]))
                    r0 = Image.open(os.path.join(real_dir_exp, real_files[0]))
                    if g0.size != r0.size:
                        raise RuntimeError(f"Resolution mismatch: gen={g0.size} vs real={r0.size}. "
                                           f"Ensure real_dir images match render_res={render_res}.")
                except Exception as e:
                    self.log(f"[sample_eval] warning: could not verify image resolution: {e}")

                fid = bool(getattr(eval_cfg, "fid", True))
                isc = bool(getattr(eval_cfg, "isc", False))
                kid = bool(getattr(eval_cfg, "kid", False))
                cuda = bool(getattr(eval_cfg, "cuda", torch.cuda.is_available()))
                eval_bs = int(getattr(eval_cfg, "batch_size", 256))

                # Lazy import: sampling itself does not depend on torch_fidelity.
                import torch_fidelity  # type: ignore

                self.log(
                    f"[sample_eval] computing metrics: fid={fid} isc={isc} kid={kid} "
                    f"(cuda={cuda}, batch_size={eval_bs})"
                )
                self.log(f"[sample_eval] gen_dir={outdir_exp}")
                self.log(f"[sample_eval] real_dir={real_dir_exp}")

                ret = torch_fidelity.calculate_metrics(
                    input1=outdir_exp,
                    input2=real_dir_exp,
                    cuda=cuda,
                    fid=fid,
                    isc=isc,
                    kid=kid,
                    batch_size=eval_bs,
                )

                out_json = os.path.join(cfg._env.save_dir, "sample_eval_metrics.json")
                payload = {
                    "gen_dir": outdir_exp,
                    "real_dir": real_dir_exp,
                    "n_samples_requested": int(n_samples),
                    "ret": {},
                }
                for k, v in (ret or {}).items():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    payload["ret"][str(k)] = fv
                    try:
                        self.log_scalar(f"sample_eval/{str(k)}", fv)
                    except Exception:
                        pass
                try:
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, sort_keys=True)
                    self.log(f"[sample_eval] wrote {out_json}")
                except Exception:
                    pass

