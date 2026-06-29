from __future__ import annotations

import copy
import os
import json
from typing import Dict, Optional

import torch
import torch.distributed as dist

import models
import utils
from trainers import register
from trainers.base_trainer import BaseTrainer

from diffusion.edm import EDMSchedule, TokenEDM, measure_sigma_data_from_manifest


def _read_manifest_shape(cfg) -> Dict[str, int]:
    """Read {G, K, C, L} from the training dataset's manifest, or {} if unavailable."""
    try:
        ds_cfg = cfg.get("datasets", {}).get("train", {})
        ds_args = ds_cfg.get("args", {}) or {}
        manifest_path = ds_args.get("manifest_path", None)
        if manifest_path is None:
            return {}
        manifest_path = str(manifest_path)
        for key in ["stage1_dir", "latents_subdir"]:
            val = cfg.get(key, None)
            if val is not None:
                manifest_path = manifest_path.replace(f"${{{key}}}", str(val))
        manifest_path = os.path.expanduser(os.path.expandvars(manifest_path))
        if not os.path.isfile(manifest_path):
            return {}
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        split = ds_args.get("split", "train")
        shape = manifest.get("splits", {}).get(split, {}).get("shape", {})
        return dict(shape) if shape else {}
    except Exception:
        return {}


@register("hip_token_dm_trainer")
class HipTokenDMTrainer(BaseTrainer):
    """
    DDP trainer for EDM diffusion over HiP token latents.

    Expected dataset items:
      - c: [B,L,C] (normalized) float16/32
      - p: [B,L,d] or [L,d]
    """

    def make_model(self, model_spec=None):
        # Infer model shape from manifest; inject into cfg.model.args (and into model_spec for resumes).
        shape = _read_manifest_shape(self.cfg)
        if shape:
            model_cfg = self.cfg.get("model", {})
            model_args = dict(model_cfg.get("args", {}) or {})
            model_name = str(model_cfg.get("name", ""))

            if model_name == "hip_dit":
                G = shape.get("G")
                K = shape.get("K")
                C = shape.get("C")
                flatten_groups = self.cfg.get("datasets", {}).get("train", {}).get("args", {}).get("flatten_groups", False)

                if G is not None and model_args.get("num_groups") in (None, "from_manifest"):
                    model_args["num_groups"] = int(G)
                if K is not None and model_args.get("tokens_per_group") in (None, "from_manifest"):
                    model_args["tokens_per_group"] = int(K)
                if C is not None and model_args.get("token_dim") in (None, "from_manifest"):
                    if flatten_groups:
                        model_args["token_dim"] = int(K) * int(C)
                    else:
                        model_args["token_dim"] = int(C)

                model_cfg["args"] = model_args
                self.cfg["model"] = model_cfg
                self.log(f"[dm_trainer] Auto-detected shape from manifest: G={G}, K={K}, C={C}, flatten_groups={flatten_groups}")

            # Back-compat: sanitize old checkpoints that still have "from_manifest" placeholders
            # in ckpt["model"]["args"]; required for eval_only loads (we now also strip these in save_ckpt).
            try:
                if isinstance(model_spec, dict) and ("name" in model_spec) and ("args" in model_spec):
                    ms_name = str(model_spec.get("name", ""))
                    ms_args = dict(model_spec.get("args", {}) or {})
                    flatten_groups = self.cfg.get("datasets", {}).get("train", {}).get("args", {}).get("flatten_groups", False)
                    G = shape.get("G")
                    K = shape.get("K")
                    C = shape.get("C")

                    if ms_name == "hip_dit":
                        if (G is not None) and (ms_args.get("num_groups") in (None, "from_manifest")):
                            ms_args["num_groups"] = int(G)
                        if (K is not None) and (ms_args.get("tokens_per_group") in (None, "from_manifest")):
                            ms_args["tokens_per_group"] = int(K)
                        if (C is not None) and (ms_args.get("token_dim") in (None, "from_manifest")):
                            ms_args["token_dim"] = int(K) * int(C) if flatten_groups else int(C)
                        model_spec["args"] = ms_args
            except Exception:
                pass

        super().make_model(model_spec=model_spec)

        dm_cfg = self.cfg.get("dm", {}) or {}
        # EDM preconditioning. The model's time embedder MUST be configured for continuous
        # c_noise input (HiPDiT: time_input='edm_cnoise'); otherwise training silently fails.
        time_input = str(self.cfg.get("model", {}).get("args", {}).get("time_input", "discrete")).lower().strip()
        if time_input != "edm_cnoise":
            raise ValueError(
                "EDM requires model.args.time_input='edm_cnoise' so the time "
                f"embedder handles continuous c_noise input (got time_input={time_input!r})."
            )
        sd_cfg = dm_cfg.get("sigma_data", 1.0)
        if sd_cfg is None:
            raise ValueError("dm.sigma_data is None; set it to 'auto' or a positive float.")
        if isinstance(sd_cfg, str) and sd_cfg.lower().strip() == "auto":
            manifest_path = str(self.cfg.datasets.train.args.manifest_path)
            manifest_path = os.path.expanduser(os.path.expandvars(manifest_path))
            ds_norm_scale = float(self.cfg.datasets.train.args.get("norm_scale", 1.0))
            sigma_data = measure_sigma_data_from_manifest(
                manifest_path, split="train", max_shards=2, norm_scale=ds_norm_scale,
            )
            self.log(f"[edm] measured sigma_data={sigma_data:.6f} (norm_scale={ds_norm_scale}, manifest={manifest_path})")
            # Persist the resolved value so the ckpt cfg has a numeric value (sample_trainer
            # rejects 'auto' at sample time, since we can't re-measure post-hoc safely).
            try:
                self.cfg.dm.sigma_data = float(sigma_data)
                if isinstance(self.cfg_dict, dict):
                    self.cfg_dict.setdefault("dm", {})["sigma_data"] = float(sigma_data)
            except Exception:
                pass
        else:
            sigma_data = float(sd_cfg)
            self.log(f"[edm] using configured sigma_data={sigma_data:.6f}")
        self.ddpm = TokenEDM(EDMSchedule(
            sigma_data=sigma_data,
            P_mean=float(dm_cfg.get("P_mean", -1.2)),
            P_std=float(dm_cfg.get("P_std", 1.2)),
            sigma_min=float(dm_cfg.get("sigma_min", 0.002)),
            sigma_max=float(dm_cfg.get("sigma_max", 80.0)),
            rho=float(dm_cfg.get("rho", 7.0)),
        ))

        self.ema_enable = bool(dm_cfg.get("ema_enable", True))
        self.ema_rate = float(dm_cfg.get("ema_rate", 0.9999))
        if self.ema_enable:
            self.model_ema = copy.deepcopy(self.model).eval().to(self.device)
            for p in self.model_ema.parameters():
                p.requires_grad_(False)
        else:
            self.model_ema = None

        # BaseTrainer resumes only model + optimizers, not EMA. Restore EMA here.
        try:
            resume_file = os.path.join(self.cfg._env.save_dir, "last-model.pth")
            if self.ema_enable and self.model_ema is not None and os.path.isfile(resume_file):
                ckpt = torch.load(resume_file, map_location="cpu", weights_only=False)
                ema_sd = ckpt.get("ema_sd", None)
                if isinstance(ema_sd, dict) and len(ema_sd) > 0:
                    self.model_ema.load_state_dict(ema_sd, strict=False)
        except Exception:
            pass

    def make_optimizers(self):
        self.optimizers = {}
        opt_cfg = self.cfg.get("optimizers", {}) or {}
        spec = opt_cfg.get("dm", None)
        if spec is None:
            raise ValueError("cfg.optimizers.dm is required for hip_token_dm_trainer.")
        params = list(self.model.parameters())
        if not params:
            raise RuntimeError("No parameters found in diffusion model.")
        self.optimizers["dm"] = utils.make_optimizer(params, spec)

    def train_step(self, data: Dict[str, torch.Tensor], bp: bool = True):
        c = data.get("c", None)
        p = data.get("p", None)
        if c is None or p is None:
            raise ValueError("Batch must contain keys 'c' and 'p'.")
        if c.ndim != 3:
            raise ValueError(f"c must be [B,L,C], got {tuple(c.shape)}")
        B = int(c.shape[0])

        eval_with_ema = (not bp) and self.ema_enable and (self.model_ema is not None)
        sigma = self.ddpm.sched.sample_training_sigma(B, c.device, dtype=torch.float32)
        noise = torch.randn_like(c.float()) if eval_with_ema else None

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            loss, _x0_hat = self.ddpm.training_loss(
                self.model_ddp, x0=c.float(), p=p, sigma=sigma, noise=noise,
            )

        ret = {"loss": float(loss.detach().item()), "dm_loss": float(loss.detach().item())}

        # `val/dm_loss` is on training weights; sampling/FID uses EMA. Also log `dm_loss_ema`
        # so `ckpt_select_metric.name: dm_loss_ema` tracks generation quality.
        if eval_with_ema:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=self.use_amp):
                loss_ema, _ = self.ddpm.training_loss(
                    self.model_ema, x0=c.float(), p=p, sigma=sigma, noise=noise,
                )
            ret["loss_ema"] = float(loss_ema.detach().item())
            ret["dm_loss_ema"] = float(loss_ema.detach().item())
            # Diagnostic #3: ema_minus_train. Positive-and-growing => EMA decay too aggressive.
            ret["ema/dm_loss_minus_train"] = float(loss_ema.detach().item()) - float(loss.detach().item())

        # Diagnostic #1: loss bucketed by sigma/t. Gated to every N iters to keep overhead low.
        diag_every = int(self.cfg.get("dm", {}).get("diag_every_iters", 50) or 0)
        diag_enable = bool(self.cfg.get("dm", {}).get("diag_enable", True))
        if (
            bp
            and diag_enable
            and diag_every > 0
            and int(getattr(self, "iter", 0)) % diag_every == 0
        ):
            try:
                self._train_diag_bucket_loss(c=c.float(), p=p, ret=ret)
            except Exception as e:
                if self.is_master:
                    self.log(f"[diag] bucket loss failed: {e}")

        if bp:
            self.model_ddp.zero_grad()
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizers["dm"])
            self.grad_scaler.update()

            if self.ema_enable and (self.model_ema is not None):
                with torch.no_grad():
                    msd = self.model.state_dict()
                    esd = self.model_ema.state_dict()
                    for k, v in esd.items():
                        if k in msd and torch.is_tensor(msd[k]) and torch.is_tensor(v):
                            v.copy_(v * self.ema_rate + msd[k].detach() * (1.0 - self.ema_rate))

        return ret

    @torch.no_grad()
    def _train_diag_bucket_loss(self, *, c: torch.Tensor, p: torch.Tensor, ret: Dict[str, float]) -> None:
        """Diagnostic #1: per-sample loss bucketed by sigma/t.

        Runs an extra forward pass on the EMA model (no grad) with a fresh sample of
        (t/sigma, noise), recovers per-sample loss, and logs bucketed means. Catches
        per-noise-regime under/over-fitting that the scalar `dm_loss` hides.
        """
        if not getattr(self, "is_master", False):
            return
        m = self.model_ema if (self.ema_enable and getattr(self, "model_ema", None) is not None) else self.model
        B = int(c.shape[0])
        device = c.device

        sigma = self.ddpm.sched.sample_training_sigma(B, device, dtype=torch.float32)
        noise = torch.randn_like(c)
        x_noisy = c + noise * sigma
        x0_hat, _F = self.ddpm.precond_forward(m, x_noisy=x_noisy, sigma=sigma, p=p)
        per_mse = (x0_hat - c).pow(2).mean(dim=(1, 2))  # [B]
        w = self.ddpm.sched.loss_weight(sigma).view(B)
        per = (per_mse * w).detach().cpu().float()
        sd = self.ddpm.sched
        x = sigma.detach().log().view(B).cpu()
        # 5 bins covering ±3*P_std around P_mean.
        lo = sd.P_mean - 3.0 * sd.P_std
        hi = sd.P_mean + 3.0 * sd.P_std
        import math as _math
        edges = [-_math.inf] + [lo + (i + 1) * (hi - lo) / 5.0 for i in range(4)] + [_math.inf]
        names = ["s_lt-3", "s_-3_-1.4", "s_-1.4_0.2", "s_0.2_1.8", "s_gt1.8"]

        for i in range(len(names)):
            mask = (x >= edges[i]) & (x < edges[i + 1])
            if int(mask.sum()) == 0:
                continue
            v = float(per[mask].mean().item())
            ret[f"diag/loss_bucket/{names[i]}"] = v

    def visualize(self):
        """Sample latents, render via stage-1, log image grid. Best-effort, master-only."""
        sv = self.cfg.get("sample_vis", None)
        if sv is None or (not bool(getattr(sv, "enable", False))):
            return
        if not self.is_master:
            if self.distributed:
                dist.barrier()
            return

        try:
            self._visualize_impl(sv)
        except Exception as e:
            self.log(f"[dm_vis] failed: {e}")
        finally:
            if self.distributed:
                dist.barrier()
        return

    @torch.no_grad()
    def _visualize_impl(self, sv):
        import torchvision
        from utils.geometry import make_coord_cell_grid

        def _read_json(path: str):
            p = os.path.expanduser(os.path.expandvars(str(path)))
            with open(p, "r") as f:
                return json.load(f)

        stage1_ckpt = getattr(sv, "stage1_ckpt", None)
        latents_manifest = getattr(sv, "latents_manifest", None)
        latents_split = str(getattr(sv, "latents_split", "train"))
        input_res = int(getattr(sv, "input_res", 32))
        render_res = int(getattr(sv, "render_res", 32))
        n_samples = int(getattr(sv, "n_samples", 16))
        steps = int(getattr(sv, "steps", 18))
        sampler = str(getattr(sv, "sampler", "edm_heun")).lower().strip()
        if sampler != "edm_heun":
            raise ValueError(f"sample_vis.sampler={sampler!r} not valid; expected 'edm_heun'")
        tag = str(getattr(sv, "tag", "samples"))
        use_ema = bool(getattr(sv, "use_ema", True))

        if stage1_ckpt is None or latents_manifest is None:
            self.log("[dm_vis] sample_vis enabled but stage1_ckpt/latents_manifest not set; skipping visualization.")
            return

        # Use no_grad (not inference_mode) to avoid "inference tensor" edge-cases when EMA is off
        # and we reuse the training model.
        with torch.no_grad():
            manifest = _read_json(latents_manifest)
            split = manifest.get("splits", {}).get(latents_split, None)
            if split is None:
                raise KeyError(f"latents_split={latents_split!r} not found in manifest.")
            shape = split.get("shape", {}) or {}
            G = int(shape.get("G"))
            K = int(shape.get("K"))
            C_orig = int(shape.get("C"))  # channels before flatten
            L_orig = G * K  # token count before flatten

            flatten_groups = False
            try:
                flatten_groups = bool(self.cfg.datasets.train.args.get("flatten_groups", False))
            except Exception:
                flatten_groups = False

            if flatten_groups:
                L = G
                C = K * C_orig
            else:
                L = L_orig
                C = C_orig

            shard0 = (split.get("shards") or [None])[0]
            if shard0 is None:
                raise ValueError("manifest split has no shards.")
            shard_payload = torch.load(str(shard0), map_location="cpu", weights_only=False)
            p_tokens = shard_payload["p"].to(dtype=torch.float32, device=self.device)  # [L_orig, d]

            # Augment p with per-group lambda_g for HiPDiT slot-position / FiLM conditioning.
            try:
                use_slot = bool(getattr(self.model, "use_within_group_slot_pos", False))
            except Exception:
                use_slot = False
            if use_slot and (not flatten_groups):
                gs = shard_payload.get("group_scales", None)
                if gs is None:
                    gs_list = split.get("group_scales", None)
                    if gs_list is not None:
                        gs = torch.tensor(gs_list, dtype=torch.float32)
                if not torch.is_tensor(gs) or gs.ndim != 2:
                    raise ValueError(
                        "HiPDiT(use_within_group_slot_pos=True) requires group_scales [G,d] "
                        "in the extraction outputs. Re-run extraction with the updated extractor."
                    )
                gs = gs.to(dtype=torch.float32, device=self.device).clamp_min(1e-6)  # [G,d]
                if int(gs.shape[0]) != int(G):
                    raise ValueError(f"group_scales has G={int(gs.shape[0])} but expected G={int(G)}")
                gs_rep = gs.repeat_interleave(int(K), dim=0).contiguous()  # [L_orig,d]
                p_tokens = torch.cat([p_tokens, gs_rep], dim=-1).contiguous()  # [L_orig,2d]

            if flatten_groups:
                # Mean position per group; works whether p is repeated or distinct within group.
                if int(p_tokens.shape[0]) == int(G * K):
                    p_tokens = p_tokens.view(G, K, -1).mean(dim=1).contiguous()  # [G, d]
                else:
                    p_tokens = p_tokens[::K].contiguous()  # [G, d] fallback

            c_real0 = shard_payload.get("c", None)
            if (not torch.is_tensor(c_real0)) or c_real0.ndim != 3:
                raise ValueError(f"Shard payload missing c [N,L,C], got {type(c_real0)} shape={getattr(c_real0,'shape',None)}")
            c_real0 = c_real0.to(dtype=torch.float32, device=self.device)
            if int(c_real0.shape[0]) < int(n_samples):
                raise ValueError(f"Shard has only N={int(c_real0.shape[0])} items; need n_samples={int(n_samples)} for vis.")
            c_real0 = c_real0[: int(n_samples)]  # [n_samples, L_orig, C_orig] (unnormalized)

            mean_raw = torch.tensor(split.get("mean", []), dtype=torch.float32, device=self.device)
            std_raw = torch.tensor(split.get("std", []), dtype=torch.float32, device=self.device).clamp_min(1e-6)
            if mean_raw.ndim == 1:
                mean = mean_raw.view(1, 1, C_orig)
                std = std_raw.view(1, 1, C_orig)
            elif mean_raw.ndim == 2:
                if flatten_groups:
                    raise ValueError("sample_vis does not support flatten_groups with token_channel mean/std.")
                if int(mean_raw.shape[0]) != int(L_orig) or int(mean_raw.shape[1]) != int(C_orig):
                    raise ValueError(f"mean/std shape mismatch: got {tuple(mean_raw.shape)} expected {(L_orig, C_orig)}")
                mean = mean_raw.view(1, L_orig, C_orig)
                std = std_raw.view(1, L_orig, C_orig)
            else:
                raise ValueError(f"Unsupported mean/std shape in manifest: mean.ndim={mean_raw.ndim}")

            # Invert dataset norm_scale at denorm time.
            norm_scale = 1.0
            try:
                norm_scale = float(self.cfg.datasets.train.args.get("norm_scale", 1.0))
            except Exception:
                norm_scale = 1.0

            if flatten_groups:
                c_real0 = c_real0.view(n_samples, G, K, C_orig).reshape(n_samples, G, K * C_orig)

            if getattr(self, "_vis_stage1", None) is None:
                ckpt = torch.load(os.path.expanduser(os.path.expandvars(str(stage1_ckpt))), map_location="cpu", weights_only=False)
                spec = ckpt["model"]
                stage1 = models.make(spec, load_sd=True).to(self.device).eval()
                for p in stage1.parameters():
                    p.requires_grad_(False)
                # Routing metadata must be populated at the stage-1 input grid resolution.
                coord_in, _cell_in = make_coord_cell_grid((input_res, input_res), device=self.device, bs=1)
                coord_in = coord_in.reshape(1, -1, 2).to(dtype=torch.float32)
                value_in = torch.zeros((1, coord_in.shape[1], 3), device=self.device, dtype=torch.float32)
                _ = stage1({"inp": {"coord": coord_in, "value": value_in}}, mode="encode")
                self._vis_stage1 = stage1
            stage1 = self._vis_stage1

        enc_regions = getattr(getattr(stage1, "encoder", None), "encoder_group_regions", None)
        if enc_regions is None:
            raise RuntimeError("Stage-1 encoder did not populate encoder_group_regions; cannot render visualization.")
        blocks = enc_regions.get("blocks", {}) or {}
        if len(blocks) != 1:
            raise ValueError("sample_vis currently supports single-block enc_regions (render_block_index=-1 stage-1 configs).")
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

        block_info = _expand_batch(block_info0, int(n_samples))

        dm = self.model_ema if (use_ema and getattr(self, "model_ema", None) is not None) else self.model
        dm.eval()
        c_norm = self.ddpm.sample_heun(
            dm,
            p=p_tokens,
            shape=(n_samples, L, C),
            num_steps=steps,
            device=self.device,
        )
        
        if flatten_groups:
            # c_norm: [n_samples, G, K*C_orig] -> unnormalize as 4D
            c_4d = c_norm.view(n_samples, G, K, C_orig)
            mean_4d = mean.view(1, 1, 1, C_orig)
            std_4d = std.view(1, 1, 1, C_orig)
            c = c_4d * (std_4d * float(norm_scale)) + mean_4d
            enc_blocks = c.contiguous()  # [n_samples, G, K, C_orig]

            c_real0_4d = c_real0.view(n_samples, G, K, C_orig)
            enc_blocks_real = c_real0_4d.contiguous()  # unnormalized [n_samples, G, K, C_orig]

            x0_real = ((c_real0_4d - mean_4d) / (std_4d * float(norm_scale))).reshape(n_samples, G, K * C_orig)
        else:
            c = c_norm * (std * float(norm_scale)) + mean
            enc_blocks = c.view(n_samples, G, K, C_orig).contiguous()
            enc_blocks_real = c_real0.view(n_samples, G, K, C_orig).contiguous()
            x0_real = (c_real0 - mean) / (std * float(norm_scale))

        # Sanity stats: do sampled latents match the real-latent scale?
        with torch.no_grad():
            gen_mu = c.mean().item()
            gen_std = c.std().item()
            real_mu = c_real0.mean().item()
            real_std = c_real0.std().item()
            self.log_scalar("dm_vis/latent_mean_gen", float(gen_mu))
            self.log_scalar("dm_vis/latent_std_gen", float(gen_std))
            self.log_scalar("dm_vis/latent_mean_real", float(real_mu))
            self.log_scalar("dm_vis/latent_std_real", float(real_std))

            # Diagnostic #5: per-channel mean/std max-deviation gen-vs-real. Catches
            # channel-wise mode collapse / scale drift hidden by global stats.
            try:
                # c and c_real0 are both [N, ...] with leading batch dim; reduce all but last C-axis.
                gen_flat = c.reshape(-1, c.shape[-1]).float()
                real_flat = c_real0.reshape(-1, c_real0.shape[-1]).float()
                ch_mean_diff = (gen_flat.mean(0) - real_flat.mean(0)).abs().max()
                ch_std_ratio = (gen_flat.std(0) + 1e-8) / (real_flat.std(0) + 1e-8)
                ch_std_log_diff = ch_std_ratio.log().abs().max()
                self.log_scalar("dm_vis/ch_mean_max_abs_diff", float(ch_mean_diff))
                self.log_scalar("dm_vis/ch_std_log_ratio_max", float(ch_std_log_diff))
            except Exception:
                pass

            # Diagnostic #7: fraction of sample tokens outside [-6, 6]*scale.
            try:
                scale = float(getattr(getattr(self.ddpm, "sched", None), "sigma_data", 1.0))
            except Exception:
                scale = 1.0
            try:
                extreme_frac = float((c_norm.abs() > 6.0 * scale).float().mean().item())
                self.log_scalar("dm_vis/sampler_extreme_frac", extreme_frac)
                if not torch.isfinite(c_norm).all():
                    self.log_scalar("dm_vis/sampler_nan_or_inf", 1.0)
                else:
                    self.log_scalar("dm_vis/sampler_nan_or_inf", 0.0)
            except Exception:
                pass

            # Diagnostic #6: time-embedder sensitivity to sigma extremes.
            try:
                sd = self.ddpm.sched
                sig_lo = torch.tensor([sd.sigma_min], device=self.device)
                sig_hi = torch.tensor([sd.sigma_max], device=self.device)
                cn_lo = sd.c_noise(sig_lo).view(-1)
                cn_hi = sd.c_noise(sig_hi).view(-1)
                emb_lo = dm.t_embed(cn_lo)
                emb_hi = dm.t_embed(cn_hi)
                rel = (emb_hi - emb_lo).norm() / (emb_lo.norm().clamp_min(1e-8))
                self.log_scalar("dm_vis/edm/time_embed_sensitivity", float(rel.item()))
            except Exception:
                pass

            # Diagnostic #4: re-measure sigma_data on shard0 and log residual vs cfg value.
            try:
                ds_norm_scale = 1.0
                try:
                    ds_norm_scale = float(self.cfg.datasets.train.args.get("norm_scale", 1.0))
                except Exception:
                    pass
                mp = str(self.cfg.datasets.train.args.manifest_path)
                mp = os.path.expanduser(os.path.expandvars(mp))
                sd_now = measure_sigma_data_from_manifest(mp, split="train", max_shards=1, norm_scale=ds_norm_scale)
                self.log_scalar("dm_vis/edm/sigma_data_measured", float(sd_now))
                self.log_scalar(
                    "dm_vis/edm/sigma_data_residual",
                    float(abs(sd_now - float(self.ddpm.sched.sigma_data))),
                )
            except Exception:
                pass

        H = W = int(render_res)
        coord, cell = make_coord_cell_grid((H, W), device=self.device, bs=n_samples)
        coord = coord.reshape(n_samples, -1, 2).to(dtype=torch.float32)
        cell = cell.reshape(n_samples, -1, 2).to(dtype=torch.float32)
        pred = stage1.renderer(
            z_dec=torch.zeros((n_samples, 1, 1, 1), device=self.device),
            coord=coord,
            cell=cell,
            enc_blocks={block_key: enc_blocks},
            enc_regions={
                "routing_space": "coord",
                "coord_dim": enc_regions.get("coord_dim", 2),
                "blocks": {block_key: block_info},
            },
        )
        img = pred.view(n_samples, H, W, 3).permute(0, 3, 1, 2).contiguous()
        img01 = (img * 0.5 + 0.5).clamp(0, 1)

        grid = torchvision.utils.make_grid(img01, nrow=int(getattr(sv, "nrow", 8)), padding=2)
        self.log_image(f"dm_vis/{tag}", grid)

        # Render real latents too as a renderer-path sanity check.
        pred_real = stage1.renderer(
            z_dec=torch.zeros((n_samples, 1, 1, 1), device=self.device),
            coord=coord,
            cell=cell,
            enc_blocks={block_key: enc_blocks_real},
            enc_regions={
                "routing_space": "coord",
                "coord_dim": enc_regions.get("coord_dim", 2),
                "blocks": {block_key: block_info},
            },
        )
        img_real = pred_real.view(n_samples, H, W, 3).permute(0, 3, 1, 2).contiguous()
        img_real01 = (img_real * 0.5 + 0.5).clamp(0, 1)
        grid_real = torchvision.utils.make_grid(img_real01, nrow=int(getattr(sv, "nrow", 8)), padding=2)
        self.log_image("dm_vis/real_latents", grid_real)

        # Save grids to disk too.
        try:
            out_dir = str(getattr(sv, "out_dir", "dm_vis") or "dm_vis")
        except Exception:
            out_dir = "dm_vis"
        out_dir = out_dir.strip()
        if out_dir == "":
            out_dir = "dm_vis"
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(self.cfg._env.save_dir, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.abspath(os.path.join(out_dir, f"{tag}_iter{int(self.iter):07d}.png"))
        out_path_real = os.path.abspath(os.path.join(out_dir, f"real_latents_iter{int(self.iter):07d}.png"))
        try:
            torchvision.utils.save_image(grid, out_path)
            torchvision.utils.save_image(grid_real, out_path_real)
        except Exception:
            # If grid save fails, write the first image.
            torchvision.utils.save_image(img01[0], out_path)

        self.log(f"[dm_vis] saved: {out_path}")
        self.log_buffer.append(f"dm_vis/{tag}={out_path}")
        self.log(f"[dm_vis] saved: {out_path_real}")
        self.log_buffer.append(f"dm_vis/real_latents={out_path_real}")

    def save_ckpt(self, filename: str):
        """Save checkpoint with EMA weights under `ema_sd` (in addition to base payload)."""
        if not self.is_master:
            return

        import copy as _copy
        from omegaconf import OmegaConf as _OmegaConf
        # Use resolved cfg.model so we don't write "from_manifest" placeholders into ckpt.
        model_spec = _copy.deepcopy(_OmegaConf.to_container(self.cfg.model, resolve=True))
        model_spec["sd"] = self.model.state_dict()

        optimizers_spec = {}
        cfg_opt = self.cfg_dict.get("optimizers", {}) or {}
        for k, opt in self.optimizers.items():
            spec = _copy.copy(cfg_opt.get(k, {}))
            spec["sd"] = opt.state_dict()
            optimizers_spec[k] = spec

        ckpt = {
            "cfg": self.cfg_dict,
            "model": model_spec,
            "optimizers": optimizers_spec,
            "iter": self.iter,
            "train_loader_epoch": self.train_loader_epoch,
            "ckpt_select_v": self.ckpt_select_v,
            "grad_scaler": self.grad_scaler.state_dict() if self.use_amp else None,
        }

        if self.ema_enable and self.model_ema is not None:
            ckpt["ema_sd"] = self.model_ema.state_dict()

        torch.save(ckpt, os.path.join(self.cfg._env.save_dir, filename))

