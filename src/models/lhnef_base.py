"""Abstract autoencoder base composing encoder + optional decoder + renderer."""

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

import models


class LHNeFBase(nn.Module):

    @staticmethod
    def _offdiag_token_cos(tokens: torch.Tensor, max_tokens: int = 32) -> float:
        """
        Mean off-diagonal cosine similarity within each sample's token set, averaged across batch.
        tokens: [B, L, C]. 1.0 means tokens within a sample are (nearly) collapsed.
        Uses at most `max_tokens` tokens per sample for efficiency.
        """
        if (tokens is None) or (not torch.is_tensor(tokens)) or tokens.ndim != 3:
            return float("nan")
        B, L, C = tokens.shape
        if L < 2:
            return float("nan")
        with torch.no_grad():
            s = int(min(int(max_tokens), int(L)))
            if s < 2:
                return float("nan")
            if L > s:
                idx = torch.randperm(L, device=tokens.device)[:s]
                t = tokens[:, idx, :]
            else:
                t = tokens
            t = F.normalize(t, dim=-1)
            sim = t @ t.transpose(1, 2)  # [B,s,s]
            m = ~torch.eye(sim.size(-1), dtype=torch.bool, device=sim.device)
            return float(sim[:, m].mean().item())

    def __init__(self, encoder, z_shape, decoder, renderer, loss_cfg=None, **kwargs):
        super().__init__()
        # Resolve z_shape (OmegaConf ListConfig safe)
        zs = z_shape
        try:
            if hasattr(OmegaConf, 'is_config') and OmegaConf.is_config(zs):
                zs = OmegaConf.to_container(zs, resolve=True)
        except Exception:
            pass
        if not isinstance(zs, Sequence) or len(zs) != 3:
            raise AssertionError(f"Bad z_shape: {z_shape!r}")
        try:
            zc, zh, zw = map(int, zs)
        except Exception as e:
            raise AssertionError(f"z_shape must be ints, got {zs!r}") from e
        self.z_shape = (zc, zh, zw)

        self.encoder = models.make(encoder)
        self.decoder = models.make(decoder)  # may be None for pure coord/value pipelines
        self.renderer = models.make(renderer)

        # Renderer may need encoder-derived metadata (HiP block dims, base resolution).
        cfg_fn = getattr(self.renderer, "configure_from_encoder", None)
        if callable(cfg_fn):
            try:
                cfg_fn(self.encoder)
            except Exception as e:
                raise RuntimeError(f"Renderer configure_from_encoder failed: {e}") from e

        self.loss_cfg = loss_cfg if loss_cfg is not None else dict()

    def get_params(self, name):
        if name == 'encoder':
            return self.encoder.parameters()
        elif name == 'decoder':
            return list(self.decoder.parameters()) if (self.decoder is not None) else []
        elif name == 'renderer':
            return self.renderer.parameters()

    def run_renderer(self, z_dec, coord, cell):
        raise NotImplementedError

    def encode_z(self, x):
        # fp16 autocast can underflow in HiP's attention; force fp32
        if torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16:
            if torch.is_tensor(x):
                return self.encoder(x)
            return self.encoder(x)  # generic coord/value dict input
        with torch.amp.autocast('cuda', enabled=False):
            if torch.is_tensor(x):
                return self.encoder(x.float())
            return self.encoder(x)  # generic coord/value dict input

    def decode_z(self, z):
        return z if (self.decoder is None) else self.decoder(z)

    def forward(self, data, mode='loss', has_opt=None, **kwargs):
        inp = data['inp']

        if mode == 'encode':
            with torch.no_grad():
                z_pre = self.encode_z(inp)
                ret_encode = {'z_pre': z_pre}
                # Encoder-side intermediates consumed by Stage 2A/2B extract trainers.
                enc_blocks = getattr(self.encoder, "last_enc_blocks", None)
                if isinstance(enc_blocks, dict) and len(enc_blocks) > 0:
                    ret_encode["enc_blocks"] = enc_blocks
                enc_regions = getattr(self.encoder, "encoder_group_regions", None)
                if enc_regions is not None:
                    ret_encode["enc_regions"] = enc_regions
                return ret_encode

        gd = self.get_gd_from_opt(has_opt)
        ret = dict()

        if gd['encoder']:
            z = self.encode_z(inp)
        else:
            with torch.no_grad():
                z = self.encode_z(inp)

        # One-time latent shape check to catch encoder/config mismatches early.
        if not hasattr(self, "_zshape_checked"):
            try:
                assert z.shape[1:] == torch.Size(self.z_shape), \
                    f"Encoder z shape {tuple(z.shape[1:])} != z_shape {tuple(self.z_shape)}"
            finally:
                self._zshape_checked = True

        # Token-diversity diagnostic on HiP encoder block outputs (renderer's KV set).
        with torch.no_grad():
            enc_blocks = getattr(self.encoder, "last_enc_blocks", None)
            if isinstance(enc_blocks, dict) and len(enc_blocks) > 0:
                def _bi(name: str) -> int:
                    try:
                        return int(str(name)[5:]) if str(name).startswith("block") else 9999
                    except Exception:
                        return 9999
                keys = [k for k in enc_blocks.keys() if isinstance(k, str) and k.startswith("block")]
                keys = sorted(keys, key=_bi)
                pick = []
                if len(keys) >= 1:
                    pick.append(keys[0])
                if len(keys) >= 2:
                    pick.append(keys[-1])
                for k in pick:
                    v = enc_blocks.get(k, None)
                    if (not torch.is_tensor(v)) or (v.ndim != 4):
                        continue
                    Bv, G, Kt, Cv = v.shape
                    t_gm = v.mean(dim=2)
                    ret[f'hip_enc_blocks/{k}_groupmean_token_cos_offdiag'] = self._offdiag_token_cos(t_gm, max_tokens=32)
                    t_full = v.reshape(Bv, G * Kt, Cv)
                    ret[f'hip_enc_blocks/{k}_full_token_cos_offdiag'] = self._offdiag_token_cos(t_full, max_tokens=32)

        if mode == 'z_dec':
            if gd['decoder']:
                z_dec = self.decode_z(z)
            else:
                with torch.no_grad():
                    z_dec = self.decode_z(z)
            ret_z = z_dec
        elif mode == 'z':
            ret_z = z
        else:
            raise ValueError(f"Unknown mode={mode!r}")

        ret['loss'] = torch.tensor(0.0, dtype=torch.float32, device=z.device)
        return ret_z, ret

    def get_gd_from_opt(self, opt):
        if opt is None:
            opt = dict()
        gd = dict()
        gd['encoder'] = opt.get('encoder', False)
        # Forward-with-grad propagates downstream: encoder grad => decoder + renderer grad.
        gd['decoder'] = opt.get('encoder', False) or opt.get('decoder', False)
        gd['renderer'] = opt.get('encoder', False) or opt.get('decoder', False) or opt.get('renderer', False)
        return gd
