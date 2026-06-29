"""LH-NeF model wrapper: adds 'pred' and 'loss' modes on top of LHNeFBase."""

import torch

from models import register
from models.lhnef_base import LHNeFBase


@register('lhnef')
class LHNeF(LHNeFBase):

    def run_renderer(self, z_dec, coord, cell):
        return self.renderer(
            z_dec=z_dec, coord=coord, cell=cell,
            enc_blocks=getattr(self.encoder, 'last_enc_blocks', None),
            enc_regions=getattr(self.encoder, 'encoder_group_regions', None),
        )

    def forward(self, data, mode, has_opt=None, **kwargs):
        gd = self.get_gd_from_opt(has_opt)
        lcfg = self.loss_cfg

        if mode == 'encode':
            return super().forward(data, mode='encode', has_opt=has_opt, **kwargs)

        if mode == 'pred':
            z_dec, _ = super().forward(data, mode='z_dec', has_opt=has_opt)
            if gd['renderer']:
                return self.run_renderer(z_dec, data['gt_coord'], data['gt_cell'])
            with torch.no_grad():
                return self.run_renderer(z_dec, data['gt_coord'], data['gt_cell'])

        if mode == 'loss':
            z_dec, ret = super().forward(data, mode='z_dec', has_opt=has_opt)
            pred_patch = self.run_renderer(z_dec, data['gt_coord'], data['gt_cell'])
            target = data['gt']

            l1_loss = torch.abs(pred_patch - target).mean()
            ret['l1_loss'] = l1_loss.item()
            l1_loss_w = lcfg.get('l1_loss', 1)
            ret['loss'] = ret['loss'] + l1_loss * l1_loss_w

            mse_loss_w = float(lcfg.get('mse_loss', 0))
            if mse_loss_w > 0:
                mse_loss = ((pred_patch - target) ** 2).mean()
                ret['mse_loss'] = mse_loss.item()
                ret['loss'] = ret['loss'] + mse_loss * mse_loss_w

            # Metrics: value_dim=1 logs MSE/MAE (+IoU if value_kind starts with "occ");
            # value_dim=3 logs PSNR by reshaping queries to a grid when gt_is_grid=1.
            eps = 1e-10
            if pred_patch.ndim == 3 and target.ndim == 3:
                mse = (target - pred_patch).pow(2).mean()
                mae = (target - pred_patch).abs().mean()
                ret['mse'] = float(mse.detach().item())
                ret['mae'] = float(mae.detach().item())

                v_dim = int(pred_patch.shape[-1])
                if v_dim == 1:
                    vk = data.get('value_kind', None)
                    if isinstance(vk, (list, tuple)):
                        vk = vk[0] if len(vk) > 0 else None
                    vk_str = str(vk).lower() if vk is not None else ""
                    if vk_str.startswith("occ"):
                        p_occ = pred_patch[..., 0] > 0.0
                        t_occ = target[..., 0] > 0.0
                        inter = (p_occ & t_occ).sum(dim=1).float()
                        union = (p_occ | t_occ).sum(dim=1).float().clamp_min(1e-8)
                        ret['iou'] = float((inter / union).mean().item())
                elif v_dim == 3:
                    # PSNR for [-1, 1] images: 10*log10(MAX^2 / MSE) with MAX=2
                    # → equivalent to -10*log10(((tgt-pred)/2)^2). Computed
                    # directly on the flat [B, Q, 3] tensors so it works for
                    # both full-grid and subsampled training (gt_is_grid=0).
                    try:
                        mse2 = ((target - pred_patch) / 2).pow(2).mean(dim=[1, 2])  # [B]
                        ret['psnr'] = float((-10 * torch.log10(mse2 + eps)).mean().item())
                    except Exception:
                        ret['psnr'] = float('nan')
            return ret

        raise ValueError(f"Unknown mode={mode!r}")
