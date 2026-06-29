import os
import random

import torch
import torch.distributed as dist
import torchvision

import utils
from utils.geometry import make_coord_cell_grid
from .trainers import register
from .base_trainer import BaseTrainer


@register('lhnef_trainer')
class LHNeFTrainer(BaseTrainer):

    def prepare_visualize(self):
        self.vis_spec = dict()

        def get_samples(dataset, s):
            n = len(dataset)
            lst = [dataset[i] for i in list(range(0, n, max(1, n // s)))[:s]]
            data = dict()
            example = lst[0]
            for k, v in example.items():
                if torch.is_tensor(v):
                    data[k] = torch.stack([item[k] for item in lst]).cuda()
                elif isinstance(v, (list, tuple)) and k == 'views' and len(v) > 0 and torch.is_tensor(v[0]):
                    num = min(2, len(v))
                    for i in range(num):
                        data[f'view_{i}'] = torch.stack([item[k][i] for item in lst]).cuda()
            return data

        self.vis_spec['ds_samples'] = self.cfg.visualize.get('ds_samples', 0)
        if self.vis_spec['ds_samples'] > 0:
            # eval_ckpt.py may drop datasets.train; visualize only what exists.
            self.vis_ds_samples = {}
            if self.datasets.get('train') is not None:
                self.vis_ds_samples['train'] = get_samples(self.datasets['train'], self.vis_spec['ds_samples'])
            if self.datasets.get('val') is not None:
                self.vis_ds_samples['val'] = get_samples(self.datasets['val'], self.vis_spec['ds_samples'])
        self.vis_ae_center_zoom_res = self.cfg.visualize.get('ae_center_zoom_res')

    def make_datasets(self):
        super().make_datasets()

        self.vis_resolution = self.cfg.visualize.resolution
        if isinstance(self.vis_resolution, int):
            self.vis_resolution = (self.vis_resolution, self.vis_resolution)
        if self.is_master:
            random.seed(0) # to get a fixed vis set from wrapper_cae
            self.prepare_visualize()
            if self.cfg.random_seed is not None:
                random.seed(self.cfg.random_seed + self.rank)
            else:
                random.seed()

    def make_model(self, model_spec=None):
        super().make_model(model_spec)
        param_summary = {}
        for name, m in self.model.named_children():
            n_params = sum(p.numel() for p in m.parameters())
            self.log(f'  .{name} {utils.compute_num_params(m)}')
            param_summary[f'model_params/{name}'] = n_params
        param_summary['model_params/total'] = sum(
            p.numel() for p in self.model.parameters()
        )
        try:
            import wandb
            if self.is_master and wandb.run is not None:
                wandb.run.summary.update(param_summary)
        except Exception:
            pass

        # Populated in make_optimizers(); avoids enabling grad paths for modules with no optimizer.
        self.has_opt = dict()

    def make_optimizers(self):
        self.optimizers = dict()
        for name, spec in self.cfg.optimizers.items():
            if spec is None:
                continue
            params = list(self.model.get_params(name))
            if len(params) == 0:
                continue
            self.optimizers[name] = utils.make_optimizer(params, spec)
            self.has_opt[name] = True

    def train_step(self, data, bp=True):
        # Apply the cosine LR schedule (no-op if lr_scheduler.name != 'cosine')
        cur_lr = self._apply_lr() if bp else None
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            ret = self.model_ddp(data, mode='loss', has_opt=self.has_opt)
        loss = ret.pop('loss')
        ret['loss'] = loss.item()
        if bp:
            ret['lr'] = float(cur_lr) if cur_lr is not None else 0.0
            self.model_ddp.zero_grad()

            if self.amp_dtype == torch.float16:
                self.grad_scaler.scale(loss).backward()
                # Unscale gradients so the per-module grad-norm/NaN checks below are meaningful under AMP.
                try:
                    for o in self.optimizers.values():
                        self.grad_scaler.unscale_(o)
                except Exception:
                    pass
            else:
                loss.backward()

            def _grad_stats(params):
                n = 0
                s = None
                has_nan = False
                has_inf = False
                for p in params:
                    if p.grad is None:
                        continue
                    g = p.grad
                    n += 1
                    if torch.isnan(g).any():
                        has_nan = True
                    if torch.isinf(g).any():
                        has_inf = True
                    gn = g.detach().float().norm()
                    s = gn if s is None else (s + gn)
                mean_norm = float((s / max(1, n)).item()) if s is not None else 0.0
                return mean_norm, float(has_nan), float(has_inf), int(n)

            _GRAD_STATS_EVERY = 50
            if int(getattr(self, 'iter', 0)) % _GRAD_STATS_EVERY == 0:
                try:
                    enc = self.model.encoder if hasattr(self.model, 'encoder') else None
                    dec = self.model.decoder if hasattr(self.model, 'decoder') else None
                    rnd = self.model.renderer if hasattr(self.model, 'renderer') else None
                    if enc is not None:
                        m, nanf, inff, cnt = _grad_stats(enc.parameters())
                        ret['grad/encoder_norm_mean'] = m
                        ret['grad/encoder_has_nan'] = nanf
                        ret['grad/encoder_has_inf'] = inff
                        ret['grad/encoder_nparams_w_grad'] = cnt
                    if dec is not None:
                        m, nanf, inff, cnt = _grad_stats(dec.parameters())
                        ret['grad/decoder_norm_mean'] = m
                        ret['grad/decoder_has_nan'] = nanf
                        ret['grad/decoder_has_inf'] = inff
                        ret['grad/decoder_nparams_w_grad'] = cnt
                    if rnd is not None:
                        m, nanf, inff, cnt = _grad_stats(rnd.parameters())
                        ret['grad/renderer_norm_mean'] = m
                        ret['grad/renderer_has_nan'] = nanf
                        ret['grad/renderer_has_inf'] = inff
                        ret['grad/renderer_nparams_w_grad'] = cnt
                except Exception:
                    pass

            if self.amp_dtype == torch.float16:
                for o in self.optimizers.values():
                    self.grad_scaler.step(o)
                self.grad_scaler.update()
            else:
                for o in self.optimizers.values():
                    o.step()

        # Surface pyramid renderer per-level stats / grad norms into `ret` for wandb/tb logging.
        try:
            rnd = getattr(self.model, 'renderer', None)
            if rnd is not None:
                lvl_stats = getattr(rnd, 'last_level_stats', None)
                if isinstance(lvl_stats, dict) and len(lvl_stats) > 0:
                    for lvl, st in lvl_stats.items():
                        if not isinstance(st, dict):
                            continue
                        for k, v in st.items():
                            key = f'pyr/level_stats/{lvl}/{k}'
                            ret[key] = float(v)
                            if getattr(self, 'is_master', False):
                                try:
                                    self.log_scalar('train/' + key, float(v))
                                except Exception:
                                    pass

                lvl_gn = getattr(rnd, 'last_level_grad_norms', None)
                if isinstance(lvl_gn, dict) and len(lvl_gn) > 0:
                    for lvl, v in lvl_gn.items():
                        key = f'pyr/level_grad_rms/{lvl}'
                        ret[key] = float(v)
                        if getattr(self, 'is_master', False):
                            try:
                                self.log_scalar('train/' + key, float(v))
                            except Exception:
                                pass
        except Exception:
            pass

        return ret

    def train_iter_start(self):
        hrft_iter = self.cfg.get('hrft_start_after_iters')
        if hrft_iter is not None and self.iter == hrft_iter + 1:
            self.train_loader = self.loaders['train_hrft']
            self.train_loader_sampler = self.loader_samplers['train_hrft']
            self.train_loader_epoch = 0
            self.train_batch_id = len(self.train_loader) - 1

        prog_iter_rng = self.cfg.get('prog_res_training')
        if prog_iter_rng is not None:
            l, r = prog_iter_rng
            ds = self.loaders['train'].dataset
            if self.iter < l:
                ds.resize_gt_ub = ds.resize_gt_lb
            elif self.iter <= r:
                ds.resize_gt_ub = round(ds.resize_gt_lb + (self.iter - l) / (r - l) * (self.prog_res_ub - ds.resize_gt_lb))
            else:
                ds.resize_gt_ub = self.prog_res_ub


    def run_training(self):
        if self.cfg.get('prog_res_training') is not None:
            ds = self.loaders['train'].dataset
            self.prog_res_ub = ds.resize_gt_ub

        super().run_training()

    def visualize(self):
        # AE-style viz assumes `inp` is an image tensor; coord/value `inp` is a dict
        # and would crash here, so fall back to BaseTrainer.visualize() in that case.
        try:
            ds_val = self.datasets.get('val', None)
            sample0 = ds_val[0] if (ds_val is not None and len(ds_val) > 0) else None
            inp0 = sample0.get('inp', None) if isinstance(sample0, dict) else None
            is_image_inp = torch.is_tensor(inp0) and inp0.ndim == 4
        except Exception:
            is_image_inp = False
        if not is_image_inp:
            return super().visualize()

        self.model_ddp.eval()
        if self.is_master:
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                if self.vis_spec['ds_samples'] > 0:
                    self.visualize_ae()
        if self.distributed:
            dist.barrier()

    def visualize_ae_(self, name, data, bs=1):
        gt = data['gt']
        inp = data.get('inp', None)
        if not (torch.is_tensor(inp) and inp.ndim == 4):
            # AE viz expects image-shaped inputs; skip for coord/value.
            return
        n = int(inp.shape[0])
        pred = []
        center_zoom = []

        for i in range(0, n, bs):
            d = {k: v[i: min(i + bs, n)] for k, v in data.items()}
            pred.append(self.model(d, mode='pred'))

            if (self.vis_ae_center_zoom_res is not None) and (not name.endswith('_whole')):
                r0 = self.vis_resolution[0] / self.vis_ae_center_zoom_res
                r1 = self.vis_resolution[1] / self.vis_ae_center_zoom_res
                d['gt_coord'], d['gt_cell'] = make_coord_cell_grid(
                    self.vis_resolution, [[-r0, r0], [-r1, r1]], device=d['gt_coord'].device, bs=d['gt_coord'].shape[0])
                center_zoom.append(self.model(d, mode='pred'))

        pred = torch.cat(pred, dim=0)
        if self.is_master:
            vimg = []
            for i in range(len(gt)):
                vimg.extend([pred[i], gt[i]])
            vimg = torch.stack(vimg)
            vimg = torchvision.utils.make_grid(vimg, nrow=4, normalize=True, value_range=(-1, 1))
            self.log_image(name, vimg)

        if (self.vis_ae_center_zoom_res is not None) and (not name.endswith('_whole')):
            center_zoom = torch.cat(center_zoom, dim=0)
            if self.is_master:
                vimg = []
                for i in range(len(gt)):
                    vimg.extend([center_zoom[i], center_zoom[i]])
                vimg = torch.stack(vimg)
                vimg = torchvision.utils.make_grid(vimg, nrow=4, normalize=True, value_range=(-1, 1))
                self.log_image(name + '_center_zoom', vimg)

    def visualize_ae(self):
        for split in ['train', 'val']:
            if self.vis_ds_samples.get(split) is None:
                continue
            data = self.vis_ds_samples[split]
            self.visualize_ae_(split, data)

            if self.cfg.visualize.get('vis_ae_whole', False):
                x = data['inp']
                coord, cell = make_coord_cell_grid(x.shape[-2:], device=x.device, bs=x.shape[0])
                data_whole = {'inp': x, 'gt': x, 'gt_coord': coord, 'gt_cell': cell}
                self.visualize_ae_(split + '_whole', data_whole)
