import os
import time
import uuid
import copy
import random
import math
from functools import partial
from datetime import timedelta

import yaml
import wandb
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import datasets
import models
import utils
from .trainers import register
from vis import visualize_reconstructions, visualize_reconstructions_3d_occ, visualize_reconstructions_temperature
import pathlib

_math = math


def _fmt_scalar(val: float) -> str:
    """Format a scalar for logging; scientific notation for very small values."""
    if val == 0.0:
        return '0.0000'
    if abs(val) < 1e-3:
        return f'{val:.4e}'
    return f'{val:.4f}'


def worker_init_fn_(worker_id, num_workers, rank, world_size, seed):
    glo_worker_id = num_workers * rank + worker_id
    worker_seed = (num_workers * world_size * seed + glo_worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


@register('base_trainer')
class BaseTrainer():

    def __init__(self, cfg):
        self.rank = int(os.environ.get('RANK', '0'))
        self.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        self.is_master = (self.rank == 0)
        # Initialize DDP early so we can coordinate the run-dir suffix across ranks below.
        self.world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.distributed = (self.world_size > 1)
        self.cfg = cfg
        env = cfg._env

        if self.distributed:
            timeout_s = int(getattr(cfg, 'ddp_timeout', 7200) or 7200)
            dist.init_process_group(backend='nccl', timeout=timedelta(seconds=timeout_s))

        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', torch.cuda.current_device())

        # Auto-unique run directory.
        if getattr(env, 'auto_unique', True) and getattr(env, 'resume_mode', 'replace') != 'resume':
            # IMPORTANT (DDP): all ranks must agree on the SAME new folder name.
            # Rank0 chooses a suffix and broadcasts it.
            if self.distributed:
                payload = [None]
                if self.is_master:
                    payload[0] = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:4]
                dist.broadcast_object_list(payload, src=0)
                suffix = payload[0]
            else:
                suffix = time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:4]
            env.exp_name = f"{env.exp_name}-{suffix}"
            # Ensure exp_group and save_root exist
            if getattr(env, 'save_root', None) is None:
                env.save_root = os.path.dirname(env.save_dir)
            if getattr(env, 'exp_group', None) is None:
                env.exp_group = 'default'
            env.save_dir = os.path.join(env.save_root, env.exp_group, env.exp_name)

        # Barrier so all ranks see the same directory choice before any IO.
        if self.distributed:
            dist.barrier()

        # Refresh cfg_dict after possible env mutation and inject absolute ckpt_path.
        self.cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        try:
            self.cfg_dict['ckpt_path'] = cfg._env.save_dir
        except Exception:
            pass

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.use_amp = bool(cfg.get('use_amp', True))
        amp_dtype_str = str(cfg.get('amp_dtype', 'fp16')).lower()
        if amp_dtype_str in ('bf16', 'bfloat16'):
            self.amp_dtype = torch.bfloat16
            self.grad_scaler = torch.amp.GradScaler('cuda', enabled=False)
        elif amp_dtype_str in ('fp16', 'float16', 'half'):
            self.amp_dtype = torch.float16
            self.grad_scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        else:
            raise ValueError(f"amp_dtype must be 'fp16' or 'bf16', got {amp_dtype_str!r}")
        self.amp_dtype_str = amp_dtype_str

        force_replace = False
        if cfg._env.resume_mode == 'resume':
            replace = False
        else:
            replace = True
            if cfg._env.resume_mode == 'force_replace':
                force_replace = True
            elif cfg._env.resume_mode != 'replace':
                raise NotImplementedError
        if self.is_master:
            utils.ensure_path(cfg._env.save_dir, replace=replace, force_replace=force_replace)

        # Setup log, tb, wandb
        if self.is_master:
            logger, writer = utils.set_save_dir(env.save_dir, replace=False)
            with open(os.path.join(env.save_dir, 'cfg.yaml'), 'w') as f:
                yaml.dump(self.cfg_dict, f, sort_keys=False)
            self.log = logger.info

            self.enable_tb = True
            self.writer = writer

            if env.wandb:
                self.enable_wandb = True
                os.environ['WANDB_NAME'] = env.exp_name
                os.environ['WANDB_DIR'] = env.save_dir
                wandb_cfg = None
                try:
                    with open('wandb.yaml', 'r') as f:
                        wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
                except Exception:
                    # Fallback to src/wandb.yaml if not in CWD
                    try:
                        fallback = os.path.join(os.path.dirname(__file__), '..', 'wandb.yaml')
                        fallback = os.path.abspath(fallback)
                        with open(fallback, 'r') as f:
                            wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
                    except Exception:
                        wandb_cfg = None
                if wandb_cfg is not None:
                    wandb.init(
                        project=wandb_cfg.get('project', None),
                        config=self.cfg_dict,
                        dir=env.save_dir,
                        name=env.exp_name,
                        resume="allow",
                    )
                else:
                    wandb.init(
                        config=self.cfg_dict,
                        dir=env.save_dir,
                        name=env.exp_name,
                        resume="allow",
                    )
                try:
                    wandb.run.summary['ckpt_path'] = cfg._env.save_dir
                except Exception as e:
                    self.log(f"Could not set wandb summary ckpt_path: {e}")
            else:
                self.enable_wandb = False
        else:
            self.log = lambda *args, **kwargs: None
            self.enable_tb = False
            self.enable_wandb = False

        if self.distributed:
            self.log(f'Distributed training enabled. World size: {self.world_size}.')
            dist.barrier()
        
        self.log(f'Environment setup done.')
        # eval-only runs call evaluate() without run_training(), so initialize here.
        self.log_buffer = []

    def seed_everything(self, seed, rank_shift=True):
        if rank_shift:
            seed += self.rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _init_lr_schedulers(self):
        """Build an iter-stepped cosine LR schedule with optional linear warmup.

        Cfg (under ``lr_scheduler``):
          name: ``cosine`` | ``none``        (default ``none`` — schedule disabled)
          schedule_iters: int                 horizon for the cosine decay; AFTER this
                                              iter LR HOLDS at base_lr * min_lr_ratio
                                              (decoupled from max_iter — pick a target
                                              like "30 pseudoepochs"; training can
                                              continue past it without breaking)
          warmup_iters: int (default 0)       linear warmup from 0 -> base_lr
          min_lr_ratio: float (default 0.0)   cosine floor / hold value

        Resume-safe: base LRs are read from cfg.optimizers.{name}.args.lr (NOT
        from the loaded optimizer state, which may be a partly-decayed value
        if resuming mid-schedule).
        """
        sched_cfg = self.cfg.get('lr_scheduler', None) or {}
        try:
            name = str(sched_cfg.get('name', 'none') or 'none').lower().strip()
        except Exception:
            name = 'none'
        self._lr_sched_active = (name == 'cosine')
        if not self._lr_sched_active:
            return

        sch_iters = int(sched_cfg.get('schedule_iters', 0) or 0)
        if sch_iters <= 0:
            raise RuntimeError(
                "lr_scheduler=cosine requires schedule_iters > 0 (target iter horizon "
                "for the cosine decay; training may continue past this point at "
                "min_lr_ratio*base_lr)."
            )
        warmup = int(sched_cfg.get('warmup_iters', 0) or 0)
        min_ratio = float(sched_cfg.get('min_lr_ratio', 0.0) or 0.0)
        if warmup < 0 or warmup >= sch_iters:
            raise ValueError(f"lr_scheduler.warmup_iters must be in [0, schedule_iters); "
                             f"got {warmup} (schedule_iters={sch_iters}).")
        if not (0.0 <= min_ratio < 1.0):
            raise ValueError(f"lr_scheduler.min_lr_ratio must be in [0, 1); got {min_ratio}.")

        self._lr_sched_schedule_iters = sch_iters
        self._lr_sched_warmup = warmup
        self._lr_sched_min_ratio = min_ratio

        # Snapshot base LRs from cfg (resume-safe). Fallback to current opt
        # state only if cfg lookup fails.
        self._lr_sched_base_lrs = {}
        for opt_name, opt in self.optimizers.items():
            cfg_lr = None
            try:
                cfg_lr = float(self.cfg.optimizers[opt_name].args.lr)
            except Exception:
                pass
            if cfg_lr is None:
                self._lr_sched_base_lrs[opt_name] = [float(pg['lr']) for pg in opt.param_groups]
            else:
                self._lr_sched_base_lrs[opt_name] = [cfg_lr for _ in opt.param_groups]

        if self.is_master:
            self.log(f"[lr_scheduler] cosine: schedule_iters={sch_iters} "
                     f"warmup_iters={warmup} min_lr_ratio={min_ratio} "
                     f"base_lrs={self._lr_sched_base_lrs}")

    def _lr_factor(self, it: int) -> float:
        """Cosine factor in [min_lr_ratio, 1]. Holds at min_lr_ratio past schedule_iters."""
        w = int(self._lr_sched_warmup)
        s = int(self._lr_sched_schedule_iters)
        if w > 0 and it < w:
            return float(it + 1) / float(w)
        if it >= s:
            return float(self._lr_sched_min_ratio)
        denom = max(1, s - w)
        progress = (int(it) - w) / float(denom)
        progress = max(0.0, min(1.0, progress))
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self._lr_sched_min_ratio + (1.0 - self._lr_sched_min_ratio) * cos_factor

    def _apply_lr(self) -> float:
        """Set per-step LR on all optimizers / param-groups using cfg-snapshotted bases.

        Returns the current first-group LR (for logging). When the scheduler is
        disabled, returns the current first-group LR unchanged. Cheap no-op
        when disabled.
        """
        if not getattr(self, '_lr_sched_active', False):
            for opt in self.optimizers.values():
                return float(opt.param_groups[0]['lr'])
            return 0.0
        factor = self._lr_factor(int(self.iter))
        cur_lr_log = 0.0
        for opt_name, opt in self.optimizers.items():
            base = self._lr_sched_base_lrs[opt_name]
            for pg, lr0 in zip(opt.param_groups, base):
                pg['lr'] = float(lr0) * float(factor)
            cur_lr_log = float(opt.param_groups[0]['lr'])
        return cur_lr_log

    def run(self):
        if self.cfg.random_seed is not None:
            self.seed_everything(self.cfg.random_seed, rank_shift=True)

        self.make_datasets()

        self._maybe_auto_schedule()

        if self.cfg.get('eval_only', False):
            model_spec = self.cfg.get('eval_model')
            if model_spec is not None:
                model_spec = torch.load(model_spec, map_location="cpu", weights_only=False)['model']
            self.make_model(model_spec); model_spec = None
            self.iter = 0
            self.evaluate()
            self.visualize()
        else:
            resume_file = os.path.join(self.cfg._env.save_dir, 'last-model.pth')
            if os.path.isfile(resume_file):
                ckpt = torch.load(resume_file, map_location="cpu", weights_only=False)

            if os.path.isfile(resume_file):
                model_spec = copy.deepcopy(OmegaConf.to_container(self.cfg.model, resolve=True))
                model_spec['sd'] = ckpt['model']['sd']
                self.make_model(model_spec)
                model_spec = None
                self.log(f'Resumed model from checkpoint {resume_file}.')
            else:
                self.make_model()

            self.make_optimizers()
            if os.path.isfile(resume_file):
                opt_dict = ckpt['optimizers']
                for k, v in opt_dict.items():
                    # Robust to optimizer groups that no longer exist in the current cfg.
                    if k in self.optimizers:
                        self.optimizers[k].load_state_dict(v['sd'])
                opt_dict = None
                self.log(f'Resumed optimizers from checkpoint {resume_file}.')
                # Only restore GradScaler when both the current run and the
                # saved checkpoint are fp16 AMP
                ckpt_amp_dtype = str(ckpt.get('amp_dtype', 'fp16')).lower()
                if (self.use_amp
                    and self.amp_dtype == torch.float16
                    and ckpt_amp_dtype in ('fp16', 'float16', 'half')
                    and ckpt.get('grad_scaler') is not None):
                    self.grad_scaler.load_state_dict(ckpt['grad_scaler'])
                    self.log('Resumed GradScaler state.')
                elif ckpt_amp_dtype != self.amp_dtype_str:
                    self.log(f"AMP dtype changed: {ckpt_amp_dtype} -> {self.amp_dtype_str}; "
                             f"skipping GradScaler state.")

            # Build LR scheduler AFTER both make_optimizers and resume. We
            # snapshot base LRs from cfg (not from loaded opt state) so the
            # cosine anchors on the cfg-declared LR even when resuming mid-decay.
            self._init_lr_schedulers()

            ckpt = None
            self.run_training()

            # After training completes, evaluate on test set if available.
            self._run_test_eval()

        if self.enable_tb:
            self.writer.close()
        if self.enable_wandb:
            wandb.finish()

    def make_distributed_loader(self, dataset, batch_size, drop_last, shuffle, num_workers):
        # num_workers is per-GPU (not divided by world_size)
        if self.cfg.random_seed is not None and num_workers > 0:
            worker_init_fn = partial(worker_init_fn_,
                num_workers=num_workers, rank=self.rank, world_size=self.world_size, seed=self.cfg.random_seed)
            persistent_workers = True
        else:
            worker_init_fn = None
            persistent_workers = False
        sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
        assert batch_size % self.world_size == 0
        loader = DataLoader(dataset, batch_size // self.world_size, drop_last=drop_last,
                            sampler=sampler, shuffle=((sampler is None) and shuffle),
                            num_workers=num_workers, pin_memory=True,
                            worker_init_fn=worker_init_fn, persistent_workers=persistent_workers)
        return loader, sampler

    def _sync_n_inp_across_splits(self, cfg):
        """Propagate train n_inp to val/test so the encoder sees the same input distribution.

        Handles both wrapper layouts: nested (wrapper_coord_value:
        datasets.{split}.args.dataset.args.n_inp) and direct
        (wrapper_cae_coord_value: datasets.{split}.args.n_inp).
        """
        try:
            train_ds = cfg.get("datasets", {}).get("train", {})
            train_wrapper_args = train_ds.get("args", {}) or {}

            train_inner = (train_wrapper_args.get("dataset", {}) or {}).get("args", {}) or {}
            train_n_inp = train_inner.get("n_inp", None)
            nested = True
            if train_n_inp is None:
                train_n_inp = train_wrapper_args.get("n_inp", None)
                nested = False
            if train_n_inp is None:
                return
            train_n_inp = int(train_n_inp)

            for split in ("val", "test"):
                split_ds = cfg.get("datasets", {}).get(split, None)
                if split_ds is None:
                    continue
                split_wrapper_args = split_ds.get("args", {}) or {}
                if nested:
                    split_inner = (split_wrapper_args.get("dataset", {}) or {}).get("args", {}) or {}
                    old = split_inner.get("n_inp", None)
                    if old is None or int(old) != train_n_inp:
                        self.log(f"[auto-sync] {split} n_inp={old} -> {train_n_inp} (matching train)")
                        split_inner["n_inp"] = train_n_inp
                else:
                    old = split_wrapper_args.get("n_inp", None)
                    if old is None or int(old) != train_n_inp:
                        self.log(f"[auto-sync] {split} n_inp={old} -> {train_n_inp} (matching train)")
                        split_wrapper_args["n_inp"] = train_n_inp
        except Exception:
            pass

    def make_datasets(self):
        cfg = self.cfg
        self.datasets = dict()
        self.loaders = dict()
        self.loader_samplers = dict()

        # Sweeps that override only train n_inp would otherwise silently create a train/val mismatch.
        self._sync_n_inp_across_splits(cfg)

        for split, spec in cfg.datasets.items():
            loader_spec = spec.pop('loader')
            dataset = datasets.make(spec)
            self.datasets[split] = dataset
            self.log(f'Datasets - {split}: len={len(dataset)}')

            if self.is_master:
                try:
                    inner = getattr(dataset, 'dataset', dataset)
                    num_classes = None
                    class_to_idx = getattr(inner, 'class_to_idx', None)
                    if isinstance(class_to_idx, dict) and len(class_to_idx) > 0:
                        num_classes = len(class_to_idx)
                    elif hasattr(inner, 'classes'):
                        try:
                            num_classes = len(getattr(inner, 'classes'))
                        except Exception:
                            num_classes = None
                    elif hasattr(inner, 'files'):
                        try:
                            files_attr = getattr(inner, 'files')
                            if isinstance(files_attr, (list, tuple)) and len(files_attr) > 0:
                                first = files_attr[0]
                                if isinstance(first, (list, tuple)) and len(first) >= 2:
                                    labels = [int(f[1]) for f in files_attr]
                                    num_classes = len(set(labels))
                        except Exception:
                            num_classes = None

                    sample = dataset[0]
                    msg = f"Dataset[{split}] OK"
                    if num_classes is not None:
                        msg += f", classes={num_classes}"
                    if isinstance(sample, dict):
                        try:
                            import torch
                            if 'inp' in sample and torch.is_tensor(sample['inp']):
                                msg += f", inp={tuple(sample['inp'].shape)}"
                            if 'gt' in sample and torch.is_tensor(sample['gt']):
                                msg += f", gt={tuple(sample['gt'].shape)}"
                            if 'gt_coord' in sample and torch.is_tensor(sample['gt_coord']):
                                msg += f", coord={tuple(sample['gt_coord'].shape)}"
                            if 'gt_cell' in sample and torch.is_tensor(sample['gt_cell']):
                                msg += f", cell={tuple(sample['gt_cell'].shape)}"
                            if 'views' in sample and isinstance(sample['views'], list):
                                msg += f", views={len(sample['views'])}"
                            if 'label' in sample:
                                try:
                                    lbl = sample['label']
                                    lbl_int = int(lbl if not hasattr(lbl, 'item') else lbl.item())
                                    msg += f", label_example={lbl_int}"
                                except Exception:
                                    msg += ", label=available"
                        except Exception:
                            pass
                    else:
                        msg += f", sample_type={type(sample).__name__}"
                    self.log(msg)
                except Exception as e:
                    self.log(f"Dataset[{split}] sanity check failed: {e}")

            drop_last = loader_spec.get('drop_last', (split == 'train'))
            shuffle = loader_spec.get('shuffle', (split == 'train'))

            # FFCV-backed datasets build their own loader (FFCV's Loader is not a
            # torch.utils.data.Dataset and must NOT be wrapped in a PyTorch DataLoader).
            if getattr(dataset, 'is_ffcv', False):
                self.loaders[split], self.loader_samplers[split] = dataset.make_loader(
                    batch_size=loader_spec.batch_size,
                    num_workers=loader_spec.num_workers,
                    distributed=self.distributed,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    device=self.device,
                    world_size=self.world_size,
                )
            else:
                self.loaders[split], self.loader_samplers[split] = self.make_distributed_loader(
                    dataset, loader_spec.batch_size, drop_last, shuffle, loader_spec.num_workers)

        self.full_loaders = {}

    def make_model(self, model_spec=None):
        if model_spec is None:
            model = models.make(self.cfg.model)
        else:
            model = models.make(model_spec, load_sd=True)
        self.log(f'Model: #params={utils.compute_num_params(model)}')

        use_compile = bool(self.cfg.get('use_compile', False))
        if use_compile:
            try:
                model = torch.compile(model, mode='reduce-overhead')
                self.log('Model compiled with torch.compile (reduce-overhead mode)')
            except Exception as e:
                self.log(f'torch.compile failed, continuing without: {e}')

        if self.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda()
            model_ddp = DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.cfg.get('find_unused_parameters', False),
                broadcast_buffers=False,
            )
        else:
            model.cuda()
            model_ddp = model

        self.model = model
        self.model_ddp = model_ddp

        try:
            if self.is_master and self.enable_wandb and (wandb.run is not None):
                latC = int(self.model.z_shape[0])
                zH = int(self.model.z_shape[1])
                zW = int(self.model.z_shape[2])
                wandb.run.config.update({
                    "model.args.z_shape": [latC, zH, zW],
                    "model.args.encoder.args.z_channels": latC,
                    "model.args.decoder.args.z_channels": latC,
                }, allow_val_change=True)
        except Exception:
            pass

    def make_optimizers(self):
        self.optimizers = {'all': utils.make_optimizer(self.model.parameters(), self.cfg.optimizers)}

    def _maybe_auto_schedule(self):
        """Derive (epoch_iter, max_iter, eval_iter, vis_iter) from train_len, global batch, and
        cfg.train_epochs / cfg.eval_every_epochs when cfg.auto_schedule is set."""
        cfg = self.cfg
        try:
            auto_flag = bool(getattr(cfg, 'auto_schedule', False))
        except Exception:
            auto_flag = False
        if not auto_flag:
            return

        if 'train' not in getattr(self, 'datasets', {}) or 'train' not in getattr(self, 'loaders', {}):
            return

        try:
            train_len = len(self.datasets['train'])
        except Exception as e:
            if self.is_master:
                self.log(f"[auto_schedule] Could not get len(train): {e}")
            return

        # Prefer config global batch; fall back to DataLoader.batch_size * world_size.
        global_batch = None
        try:
            global_batch = int(cfg.datasets.train.loader.batch_size)
        except Exception:
            global_batch = None

        if (global_batch is None) or (global_batch <= 0):
            try:
                train_loader = self.loaders.get('train', None)
                if train_loader is not None and getattr(train_loader, 'batch_size', None) is not None:
                    per_device_bs = int(train_loader.batch_size)
                    if per_device_bs > 0:
                        global_batch = per_device_bs * max(1, int(self.world_size))
            except Exception:
                global_batch = None

        if global_batch is None or global_batch <= 0:
            if self.is_master:
                self.log(f"[auto_schedule] Could not determine global batch size (got {global_batch}).")
            return

        iters_per_epoch = max(1, math.ceil(float(train_len) / float(global_batch)))

        try:
            epochs = int(getattr(cfg, 'train_epochs', 200) or 200)
        except Exception:
            epochs = 200
        try:
            eval_every = int(getattr(cfg, 'eval_every_epochs', 5) or 5)
        except Exception:
            eval_every = 5

        cfg.epoch_iter = int(iters_per_epoch)
        cfg.max_iter = int(iters_per_epoch * epochs)
        cfg.eval_iter = int(iters_per_epoch * eval_every)
        cfg.vis_iter = int(cfg.eval_iter)

        # save_iter must be a multiple of epoch_iter (asserted in run_training).
        # 'last-model.pth' is written every epoch regardless.
        save_every_ep = 0
        try:
            save_every_ep = int(getattr(cfg, 'save_every_epochs', 0) or 0)
        except Exception:
            save_every_ep = 0
        if save_every_ep > 0:
            cfg.save_iter = int(iters_per_epoch * save_every_ep)
        else:
            cfg.save_iter = None

        if self.is_master:
            try:
                self.log(
                    f"[auto_schedule] len(train)={train_len}, global_batch={global_batch}, "
                    f"iters/epoch={iters_per_epoch}, max_iter={cfg.max_iter}, "
                    f"eval_iter={cfg.eval_iter}, vis_iter={cfg.vis_iter}"
                )
            except Exception:
                pass

    def run_training(self):
        cfg = self.cfg
        max_iter = cfg['max_iter']
        epoch_iter = cfg['epoch_iter']
        assert max_iter % epoch_iter == 0
        max_epoch = max_iter // epoch_iter

        save_iter = cfg.get('save_iter')
        assert save_iter is None or save_iter % epoch_iter == 0
        save_epoch = save_iter // epoch_iter if save_iter is not None else max_epoch + 1

        eval_iter = cfg.get('eval_iter')
        assert eval_iter is None or eval_iter % epoch_iter == 0
        eval_epoch = eval_iter // epoch_iter if eval_iter is not None else max_epoch + 1

        vis_iter = cfg.get('vis_iter')
        assert vis_iter is None or vis_iter % epoch_iter == 0
        vis_epoch = vis_iter // epoch_iter if vis_iter is not None else max_epoch + 1

        if cfg.get('ckpt_select_metric') is not None:
            m = cfg.ckpt_select_metric
            self.ckpt_select_metric = m.name
            self.ckpt_select_type = m.type
            if m.type == 'min':
                self.ckpt_select_v = 1e18
            elif m.type == 'max':
                self.ckpt_select_v = -1e18
        else:
            self.ckpt_select_metric = None
            self.ckpt_select_v = None

        self.train_loader = self.loaders['train']
        self.train_loader_sampler = self.loader_samplers['train']
        self.train_loader_epoch = 0
        self.train_batch_id = len(self.train_loader) - 1

        self.iter = 0

        resume_file = os.path.join(self.cfg._env.save_dir, 'last-model.pth')
        if os.path.isfile(resume_file):
            ckpt = torch.load(resume_file, map_location="cpu", weights_only=False)
            for _ in range(ckpt['iter']):
                self.iter += 1
                self.train_iter_start()
            if 'ckpt_select_v' in ckpt:
                self.ckpt_select_v = ckpt['ckpt_select_v']
            self.train_loader_epoch = ckpt['train_loader_epoch']
            self.train_batch_id = len(self.train_loader) - 1
            ckpt = None
            self.log(f'Resumed iter status from checkpoint {resume_file}.')

        start_epoch = self.iter // epoch_iter + 1
        epoch_timer = utils.EpochTimer(max_epoch - start_epoch + 1)
        for epoch in range(start_epoch, max_epoch + 1):
            self.log_buffer = [f'Epoch {epoch}']

            if self.distributed:
                for sampler in self.loader_samplers.values():
                    if sampler is not self.train_loader_sampler:
                        sampler.set_epoch(epoch)

            self.model_ddp.train()

            ave_scalars = dict()
            pbar = range(1, epoch_iter + 1)
            if self.is_master:
                pbar = tqdm(pbar, desc='train', leave=False)

            t_data = 0
            t_model = 0
            t1 = time.time()
            for _ in pbar:
                self.iter += 1
                self.train_iter_start()

                self.train_batch_id += 1
                if self.train_batch_id == len(self.train_loader):
                    self.train_loader_epoch += 1
                    if self.distributed:
                        self.train_loader_sampler.set_epoch(self.train_loader_epoch)
                    self.train_loader_iter = iter(self.train_loader)
                    self.train_batch_id = 0

                data = next(self.train_loader_iter)
                # Recursive to-device for nested dict/list batches.
                def _to_device(x):
                    if torch.is_tensor(x):
                        return x.cuda()
                    if isinstance(x, (list, tuple)):
                        return [_to_device(_v) for _v in x]
                    if isinstance(x, dict):
                        return {kk: _to_device(vv) for kk, vv in x.items()}
                    return x
                data = _to_device(data)
                t0 = time.time()
                t_data += t0 - t1

                ret = self.train_step(data)
                t1 = time.time()
                t_model += t1 - t0

                def _batch_size_from(sample):
                    if torch.is_tensor(sample):
                        return int(sample.shape[0])
                    if isinstance(sample, (list, tuple)) and sample and torch.is_tensor(sample[0]):
                        return int(sample[0].shape[0])
                    if isinstance(sample, dict):
                        for v in sample.values():
                            b = _batch_size_from(v)
                            if b is not None:
                                return int(b)
                    return None
                bs = _batch_size_from(data) or 1
                for k, v in ret.items():
                    if ave_scalars.get(k) is None:
                        ave_scalars[k] = utils.Averager()
                    ave_scalars[k].add(v, n=bs)

                if self.is_master:
                    pbar.set_description(desc=f"train: loss={ret['loss']:.4f}")

            self.sync_ave_scalars_(ave_scalars)

            logtext = 'train:'
            for k, v in ave_scalars.items():
                val = v.item()
                if _math.isnan(val) or _math.isinf(val):
                    continue
                logtext += f' {k}={_fmt_scalar(val)}'
                self.log_scalar('train/' + k, val)
            logtext += f' (d={t_data / (t_data + t_model):.2f})'
            self.log_buffer.append(logtext)

            try:
                if 'psnr' in ave_scalars:
                    psnr_val = ave_scalars['psnr'].item()
                    if not (math.isnan(psnr_val) or math.isinf(psnr_val)):
                        self.log_buffer.append(f">>> TRAIN PSNR (full epoch): {psnr_val:.4f} <<<")
            except Exception:
                pass

            if epoch % save_epoch == 0 and epoch != max_epoch:
                self.save_ckpt(f'epoch-{epoch}.pth')

            try:
                self._train_psnr_proxy(epoch=epoch)
            except Exception as e:
                if self.is_master:
                    self.log(f"[train_psnr_proxy] failed: {e}")

            if epoch % eval_epoch == 0:
                eval_ave_scalars = self.evaluate()

                if self.ckpt_select_metric is not None and self.ckpt_select_metric in eval_ave_scalars:
                    v = eval_ave_scalars[self.ckpt_select_metric].item()
                    if ((self.ckpt_select_type == 'min' and v < self.ckpt_select_v) or
                        (self.ckpt_select_type == 'max' and v > self.ckpt_select_v)):
                        self.ckpt_select_v = v
                        self.save_ckpt('best-model.pth')

            if epoch % vis_epoch == 0:
                self.visualize()

            # save_last_ckpt is configurable since per-epoch writes are heavy on network filesystems.
            if bool(self.cfg.get("save_last_ckpt", True)):
                self.save_ckpt('last-model.pth')

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            self.log_buffer.append(f'{epoch_time} {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))

    def train_iter_start(self):
        pass

    def train_step(self, data, bp=True):
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
            ret = self.model_ddp(data)
        loss = ret.pop('loss')
        ret['loss'] = loss.item()
        # Coerce scalar tensors to floats for Averager/logging.
        for k in list(ret.keys()):
            v = ret[k]
            if isinstance(v, torch.Tensor) and v.ndim == 0:
                ret[k] = float(v.detach().item())
        if bp:
            self.model_ddp.zero_grad()
            self.grad_scaler.scale(loss).backward()
            # GradScaler.step() handles unscale internally and skips on inf/nan.
            for o in self.optimizers.values():
                self.grad_scaler.step(o)
            self.grad_scaler.update()
        return ret

    @torch.no_grad()
    def _train_psnr_proxy(self, *, epoch: int):
        """Cheap train-set PSNR proxy via FULL-grid rendering on a small random subset.

        Config keys under cfg.train_psnr_proxy:
        enable, every_epochs, n_samples, batch_size, num_workers.
        """
        cfg = getattr(self.cfg, 'train_psnr_proxy', None)
        if cfg is None:
            return
        try:
            enable = bool(getattr(cfg, 'enable', False))
        except Exception:
            enable = False
        if not enable:
            return
        try:
            every = int(getattr(cfg, 'every_epochs', 1) or 1)
        except Exception:
            every = 1
        if every <= 0 or (epoch % every) != 0:
            return

        # Master only; barrier keeps ranks aligned.
        if not self.is_master:
            if self.distributed:
                dist.barrier()
            return

        try:
            n_samples = int(getattr(cfg, 'n_samples', 1000) or 1000)
        except Exception:
            n_samples = 1000
        try:
            bs = int(getattr(cfg, 'batch_size', 8) or 8)
        except Exception:
            bs = 8
        try:
            nw = int(getattr(cfg, 'num_workers', 0) or 0)
        except Exception:
            nw = 0

        if 'train' not in self.datasets or 'train' not in self.loaders:
            return

        ds = self.datasets['train']
        N = len(ds)
        if N <= 0:
            return
        n_samples = min(int(n_samples), int(N))
        if n_samples <= 0:
            return

        # Force full GT rendering for this proxy pass if the dataset supports it.
        setter = getattr(ds, 'set_force_full_gt', None)
        had_force = getattr(ds, 'force_full_gt', False)
        if callable(setter):
            setter(True)

        rng = np.random.RandomState(int(getattr(self.cfg, 'random_seed', 0) or 0) + int(epoch) * 9973)
        idx = rng.choice(np.arange(N), size=n_samples, replace=False).tolist()
        # FFCV datasets need a Loader with indices=, not a Subset+DataLoader (per-sample
        # __getitem__ rebuilds an FFCV Loader from scratch — way too slow on a 237 GB train.beton).
        if getattr(ds, 'is_ffcv', False):
            loader = ds.make_subset_loader(
                idx, batch_size=bs, num_workers=nw, device=self.device,
            )
        else:
            sub = Subset(ds, idx)
            loader = DataLoader(sub, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=True)

        was_training = self.model_ddp.training
        self.model_ddp.eval()

        ave = {}
        seen = 0
        for batch in loader:
            def _to_device(x):
                if torch.is_tensor(x):
                    return x.cuda()
                if isinstance(x, (list, tuple)):
                    return [_to_device(_v) for _v in x]
                if isinstance(x, dict):
                    return {kk: _to_device(vv) for kk, vv in x.items()}
                return x
            batch = _to_device(batch)
            ret = self.train_step(batch, bp=False)
            def _batch_size_from(sample):
                if torch.is_tensor(sample):
                    return int(sample.shape[0])
                if isinstance(sample, (list, tuple)) and sample:
                    return _batch_size_from(sample[0])
                if isinstance(sample, dict):
                    for v in sample.values():
                        b = _batch_size_from(v)
                        if b is not None:
                            return int(b)
                return None
            bs_eff = _batch_size_from(batch) or 1
            for k, v in ret.items():
                if ave.get(k) is None:
                    ave[k] = utils.Averager()
                ave[k].add(v, n=bs_eff)
            seen += bs_eff
            if seen >= n_samples:
                break

        if was_training:
            self.model_ddp.train()
        if callable(setter):
            setter(bool(had_force))

        if 'psnr' in ave:
            v = float(ave['psnr'].item())
            self.log_scalar('train_proxy/psnr', v)
            self.log_buffer.append(f"train_proxy/psnr={v:.4f}")

        if self.distributed:
            dist.barrier()

    def evaluate(self):
        # eval_only runs may call evaluate() before run_training() initializes log_buffer.
        if not hasattr(self, "log_buffer") or self.log_buffer is None:
            self.log_buffer = []
        self.model_ddp.eval()

        ave_scalars = dict()
        # Optional cheap val subset; DDP-safe: rank 0 picks indices and broadcasts.
        val_loader = self.loaders['val']
        try:
            subcfg = getattr(self.cfg, 'eval_subset', None)
        except Exception:
            subcfg = None
        if subcfg is not None and bool(getattr(subcfg, 'enable', False)):
            try:
                ds = self.datasets.get('val', None)
                if ds is not None:
                    N = int(len(ds))
                    n_samples = int(getattr(subcfg, 'n_samples', 0) or 0)
                    frac = float(getattr(subcfg, 'frac', 0.0) or 0.0)
                    if n_samples <= 0 and frac > 0:
                        n_samples = int(max(1, round(frac * N)))
                    n_samples = int(min(max(1, n_samples), N))

                    seed = int(getattr(subcfg, 'seed', 0) or 0)
                    if self.is_master:
                        rng = np.random.RandomState(seed)
                        idx = rng.choice(np.arange(N), size=n_samples, replace=False).astype(np.int64)
                        idx_t = torch.from_numpy(idx)
                    else:
                        idx_t = torch.empty((n_samples,), dtype=torch.int64)
                    if self.distributed:
                        idx_t = idx_t.to(self.device)
                        dist.broadcast(idx_t, src=0)
                        idx = idx_t.cpu().numpy()
                    else:
                        idx = idx_t.numpy()

                    try:
                        bs = int(getattr(self.cfg.datasets.val.loader, 'batch_size', 1) or 1)
                    except Exception:
                        bs = int(getattr(val_loader, 'batch_size', 1) or 1)
                    try:
                        nw = int(getattr(self.cfg.datasets.val.loader, 'num_workers', 0) or 0)
                    except Exception:
                        nw = 0
                    drop_last = False
                    shuffle = False
                    if getattr(ds, 'is_ffcv', False):
                        # FFCV: build a Loader over the explicit indices (fast path).
                        val_loader = ds.make_subset_loader(
                            idx.tolist(), batch_size=bs, num_workers=nw, device=self.device,
                        )
                    else:
                        sub = Subset(ds, idx.tolist())
                        if self.distributed:
                            val_loader, _sampler = self.make_distributed_loader(
                                sub, batch_size=bs, drop_last=drop_last, shuffle=shuffle, num_workers=nw
                            )
                        else:
                            val_loader = DataLoader(
                                sub, batch_size=bs, shuffle=shuffle, drop_last=drop_last, num_workers=nw, pin_memory=True
                            )
                    if self.is_master:
                        self.log(f"[eval_subset] val subset: n={n_samples}/{N} (seed={seed}), bs={bs}")
            except Exception as e:
                if self.is_master:
                    self.log(f"[eval_subset] disabled due to error: {e}")

        pbar = val_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='val', leave=False)

        for data in pbar:
            def _to_device(x):
                if torch.is_tensor(x):
                    return x.cuda()
                if isinstance(x, (list, tuple)):
                    return [_to_device(_v) for _v in x]
                if isinstance(x, dict):
                    return {kk: _to_device(vv) for kk, vv in x.items()}
                return x
            data = _to_device(data)
            with torch.no_grad():
                ret = self.train_step(data, bp=False)

            def _batch_size_from(sample):
                if torch.is_tensor(sample):
                    return int(sample.shape[0])
                if isinstance(sample, (list, tuple)) and sample and torch.is_tensor(sample[0]):
                    return int(sample[0].shape[0])
                if isinstance(sample, dict):
                    for v in sample.values():
                        b = _batch_size_from(v)
                        if b is not None:
                            return int(b)
                return None
            bs = _batch_size_from(data) or 1
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=bs)

            if self.is_master:
                pbar.set_description(desc=f'val: loss={ret["loss"]:.4f}')

        self.sync_ave_scalars_(ave_scalars)

        logtext = 'val:'
        for k, v in ave_scalars.items():
            val = v.item()
            if _math.isnan(val) or _math.isinf(val):
                continue
            logtext += f' {k}={_fmt_scalar(val)}'
            self.log_scalar('val/' + k, val)
        self.log_buffer.append(logtext)

        try:
            if 'psnr' in ave_scalars:
                psnr_val = ave_scalars['psnr'].item()
                if not (_math.isnan(psnr_val) or _math.isinf(psnr_val)):
                    self.log_buffer.append(f">>> VAL PSNR (full set): {psnr_val:.4f} <<<")
        except Exception:
            pass

        return ave_scalars

    def _run_test_eval(self):
        """Evaluate on test set after training, loading best-model.pth. Falls back to cloning the
        val dataset spec with split='test' when no explicit test config exists."""
        if not self.is_master:
            return

        best_ckpt_path = os.path.join(self.cfg._env.save_dir, 'best-model.pth')
        if not os.path.isfile(best_ckpt_path):
            self.log("[test] No best-model.pth found, skipping test evaluation.")
            return

        test_spec = None
        if hasattr(self.cfg.datasets, 'test'):
            test_spec = copy.deepcopy(OmegaConf.to_container(self.cfg.datasets.test, resolve=True))
        elif hasattr(self.cfg.datasets, 'val'):
            test_spec = copy.deepcopy(OmegaConf.to_container(self.cfg.datasets.val, resolve=True))
            try:
                test_spec['args']['dataset']['args']['split'] = 'test'
            except (KeyError, TypeError):
                try:
                    test_spec['args']['split'] = 'test'
                except (KeyError, TypeError):
                    self.log("[test] Could not derive test split from val config, skipping.")
                    return
        else:
            self.log("[test] No val or test dataset config found, skipping.")
            return

        try:
            loader_spec = test_spec.pop('loader', {})
            test_dataset = datasets.make(test_spec)
            self.log(f"[test] Test dataset: len={len(test_dataset)}")
        except Exception as e:
            self.log(f"[test] Failed to create test dataset: {e}")
            return

        bs = int(loader_spec.get('batch_size', 32))
        nw = int(loader_spec.get('num_workers', 4))
        if getattr(test_dataset, 'is_ffcv', False):
            # FFCV path: avoids vanilla DataLoader's fork-after-CUDA-init crash.
            test_loader, _sampler = test_dataset.make_loader(
                batch_size=bs, num_workers=nw, distributed=False,
                shuffle=False, drop_last=False, device=self.device,
            )
        else:
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=bs, shuffle=False, drop_last=False,
                num_workers=nw, pin_memory=True,
            )

        self.log(f"[test] Loading best checkpoint: {best_ckpt_path}")
        ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt['model']['sd']
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing:
            self.log(f"[test] Warning: missing keys: {missing[:5]}...")
        self.model_ddp.eval()

        ave_scalars = dict()
        pbar = tqdm(test_loader, desc='test', leave=False)
        for data in pbar:
            def _to_device(x):
                if torch.is_tensor(x):
                    return x.cuda()
                if isinstance(x, (list, tuple)):
                    return [_to_device(_v) for _v in x]
                if isinstance(x, dict):
                    return {kk: _to_device(vv) for kk, vv in x.items()}
                return x
            data = _to_device(data)
            with torch.no_grad():
                ret = self.train_step(data, bp=False)

            def _batch_size_from(sample):
                if torch.is_tensor(sample):
                    return int(sample.shape[0])
                if isinstance(sample, (list, tuple)) and sample and torch.is_tensor(sample[0]):
                    return int(sample[0].shape[0])
                if isinstance(sample, dict):
                    for v in sample.values():
                        b = _batch_size_from(v)
                        if b is not None:
                            return int(b)
                return None
            bs_actual = _batch_size_from(data) or 1
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=bs_actual)

            pbar.set_description(desc=f'test: loss={ret["loss"]:.4f}')

        logtext = 'test:'
        results = {}
        for k, v in ave_scalars.items():
            val = v.item()
            if _math.isnan(val) or _math.isinf(val):
                continue
            logtext += f' {k}={_fmt_scalar(val)}'
            results[f'test/{k}'] = val
            self.log_scalar('test/' + k, val)
        self.log(logtext)

        import json
        metrics_path = os.path.join(self.cfg._env.save_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.log(f"[test] Saved metrics to {metrics_path}")

    def visualize(self):
        if not self.is_master:
            if self.distributed:
                dist.barrier()
            return
        if 'val' not in self.loaders:
            if self.distributed:
                dist.barrier()
            return
        vis_cfg = getattr(self.cfg, 'visualize', None)
        if vis_cfg is None:
            if self.distributed:
                dist.barrier()
            return
        try:
            res = int(vis_cfg.resolution) if isinstance(vis_cfg.resolution, (int, float)) else int(vis_cfg.resolution[0])
        except Exception:
            res = 32
        max_samples = int(getattr(vis_cfg, 'ds_samples', 16) or 16)
        if res <= 0 or max_samples <= 0:
            if self.distributed:
                dist.barrier()
            return

        # Optional random val subset for visualization; full-val eval is unaffected.
        vis_loader = self.loaders['val']
        try:
            vsub = getattr(self.cfg, 'visualize_subset', None)
        except Exception:
            vsub = None
        if vsub is not None and bool(getattr(vsub, 'enable', False)):
            try:
                ds = self.datasets.get('val', None)
                if ds is not None:
                    N = int(len(ds))
                    n_samples = int(getattr(vsub, 'n_samples', 0) or 0)
                    frac = float(getattr(vsub, 'frac', 0.0) or 0.0)
                    if n_samples <= 0 and frac > 0:
                        n_samples = int(max(1, round(frac * N)))
                    n_samples = int(min(max(1, n_samples), N))
                    seed = int(getattr(vsub, 'seed', 0) or 0)
                    # Optionally vary the seed each call so the vis subset rotates over training.
                    try:
                        if bool(getattr(vsub, 'shuffle_each_vis', False)):
                            seed = int(seed) + int(self.iter) * 9973
                    except Exception:
                        pass
                    rng = np.random.RandomState(seed)
                    idx = rng.choice(np.arange(N), size=n_samples, replace=False).tolist()

                    try:
                        bs = int(getattr(vsub, 'batch_size', 0) or 0)
                    except Exception:
                        bs = 0
                    if bs <= 0:
                        try:
                            bs = int(getattr(self.cfg, 'eval_batch_size', 0) or 0)
                        except Exception:
                            bs = 0
                    if bs <= 0:
                        bs = int(getattr(self.loaders['val'], 'batch_size', 1) or 1)

                    try:
                        nw = int(getattr(vsub, 'num_workers', 0) or 0)
                    except Exception:
                        nw = 0
                    if getattr(ds, 'is_ffcv', False):
                        vis_loader = ds.make_subset_loader(idx, batch_size=bs, num_workers=nw, device=self.device)
                    else:
                        sub = Subset(ds, idx)
                        vis_loader = DataLoader(sub, batch_size=bs, shuffle=False, drop_last=False, num_workers=nw, pin_memory=True)
                    self.log(f"[visualize_subset] val subset: n={n_samples}/{N} (seed={seed}), bs={bs}")
            except Exception as e:
                self.log(f"[visualize_subset] disabled due to error: {e}")

        save_dir = pathlib.Path(self.cfg._env.save_dir) / f'vis_iter_{self.iter:07d}'
        img_path, grid = visualize_reconstructions(
            self.model, vis_loader, self.device, save_dir, res,
            max_samples=max_samples, use_amp=self.use_amp, amp_dtype=self.amp_dtype,
        )
        # Temperature (ERA5) must come before 3D occ: otherwise temperature is misread as occupancy.
        if img_path is None:
            try:
                img_path, grid = visualize_reconstructions_temperature(
                    self.model, vis_loader, self.device, save_dir,
                    max_samples=min(max_samples, 4),
                    use_amp=self.use_amp, amp_dtype=self.amp_dtype,
                )
            except Exception as e:
                self.log(f"[visualize_temperature] failed: {e}")
        if img_path is None:
            try:
                img_path, grid = visualize_reconstructions_3d_occ(
                    self.model, vis_loader, self.device, save_dir,
                    vis_res=res, max_samples=min(max_samples, 4),
                    use_amp=self.use_amp, amp_dtype=self.amp_dtype,
                )
            except Exception as e:
                self.log(f"[visualize_3d_occ] failed: {e}")
        try:
            if img_path is not None:
                self.log(f"[visualize] wrote: {str(img_path)}")
            else:
                self.log(f"[visualize] no image written (img_path=None). save_dir={str(save_dir)}")
        except Exception:
            pass
        if img_path is not None:
            self.log_image('val/recon', grid)
        if self.distributed:
            dist.barrier()

    def save_ckpt(self, filename):
        if not self.is_master:
            return
        # Re-resolve model spec at save time: some trainers mutate cfg.model.args at runtime
        # (e.g. diffusion filling in "from_manifest" shapes), and cfg_dict was frozen at init.
        model_spec = copy.deepcopy(OmegaConf.to_container(self.cfg.model, resolve=True))
        model_spec['sd'] = self.model.state_dict()
        optimizers_spec = dict()
        cfg_opt = self.cfg_dict.get('optimizers', {}) or {}
        for k, opt in self.optimizers.items():
            spec = copy.copy(cfg_opt.get(k, {}))
            spec['sd'] = opt.state_dict()
            optimizers_spec[k] = spec
        ckpt = {
            'cfg': self.cfg_dict,
            'model': model_spec,
            'optimizers': optimizers_spec,
            'iter': self.iter,
            'train_loader_epoch': self.train_loader_epoch,
            'ckpt_select_v': self.ckpt_select_v,
            'amp_dtype': self.amp_dtype_str,
            'grad_scaler': (self.grad_scaler.state_dict()
                            if (self.use_amp and self.amp_dtype == torch.float16)
                            else None),
        }
        torch.save(ckpt, os.path.join(self.cfg._env.save_dir, filename))

    def sync_ave_scalars_(self, ave_scalars):
        if not self.distributed:
            return
        for k, v in ave_scalars.items():
            t = torch.tensor(v.item(), device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t.div_(self.world_size)
            ave_scalars[k].v = t.item()
            ave_scalars[k].n *= self.world_size

    def log_scalar(self, k, v):
        if self.enable_tb:
            self.writer.add_scalar(k, v, global_step=self.iter)
        if self.enable_wandb:
            wandb.log({k: v}, step=self.iter)

    def log_image(self, k, v):
        if self.enable_tb:
            self.writer.add_image(k, v, global_step=self.iter)
        if self.enable_wandb:
            wandb.log({k: wandb.Image(v)}, step=self.iter)
