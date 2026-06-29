import os
import shutil
import time
import logging
import ast

import numpy as np
from torch.optim import SGD, Adam, AdamW
from tensorboardX import SummaryWriter


def ensure_path(path, replace=True, force_replace=False):
    """
    Ensure `path` is a usable directory.

    Behavior:
      - If `path` does not exist: `os.makedirs(path)`.
      - If `path` exists and `replace=False`: leave it alone.
      - If `path` exists and `replace=True` and either `force_replace=True` or
        the path basename starts with '_' (treated as scratch): wipe and recreate.
      - Otherwise (path exists, replace=True, but neither force_replace nor scratch):
        raise a hard error. We never prompt — sweep/SLURM jobs are non-interactive
        and silent overwrites are dangerous (this is what wiped a paper-winning
        checkpoint once before).
    """
    is_temp = os.path.basename(path.rstrip('/')).startswith('_')
    if not os.path.exists(path):
        os.makedirs(path)
        return
    if not replace:
        return
    if is_temp or force_replace:
        shutil.rmtree(path)
        os.mkdir(path)
        return
    raise FileExistsError(
        f"ensure_path: {path!r} already exists and replace=True, force_replace=False. "
        "Refusing to overwrite. Either pick a unique exp_name (auto_unique=True), "
        "set _env.resume_mode='force_replace' to wipe explicitly, "
        "or set _env.resume_mode='resume' to reuse the directory."
    )


def set_logger(file_path):
    logger = logging.getLogger()
    logger.setLevel('INFO')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_path, 'a')
    formatter = logging.Formatter('[%(asctime)s] %(message)s', '%m-%d %H:%M:%S')
    for handler in [stream_handler, file_handler]:
        handler.setFormatter(formatter)
        handler.setLevel('INFO')
        logger.addHandler(handler)
    return logger


def set_save_dir(save_dir, replace=True):
    ensure_path(save_dir, replace=replace)
    logger = set_logger(os.path.join(save_dir, 'log.txt'))
    writer = SummaryWriter(os.path.join(save_dir, 'tensorboard'))
    return logger, writer


def compute_num_params(model, text=True):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            s = '{:.1f}M'.format(tot / 1e6)
        else:
            s = '{:.1f}K'.format(tot / 1e3)
        return f'{s} ({tot})'
    else:
        return tot


def make_optimizer(params, optimizer_spec, load_sd=False):
    # W&B sweeps / CLI overrides can sometimes pass optimizer args as strings
    # (e.g. lr="5e-05"). Torch optimizers expect proper Python scalars/tuples.
    args = dict(optimizer_spec.get('args', {}))
    for k, v in list(args.items()):
        if isinstance(v, str):
            try:
                args[k] = ast.literal_eval(v)
            except Exception:
                # leave as-is if it isn't a literal (e.g. a path string)
                pass

    optimizer = {
        'sgd': SGD,
        'adam': Adam,
        'adamw': AdamW,
    }[optimizer_spec['name']](params, **args)
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class EpochTimer():

    def __init__(self, max_epoch):
        self.max_epoch = max_epoch
        self.epoch = 0
        self.t_start = time.time()
        self.t_last = self.t_start

    def epoch_done(self):
        t_cur = time.time()
        self.epoch += 1
        epoch_time = t_cur - self.t_last
        tot_time = t_cur - self.t_start
        est_time = tot_time / self.epoch * self.max_epoch
        self.t_last = t_cur
        return time_text(epoch_time), time_text(tot_time), time_text(est_time)


def time_text(sec):
    if sec >= 3600:
        return f'{sec / 3600:.1f}h'
    elif sec >= 60:
        return f'{sec / 60:.1f}m'
    else:
        return f'{sec:.1f}s'
