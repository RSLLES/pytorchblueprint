# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for checkpoints; path handling, load, save."""

import os

from lightning_fabric import Fabric
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

### Logs ###


def get_ckpts_path(log_dir: str, ckpt_dir: str | None = None) -> tuple[str, str]:
    """Return the paths to the last and best checkpoint files."""
    root = os.path.join(log_dir, ckpt_dir) if ckpt_dir is not None else log_dir
    path_best_ckpt = os.path.join(root, "best.tar")
    path_last_ckpt = os.path.join(root, "last.tar")
    return path_last_ckpt, path_best_ckpt


### Load ###


def load_training(
    fabric: Fabric,
    ckpt_path: str,
    model: Module,
    optimizer: Optimizer | list[Optimizer] | None,
    scheduler: LRScheduler | list[LRScheduler] | None,
    strict: bool = False,
) -> tuple[int, int]:
    """Load training module and optimizer states, returning epoch and step."""
    d = fabric.load(ckpt_path)
    model.load_state_dict(d["model"], strict=strict)

    if isinstance(optimizer, Optimizer):
        optimizer.load_state_dict(d["optimizer"])
    if isinstance(optimizer, list):
        for opt, opt_state in zip(optimizer, d["optimizer"]):
            opt.load_state_dict(opt_state)

    if isinstance(scheduler, LRScheduler):
        scheduler.load_state_dict(d["scheduler"])
    if isinstance(scheduler, list):
        for sche, sche_state in zip(scheduler, d["scheduler"]):
            sche.load_state_dict(sche_state)

    return d["epoch"], d["step"]


### Save ###


def save_training(
    fabric: Fabric,
    path: str,
    epoch: int,
    step: int,
    model: Module,
    optimizer: Optimizer | list[Optimizer],
    scheduler: LRScheduler | list[LRScheduler],
):
    """Save training state to a checkpoint file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "step": step,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "model": model,
    }
    fabric.save(path, state)
