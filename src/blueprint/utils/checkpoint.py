# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for checkpoints; path handling, load, save."""

import os

from lightning.fabric import Fabric
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
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    strict: bool = True,
) -> tuple[int, int, float, int | None]:
    """Load training states, returning epoch, step, best metric and best epoch."""
    d = fabric.load(ckpt_path)
    model.load_state_dict(d["model"], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(d["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(d["scheduler"])
    return d["epoch"], d["step"], d["best_metric"], d["best_epoch"]


### Save ###


def save_training(
    fabric: Fabric,
    path: str,
    epoch: int,
    step: int,
    best_metric: float,
    best_epoch: int | None,
    model: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
) -> None:
    """Save training state to a checkpoint file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "model": model,
    }
    tmp_path = f"{path}.tmp"
    fabric.save(tmp_path, state)
    if fabric.is_global_zero:
        os.replace(tmp_path, path)
