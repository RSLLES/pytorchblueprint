# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import operator
import os
from typing import Any

from lightning.fabric import Fabric
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from blueprint import engine, utils
from blueprint.utils.progress_bar import TrainingClock


def train(
    fabric: Fabric,
    model: nn.Module,
    training_module: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    dl_train: DataLoader,
    dl_val: DataLoader,
    watched_metric: str | None,
    watched_metric_strategy: str,
    log_dir: str,
    cfg_str: str,
    begin_epoch: int = 0,
    begin_step: int = 0,
    n_epochs: int = -1,
    total_steps: int = -1,
    n_accum_steps: int = 1,
    patience: int = -1,
    val_stride: int = 1,
    grad_clip_norm: float | None = None,
    detect_divergence: bool = True,
    enable_profiling: bool = False,
) -> float:
    """Train for n epochs, or until interrupted when n_epochs is negative."""
    watched_metric_strategy = utils.format.format_string(watched_metric_strategy)
    if watched_metric_strategy not in ["min", "max"]:
        raise ValueError(
            "watched_metric_strategy must be 'min' or 'max', "
            f"found {watched_metric_strategy}."
        )
    if watched_metric is None and (detect_divergence or patience > 0):
        raise ValueError(
            "detect_divergence and patience need a watched_metric: they rely on the "
            "best checkpoint."
        )
    best_epoch: int | None = None
    best_metric = float("-inf") if watched_metric_strategy == "max" else float("inf")
    is_better = operator.ge if watched_metric_strategy == "max" else operator.le

    logger = None  # delayed initialization
    loss_history = []
    path_last_ckpt, path_best_ckpt = utils.checkpoint.get_ckpts_path(log_dir)
    step = begin_step
    clock = TrainingClock(steps_left=total_steps - step) if total_steps > 0 else None
    takes_optimizer_steps = len(dl_train) >= n_accum_steps

    epochs = (
        range(begin_epoch, n_epochs) if n_epochs >= 0 else itertools.count(begin_epoch)
    )
    for epoch in epochs:
        dl_train.dataset.increment_seed()
        metrics = {}
        diverged = False
        local_oom = False
        try:
            metrics, step = engine.train_one_epoch(
                fabric=fabric,
                dl=dl_train,
                n_accum_steps=n_accum_steps,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                total_steps=total_steps,
                clock=clock,
                grad_clip_norm=grad_clip_norm,
                training_module=training_module,
                enable_profiling=enable_profiling,
            )
            if detect_divergence and takes_optimizer_steps:
                loss_history.append(metrics["loss"])
                diverged = utils.divergence.detect_divergence(loss_history)
            if not diverged and should_validate(val_stride, epoch, n_epochs):
                metrics |= engine.validate(fabric=fabric, model=model, dl=dl_val)
        except RuntimeError as error:
            if not utils.oom.is_oom(error):
                raise
            local_oom = True
        utils.oom.raise_on_any_rank_oom(fabric, local_oom)

        if fabric.is_global_zero:
            if logger is None:
                os.makedirs(log_dir, exist_ok=True)
                utils.logs.write_file(cfg_str, filename="config.yaml", log_dir=log_dir)
                logger = SummaryWriter(log_dir)
            _log_epoch(logger, {"epoch": epoch, "step": step} | metrics, step)

        if diverged:
            if best_epoch is None:
                raise RuntimeError(
                    "Training diverged before a best checkpoint was saved."
                )
            if fabric.is_global_zero:
                print("Divergence detected. Reverting to previous best weights.")
            utils.checkpoint.load_training(
                fabric=fabric,
                ckpt_path=path_best_ckpt,
                optimizer=optimizer,
                scheduler=scheduler,
                model=training_module,
            )
            loss_history = []
            continue

        utils.checkpoint.save_training(
            fabric=fabric,
            path=path_last_ckpt,
            epoch=epoch,
            step=step,
            optimizer=optimizer,
            scheduler=scheduler,
            model=training_module,
        )
        if watched_metric in metrics and is_better(
            metrics[watched_metric], best_metric
        ):
            best_epoch = epoch
            best_metric = metrics[watched_metric]
            utils.checkpoint.save_training(
                fabric=fabric,
                path=path_best_ckpt,
                epoch=epoch,
                step=step,
                optimizer=optimizer,
                scheduler=scheduler,
                model=training_module,
            )

        if patience > 0 and best_epoch is not None and epoch > best_epoch + patience:
            if fabric.is_global_zero:
                print(f"Exceeded patience of {patience} epochs; early stopping.")
            break

    if logger is not None:
        logger.close()
    return best_metric


def _log_epoch(logger: SummaryWriter, logs: dict[str, Any], step: int) -> None:
    for key, value in logs.items():
        if isinstance(value, Tensor) and value.nelement() > 1:
            logger.add_tensor(tag=key, tensor=value, global_step=step)
        elif isinstance(value, Figure):
            logger.add_figure(tag=key, figure=value, global_step=step)
        else:
            logger.add_scalar(tag=key, scalar_value=value, global_step=step)
    print(utils.format.format_metrics(logs))


def should_validate(val_stride: int, epoch: int, n_epochs: int) -> bool:
    """Validate every val_stride epochs and at the last one; never if 0, last if <0."""
    if val_stride == 0:
        return False
    if epoch == n_epochs - 1:
        return True
    if val_stride < 0:
        return False
    return (epoch + 1) % val_stride == 0
