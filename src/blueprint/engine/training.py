# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import operator
import os
import sys

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
    optimizer: Optimizer | list[Optimizer],
    scheduler: LRScheduler | list[LRScheduler],
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
):
    """Train a model for n epochs."""
    # init
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
    best_epoch = begin_epoch
    best_metric = float("-inf") if watched_metric_strategy == "max" else float("inf")
    is_better = operator.ge if watched_metric_strategy == "max" else operator.le

    optimizers = [optimizer] if isinstance(optimizer, Optimizer) else optimizer
    schedulers = [scheduler] if isinstance(scheduler, LRScheduler) else scheduler

    logger = None  # delayed initialization
    loss_history = []
    n_epochs = n_epochs if n_epochs >= 0 else sys.maxsize**10
    path_last_ckpt, path_best_ckpt = utils.checkpoint.get_ckpts_path(log_dir)
    step = begin_step
    clock = TrainingClock(steps_left=total_steps - step) if total_steps > 0 else None

    for epoch in range(begin_epoch, n_epochs):
        # train, validate
        dl_train.dataset.increment_seed()
        metrics = {}
        local_oom = False
        try:
            train_metrics, step = engine.train_one_epoch(
                fabric=fabric,
                dl=dl_train,
                n_accum_steps=n_accum_steps,
                optimizers=optimizers,
                schedulers=schedulers,
                step=step,
                total_steps=total_steps,
                clock=clock,
                grad_clip_norm=grad_clip_norm,
                training_module=training_module,
                enable_profiling=enable_profiling,
            )
            metrics = train_metrics
            if should_validate(val_stride, epoch=epoch, n_epochs=n_epochs):
                metrics |= engine.validate(fabric=fabric, model=model, dl=dl_val)
        except RuntimeError as error:
            if not utils.oom.is_oom(error):
                raise
            local_oom = True
        utils.oom.raise_on_any_rank_oom(fabric, local_oom)

        # log
        if fabric.is_global_zero:
            logs = {"epoch": epoch, "step": step} | metrics
            if logger is None:
                os.makedirs(log_dir, exist_ok=True)
                utils.logs.write_file(cfg_str, filename="config.yaml", log_dir=log_dir)
                logger = SummaryWriter(log_dir)
            for key, value in logs.items():
                if isinstance(value, Tensor) and value.nelement() > 1:
                    logger.add_tensor(tag=key, tensor=value, global_step=step)
                elif isinstance(value, Figure):
                    logger.add_figure(tag=key, figure=value, global_step=step)
                else:
                    logger.add_scalar(tag=key, scalar_value=value, global_step=step)
            print(utils.format.format_metrics(logs))

        # save
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

        # early stopping
        if patience > 0 and epoch > best_epoch + patience:
            if fabric.is_global_zero:
                print(f"Exceeded patience of {patience} epochs; early stopping.")
            break

        # divergence
        if detect_divergence and len(dl_train) != 0:
            loss_history.append(metrics["loss"])
            if utils.divergence.detect_divergence(loss_history):
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

    return best_metric


def should_validate(val_stride: int, epoch: int, n_epochs: int) -> bool:
    """Validate every val_stride epochs and at the last one; never if 0, last if <0."""
    if val_stride == 0:
        return False
    if epoch == n_epochs - 1:
        return True
    if val_stride < 0:
        return False
    return (epoch + 1) % val_stride == 0
