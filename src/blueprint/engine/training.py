# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

from lightning_fabric import Fabric
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from blueprint import engine, utils


def train(
    fabric: Fabric,
    model: nn.Module,
    training_module: nn.Module,
    opt: Optimizer,
    scheduler: LRScheduler,
    dl_train: DataLoader,
    dl_val: DataLoader,
    watched_metric: str,
    log_dir: str,
    cfg_str: str,
    begin_epoch: int = 0,
    begin_step: int = 0,
    n_epochs: int = -1,
    n_accum_steps: int = 1,
    patience: int = -1,
):
    """Train a model for n epochs."""
    # init
    best_epoch = begin_epoch
    best_metric = float("-inf")
    logger = None  # delayed initialization
    loss_history = []
    n_epochs = n_epochs if n_epochs >= 0 else sys.maxsize**10
    path_last_ckpt, path_best_ckpt = utils.checkpoint.get_ckpts_path(log_dir)
    step = begin_step

    for epoch in range(begin_epoch, n_epochs):
        # train, validate
        dl_train.dataset.increment_seed()
        train_metrics, step = engine.train_one_epoch(
            fabric=fabric,
            dl=dl_train,
            n_accum_steps=n_accum_steps,
            opt=opt,
            scheduler=scheduler,
            step=step,
            training_module=training_module,
        )
        val_metrics = engine.validate(fabric=fabric, model=model, dl=dl_val)
        metrics = train_metrics | val_metrics

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
            print(utils.strings.format_metrics(logs))

        # save
        utils.checkpoint.save_training(
            fabric=fabric,
            path=path_last_ckpt,
            epoch=epoch,
            step=step,
            optimizer_state_dict=opt.state_dict(),
            scheduler_state_dict=scheduler.state_dict(),
            training_module_state_dict=training_module.state_dict(),
        )
        if metrics[watched_metric] >= best_metric:
            best_epoch = epoch
            best_metric = metrics[watched_metric]
            utils.checkpoint.save_training(
                fabric=fabric,
                path=path_best_ckpt,
                epoch=epoch,
                step=step,
                optimizer_state_dict=opt.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
                training_module_state_dict=training_module.state_dict(),
            )

        # early stopping
        if patience > 0 and epoch > best_epoch + patience:
            if fabric.is_global_zero:
                print(f"Exceeded patience of {patience} epochs; early stopping.")
            break

        # divergence
        loss_history.append(metrics["loss"])
        if len(dl_train) != 0 and utils.divergence.detect_divergence(loss_history):
            if fabric.is_global_zero:
                print("Divergence detected. Reverting to previous best weights.")
            utils.checkpoint.load_training(
                fabric=fabric,
                ckpt_path=path_best_ckpt,
                optimizer=opt,
                scheduler=scheduler,
                training_module=training_module,
            )
            loss_history = []

    return best_metric
