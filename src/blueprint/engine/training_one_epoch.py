# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from lightning_fabric import Fabric
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from blueprint import utils
from blueprint.metrics import LazyMetricCollection, MedianOfMeans
from blueprint.utils import profiler
from blueprint.utils.progress_bar import WrappedTextProgressBar


@torch.enable_grad()
def train_one_epoch(
    fabric: Fabric,
    dl: DataLoader,
    n_accum_steps: int,
    optimizers: list[Optimizer],
    schedulers: list[LRScheduler],
    step: int,
    grad_clip_norm: float | None,
    training_module: nn.Module,
    enable_profiling: bool,
) -> tuple[dict, int]:
    """Train for one epoch."""
    if len(dl) == 0:
        return {"loss": torch.tensor(torch.nan, device=fabric.device)}, step

    n_steps = len(dl) // n_accum_steps
    n_steps += 1 if len(dl) % n_accum_steps > 0 else 0

    training_module.train()
    for opt in optimizers:
        opt.zero_grad()

    step_metrics = LazyMetricCollection(lambda: MeanMetric().to(device=fabric.device))
    epoch_metrics = LazyMetricCollection(
        lambda: MedianOfMeans(n_groups=MedianOfMeans.recommended_n_groups(n_steps)).to(
            device=fabric.device
        )
    )

    disable_stdout = not fabric.is_global_zero
    with WrappedTextProgressBar(total=n_steps, disable=disable_stdout) as bar:
        with profiler.Profiler(disable=not enable_profiling) as prof:
            for batch_idx, batch in enumerate(dl):
                is_accumulating = (batch_idx + 1) % n_accum_steps != 0
                batch["step"] = torch.tensor(step, device=fabric.device)

                with fabric.no_backward_sync(training_module, enabled=is_accumulating):
                    metrics = training_module(batch)
                    fabric.backward(metrics["loss"] / n_accum_steps)
                step_metrics.update(metrics)

                if is_accumulating:
                    continue

                metrics = step_metrics.compute()

                for opt in optimizers:
                    if grad_clip_norm is not None:
                        fabric.clip_gradients(
                            training_module, opt, max_norm=grad_clip_norm
                        )
                    opt.step()
                    opt.zero_grad()
                for i, sche in enumerate(schedulers):
                    metrics[f"lr{i}"] = sche.get_last_lr()[0]
                    sche.step()

                bar.set_text(utils.format.format_metrics({"step": step} | metrics))
                bar.update(1)
                prof.step()
                step += 1
                step_metrics.reset()
                epoch_metrics.update(metrics)

    if is_accumulating:
        for opt in optimizers:
            opt.zero_grad()

    return epoch_metrics.compute(), step
