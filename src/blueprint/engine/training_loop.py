# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from lightning_fabric import Fabric
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from blueprint import utils
from blueprint.metrics import MeanDictMetric
from blueprint.utils.profiler import MemoryProfiler


@torch.enable_grad()
def training_loop(
    fabric: Fabric,
    dl: DataLoader,
    n_accum_steps: int,
    opt: Optimizer,
    scheduler: LRScheduler,
    step: int,
    training_module: nn.Module,
) -> tuple[dict, int]:
    """Training loop for one epoch."""
    if len(dl) == 0:
        return {"loss": torch.tensor(torch.nan, device=fabric.device)}, step

    training_module.train()
    opt.zero_grad()
    d = 1.0  # prodigy's multiplicative constant

    avg_metrics = MeanDictMetric(device=fabric.device)

    n_steps = len(dl) // n_accum_steps
    n_steps += 1 if len(dl) % n_accum_steps > 0 else 0
    with tqdm(total=n_steps, leave=False, disable=not fabric.is_global_zero) as pbar:
        with MemoryProfiler(disable=True):
            for i, batch in enumerate(dl):
                # +1 so we start accumulating at the first step
                is_accumulating = (i + 1) % n_accum_steps != 0

                with fabric.no_backward_sync(training_module, enabled=is_accumulating):
                    metrics = training_module(batch)
                    fabric.backward(metrics["loss"] / n_accum_steps)

                if "d" in opt.param_groups[0]:
                    d = opt.param_groups[0]["d"]
                metrics["lr"] = d * scheduler.get_last_lr()[0]
                avg_metrics.update(metrics)
                pbar.desc = utils.strings.format_metrics({"step": step} | metrics)

                if not is_accumulating:
                    opt.step()
                    scheduler.step()
                    opt.zero_grad()
                    step += 1
                    pbar.update(1)

        # n_accum_steps may not divide exactly the dataset
        if is_accumulating:
            opt.step()
            opt.zero_grad()
            step += 1
            pbar.update(1)

    avg_metrics = avg_metrics.compute()
    return avg_metrics, step
