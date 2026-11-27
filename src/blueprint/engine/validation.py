# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from lightning.fabric import Fabric
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics import CatMetric
from tqdm import tqdm

from blueprint.metrics import WassersteinDistance
from blueprint.utils.progress_bar import TrainingClock


@torch.no_grad()
def validate(
    fabric: Fabric,
    dl: DataLoader,
    model: nn.Module,
    clock: TrainingClock | None = None,
    max_batches: int | None = None,
) -> dict[str, Tensor | Figure]:
    """Compute the earth mover's distance and a scatter plot of the predictions."""
    source, result, target = CatMetric(), CatMetric(), CatMetric()
    distance = WassersteinDistance(p=1.0)
    for metric in (source, result, target, distance):
        metric.to(fabric.device)
    precision = fabric.strategy.precision
    model.eval()
    batches = itertools.islice(dl, max_batches)
    total = len(dl) if max_batches is None else min(max_batches, len(dl))
    disable = not fabric.is_global_zero
    for batch in tqdm(batches, total=total, leave=False, desc="val", disable=disable):
        x_source, x_target = batch["sourcedist"], batch["targetdist"]
        with precision.forward_context():
            x_pred = model(precision.convert_input(x_source))
        x_pred = precision.convert_output(x_pred)
        source.update(x_source)
        result.update(x_pred)
        target.update(x_target)
        distance.update(x_pred, x_target)
        if clock is not None:
            clock.count_val_batch()

    metrics: dict[str, Tensor | Figure] = {"earth_mover_dist": distance.compute()}
    panels = [
        ("Source", source.compute(), "blue"),
        ("Results", result.compute(), "red"),
        ("Target", target.compute(), "orange"),
    ]
    if not fabric.is_global_zero:
        return metrics
    # a bare Figure is not registered with pyplot, so nothing has to close it
    fig = Figure(figsize=(12, 4))
    for ax, (title, x, color) in zip(fig.subplots(1, 3), panels, strict=True):
        x = x.cpu().numpy()
        ax.scatter(x[:, 0], x[:, 1], color=color, alpha=0.5)
        ax.set_title(title)
        ax.set_aspect("equal")
    fig.tight_layout()
    return metrics | {"fig": fig}
