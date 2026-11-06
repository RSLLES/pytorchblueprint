# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from lightning.fabric import Fabric
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchmetrics import CatMetric
from tqdm import tqdm

from blueprint.metrics import WassersteinDistance


@torch.no_grad()
def validate(
    fabric: Fabric, dl: DataLoader, model: nn.Module
) -> dict[str, Tensor | Figure]:
    """Compute the earth mover's distance and a scatter plot of the predictions."""
    source, result, target = CatMetric(), CatMetric(), CatMetric()
    distance = WassersteinDistance(p=1.0)
    for metric in (source, result, target, distance):
        metric.to(fabric.device)
    model.eval()
    for batch in tqdm(dl, leave=False, desc="val", disable=not fabric.is_global_zero):
        x_source, x_target = batch["sourcedist"], batch["targetdist"]
        x_pred = model(x_source)
        source.update(x_source)
        result.update(x_pred)
        target.update(x_target)
        distance.update(x_pred, x_target)

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
