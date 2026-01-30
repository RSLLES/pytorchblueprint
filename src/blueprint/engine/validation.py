# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import torch
import torchmetrics
from lightning_fabric import Fabric
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from blueprint.metrics import WassersteinDistance


@torch.no_grad()
def validate(fabric: Fabric, dl: DataLoader, model: nn.Module):
    """Compute validation metrics given a model for one dataset."""
    source, result, target = (
        torchmetrics.CatMetric(),
        torchmetrics.CatMetric(),
        torchmetrics.CatMetric(),
    )
    distance = WassersteinDistance(p=1.0)
    model.eval()
    for batch in tqdm(dl, leave=False, desc="val", disable=not fabric.is_global_zero):
        x_source = batch["sourcedist"]
        x_target = batch["targetdist"]
        x_pred = model(x_source)
        source.update(x_source)
        target.update(x_target)
        result.update(x_pred)
        distance.update(x_pred, x_target)

    x_source = source.compute().cpu().numpy()
    x_target = target.compute().cpu().numpy()
    x_res = result.compute().cpu().numpy()
    emd = distance.compute()

    fig = None
    if fabric.is_global_zero:
        plt.clf()
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].scatter(x_source[:, 0], x_source[:, 1], color="blue", alpha=0.5)
        axs[0].set_title("Source")
        axs[0].set_aspect("equal")
        axs[1].scatter(x_res[:, 0], x_res[:, 1], color="red", alpha=0.5)
        axs[1].set_title("Results")
        axs[1].set_aspect("equal")
        axs[2].scatter(x_target[:, 0], x_target[:, 1], color="orange", alpha=0.5)
        axs[2].set_title("Target")
        axs[2].set_aspect("equal")
        fig.tight_layout()
    return {"earth_mover_dist": emd, "fig": fig}
