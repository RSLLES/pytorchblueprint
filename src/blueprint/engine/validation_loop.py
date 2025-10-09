# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import torch
import torchmetrics
from lightning_fabric import Fabric
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from blueprint.metrics import EarthMoverDistance


@torch.no_grad()
def validation_loop(fabric: Fabric, dl: DataLoader, model: nn.Module):
    """Compute validation metrics given a model for one dataset."""
    source, dest = torchmetrics.CatMetric(), torchmetrics.CatMetric()
    distance = EarthMoverDistance()
    model.eval()
    for batch in tqdm(dl, leave=False, desc="val", disable=not fabric.is_global_zero):
        x_source = batch["sourcedist"]
        x_target = batch["targetdist"]
        x_pred = model(x_source)
        source.update(x_source)
        dest.update(x_pred)
        distance.update(x_pred, x_target)

    x0 = source.compute().cpu().numpy()
    x1 = dest.compute().cpu().numpy()
    emd = distance.compute()

    fig = None
    if fabric.is_global_zero:
        plt.clf()
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].scatter(x0[:, 0], x0[:, 1], color="blue", alpha=0.5)
        axs[0].set_title("Source")
        axs[0].set_aspect("equal")
        axs[1].scatter(x1[:, 0], x1[:, 1], color="orange", alpha=0.5)
        axs[1].set_title("Target")
        axs[1].set_aspect("equal")
        fig.tight_layout()
    return {"earth_mover_dist": emd, "fig": fig}
