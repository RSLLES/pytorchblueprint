# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import torch
import torchmetrics
from lightning_fabric import Fabric
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def validation_loop(
    fabric: Fabric, dl: DataLoader, model: nn.Module, desc: str = "validation"
):
    source, target = torchmetrics.CatMetric(), torchmetrics.CatMetric()
    model.eval()
    for batch in tqdm(dl, leave=False, desc=desc, disable=not fabric.is_global_zero):
        x0 = batch
        x1 = model(x0)
        source.update(x0)
        target.update(x1)

    x0 = source.compute().cpu().numpy()
    x1 = target.compute().cpu().numpy()

    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].scatter(x0[:, 0], x0[:, 1], color="blue", alpha=0.5)
    axs[0].set_title("Source")
    axs[0].set_aspect("equal")
    axs[1].scatter(x1[:, 0], x1[:, 1], color="orange", alpha=0.5)
    axs[1].set_title("Target")
    axs[1].set_aspect("equal")
    plt.tight_layout()
    return {"fig": fig}
