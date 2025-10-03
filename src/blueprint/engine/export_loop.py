# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from lightning import Fabric
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# TODO: ce fichier
@torch.no_grad()
def export_loop(
    fabric: Fabric, dl: DataLoader, model: nn.Module, writer: WriterInterface
):
    model.eval()
    if fabric.world_size > 1:
        raise ValueError("Export only works with one gpu.")

    frame_idx = 1
    n_detections = 0

    with writer:
        for batch in tqdm(dl, desc="export"):
            y = batch["y"]
            x = model(y)

            for x_, y_ in zip(x, y):
                frame = torch.full_like(x_[:, 0], frame_idx)
                data = torch.cat([frame[:, None], x_[:, :4]], dim=-1)
                writer.write(data)
                n_detections += len(x_)
                frame_idx += 1

    n_detections_mean = n_detections / frame_idx - 1
    print(f"Mean detection per frame: {n_detections_mean:0.1f}")
