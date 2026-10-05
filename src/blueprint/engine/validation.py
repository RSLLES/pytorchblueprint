# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from lightning.fabric import Fabric
from matplotlib.figure import Figure
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluators import Evaluator


def predict(model: nn.Module, batch: dict[str, Tensor]) -> Tensor:
    """Run the model on a validation batch; the one place tied to the batch layout."""
    return model(batch["sourcedist"])


@torch.no_grad()
def validate(
    fabric: Fabric, dl: DataLoader, model: nn.Module, evaluators: Sequence[Evaluator]
) -> dict[str, Tensor | Figure]:
    """Run every applicable evaluator over a single pass and merge their metrics."""
    model.eval()
    for evaluator in evaluators:
        evaluator.prepare(fabric, dl, model)
        evaluator.reset()

    applicable = []
    for batch_idx, batch in enumerate(
        tqdm(dl, leave=False, desc="val", disable=not fabric.is_global_zero)
    ):
        if batch_idx == 0:
            applicable = [
                e for e in evaluators if all(k in batch for k in e.REQUIRED_KEYS)
            ]
        output = predict(model, batch)
        for evaluator in applicable:
            evaluator.update(output, batch)

    metrics = {}
    for evaluator in applicable:
        metrics |= {f"val/{k}": v for k, v in evaluator.compute().items()}
    return metrics
