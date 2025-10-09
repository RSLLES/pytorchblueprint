# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Earth Mover's distance as a torchmetric."""

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchmetrics import Metric


def dim_zero_cat(x: list | Tensor) -> Tensor:
    """Concatenate along the first dimension."""
    if isinstance(x, torch.Tensor):
        return x
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)


class EarthMoverDistance(Metric):
    """Compute Earth Mover's Distance between predictions and targets."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("x_pred", default=[], dist_reduce_fx="cat")
        self.add_state("x_target", default=[], dist_reduce_fx="cat")

    def update(self, x_pred: Tensor, x_target: Tensor):  # noqa: D102
        self.x_pred.append(x_pred)
        self.x_target.append(x_target)

    def compute(self) -> Tensor:  # noqa: D102
        x_pred = dim_zero_cat(self.x_pred)
        x_target = dim_zero_cat(self.x_target)
        M = x_pred[:, None] - x_target[None, :]
        M = torch.square(M).sum(dim=-1)
        idx_pred, idx_target = linear_sum_assignment(M.cpu().numpy())
        distance = M[idx_pred, idx_target].mean()
        return distance
