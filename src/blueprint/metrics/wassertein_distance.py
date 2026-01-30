# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Wasserstein distance as a torchmetric."""

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


class WassersteinDistance(Metric):
    """Compute the p-Wasserstein distance between predictions and targets."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, p: float = 1.0):
        super().__init__()
        self.add_state("x_pred", default=[], dist_reduce_fx="cat")
        self.add_state("x_target", default=[], dist_reduce_fx="cat")
        self.p = p

    def update(self, x_pred: Tensor, x_target: Tensor):  # noqa: D102
        self.x_pred.append(x_pred)
        self.x_target.append(x_target)

    def compute(self) -> Tensor:  # noqa: D102
        x_pred = dim_zero_cat(self.x_pred)
        x_target = dim_zero_cat(self.x_target)
        dists = torch.cdist(x_pred, x_target, p=2.0)
        cost_matrix = dists.pow(self.p)
        idx_pred, idx_target = linear_sum_assignment(cost_matrix.cpu().numpy())
        distance = cost_matrix[idx_pred, idx_target].mean()
        return distance
