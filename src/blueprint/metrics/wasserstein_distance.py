# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Wasserstein distance as a torchmetric."""

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


class WassersteinDistance(Metric):
    """p-Wasserstein distance between equally sized sets of predictions and targets.

    The optimal transport between two uniform empirical measures of the same size is
    an assignment, which scipy solves exactly.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False
    x_pred: list[Tensor]
    x_target: list[Tensor]

    def __init__(self, p: float = 1.0):
        super().__init__()
        self.add_state("x_pred", default=[], dist_reduce_fx="cat")
        self.add_state("x_target", default=[], dist_reduce_fx="cat")
        self.p = p

    def update(self, x_pred: Tensor, x_target: Tensor) -> None:  # noqa: D102
        self.x_pred.append(x_pred)
        self.x_target.append(x_target)

    def compute(self) -> Tensor:  # noqa: D102
        x_pred = dim_zero_cat(self.x_pred)
        x_target = dim_zero_cat(self.x_target)
        dists = torch.cdist(x_pred, x_target, p=2.0)
        cost_matrix = dists.pow(self.p)
        idx_pred, idx_target = linear_sum_assignment(cost_matrix.cpu().numpy())
        return cost_matrix[idx_pred, idx_target].mean() ** (1 / self.p)
