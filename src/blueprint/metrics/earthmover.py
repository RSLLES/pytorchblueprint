# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchmetrics import Metric


class EarthMoverDistance(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self):
        super().__init__(sync_on_compute=True)
        self.add_state("x_pred", default=[], dist_reduce_fx="cat")
        self.add_state("x_target", default=[], dist_reduce_fx="cat")

    def update(self, x_pred: Tensor, x_target: Tensor):
        self.x_pred.append(x_pred)
        self.x_target.append(x_target)

    def compute(self) -> Tensor:
        x_pred = torch.cat(self.x_pred, dim=0)
        x_target = torch.cat(self.x_target, dim=0)
        M = x_pred[:, None] - x_target[None, :]
        M = torch.square(M).sum(dim=-1)
        idx_pred, idx_target = linear_sum_assignment(M.cpu().numpy())
        distance = M[idx_pred, idx_target].mean()
        return distance
