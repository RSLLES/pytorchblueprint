# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Moment matching training step."""

import torch
from torch import Tensor, nn

from blueprint.losses.ot import sinkhorn_knopp_log


class WasserteinMatchingTrainer(nn.Module):
    """Train a model with Hyvarinen Score Matching."""

    def __init__(self, model: nn.Module, reg: float, n_iters: int):
        super().__init__()
        self.model = model
        self.reg = reg
        self.n_iters = n_iters

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute one training step loss for a batch of distributions."""
        x_source = x["sourcedist"]
        x_target = x["targetdist"]
        x_pred = self.model(x_source)

        B = x_source.size(0)
        cost_matrix = torch.cdist(x_pred, x_target, p=2.0)
        reg = self.reg * cost_matrix.detach().median()
        uniform_weights = torch.full_like(cost_matrix[0], 1 / B)
        transport_plan = sinkhorn_knopp_log(
            M=cost_matrix[None],
            a=uniform_weights[None],
            b=uniform_weights[None],
            reg=reg,
            n_iters=self.n_iters,
        )
        loss = transport_plan.flatten() @ cost_matrix.flatten()
        return {"loss": loss}
