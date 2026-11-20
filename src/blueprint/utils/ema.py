# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Implement a multi-GPU and compilation friendly exponential moving average.

The bias correction follows Adam [1].

[1] Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization.
ICLR, 2015. https://arxiv.org/abs/1412.6980
"""

import torch
from torch import Tensor, nn

from blueprint.utils.distributed import reduce_across_ranks


class ExpMovingAverage(nn.Module):
    """Exponential moving average, bias-corrected as in Adam [1]."""

    weighted_sum: Tensor
    total_weight: Tensor

    def __init__(self, lambd: float, n_elements: int = 1) -> None:
        super().__init__()
        self.lambd = lambd
        self.register_buffer("weighted_sum", torch.zeros((n_elements,)))
        self.register_buffer("total_weight", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        """Update the current ema."""
        x_synced = reduce_across_ranks(x)
        self.weighted_sum.mul_(1.0 - self.lambd).add_(x_synced, alpha=self.lambd)
        self.total_weight.mul_(1.0 - self.lambd).add_(self.lambd)

    def value(self) -> Tensor:
        """Return the current estimate, NaN before the first update."""
        return self.weighted_sum / self.total_weight

    @torch.no_grad()
    def forward(self, x: Tensor | None = None) -> Tensor:
        """Update the ema if x is provided, and return the (updated) value."""
        if x is not None:
            self.update(x)
        return self.value()
