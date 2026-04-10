# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Implement a multi-GPU and compilation friendly exponential moving average."""

import torch
import torch.distributed as dist
from torch import Tensor, nn


class ExpMovingAverage(nn.Module):
    """Exponential moving average."""

    mean: Tensor
    is_initialized: Tensor

    def __init__(self, lambd: float, n_elements: int = 1) -> None:
        super().__init__()
        self.lambd = lambd
        self.register_buffer("mean", torch.zeros((n_elements,)))
        self.register_buffer("is_initialized", torch.tensor(False, dtype=torch.bool))

    @torch.compiler.disable
    def _sync_values(self, x: Tensor) -> Tensor:
        if dist.is_available() and dist.is_initialized():
            x = x.clone()
            dist.all_reduce(x, op=dist.ReduceOp.AVG)
        return x

    @torch.compiler.disable
    def _initialize_buffer(self, x: Tensor):
        x_synced = self._sync_values(x)
        self.mean.copy_(x_synced)
        self.is_initialized.fill_(True)

    @torch.no_grad()
    def update(self, x: Tensor):
        """Update the current ema."""
        if not self.is_initialized:
            self._initialize_buffer(x)
            return
        x_synced = self._sync_values(x)
        self.mean.mul_(1.0 - self.lambd).add_(x_synced, alpha=self.lambd)

    def value(self) -> Tensor:
        """Return the current estimate."""
        return self.mean

    @torch.no_grad()
    def forward(self, x: Tensor | None = None) -> Tensor:
        """Update the ema if x is provided, and return the (updated) value."""
        if x is not None:
            self.update(x)
        return self.value()
