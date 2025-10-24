# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Uncertainty weighting for regression, as described in kendall2018multi."""

import torch
import torch.distributed as dist
from torch import Tensor, nn


class RegressionUncertaintyWeighting(nn.Module):
    """Compute a weighted sum of losses based on learned uncertainties."""

    def __init__(self, n_losses: int, eps: float = 1e-9):
        super().__init__()
        if n_losses < 1:
            raise ValueError(f"n_losses must be >=1, got {n_losses}.")
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer("log_sigma_ref", torch.empty((n_losses,)))
        self.log_sigma_weights = nn.Parameter(torch.ones((n_losses,)))
        self.eps = eps

    @torch.compiler.disable()
    @torch.no_grad()
    def initialize(self, losses: Tensor):
        """Initialize log_sigma_ref once at the first pass."""
        init = 0.5 * torch.log(losses.abs() + self.eps)
        if dist.is_initialized():
            dist.broadcast(init, src=0)
        self.log_sigma_ref.copy_(init)
        self.initialized.fill_(True)

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """Return the weighted loss sum."""
        if losses.dim() > 1 or losses.numel() != self.log_sigma_ref.numel():
            raise ValueError(f"Wrong losses shape: {losses.size()}")

        if not self.initialized:
            self.initialize(losses)
        log_sigma = self.log_sigma_ref * self.log_sigma_weights
        losses = 0.5 * torch.exp(-2.0 * log_sigma) * losses + log_sigma
        loss = losses.sum()
        return loss
