# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Implement the Peacock weighting method, see [1].

[1] Neo, Dexter and Chen, Tsuhan (2025).
Multi-Objective Optimization for Deep Neural Network Calibration.
Proceedings of the IEEE/CVF International Conference on Computer Vision
https://openaccess.thecvf.com/content/ICCV2025W/Findings/papers/Neo_Multi-Objective_Optimization_for_Deep_Neural_Network_Calibration_ICCVW_2025_paper.pdf
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


class ExpMovingAverage(nn.Module):
    """Exponential moving average for a 1D Tensor."""

    mean: Tensor
    is_initialized: Tensor

    def __init__(self, lambd: float, n_values: int = 1) -> None:
        super().__init__()
        self.lambd = lambd
        self.register_buffer("mean", torch.zeros((n_values,)))
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
    def forward(self, x: Tensor) -> Tensor:
        """Update the current ema and return the new value."""
        if not self.is_initialized:
            self._initialize_buffer(x)
            return self.mean
        x_synced = self._sync_values(x)
        self.mean.mul_(1.0 - self.lambd).add_(x_synced, alpha=self.lambd)
        return self.mean


class PeacockWeighting(nn.Module):
    """Implementent the Peacock weighting strategy for MOO, see [1]."""

    def __init__(self, n_losses: int, eps: float = 1e-8):
        super().__init__()
        self.n_losses = n_losses
        self.eps = eps
        self.Wq = nn.Linear(n_losses, n_losses, bias=False)
        self.Wk = nn.Linear(n_losses, n_losses, bias=False)
        self.Wv = nn.Linear(n_losses, n_losses, bias=False)
        self.scale = n_losses**-0.5
        self.ema = ExpMovingAverage(lambd=1e-2, n_values=n_losses)

    def compute_weights(self, losses: Tensor) -> Tensor:
        """Return loss weights (computed by a single self-attention block)."""
        losses = losses.detach()
        q, k, v = self.Wq(losses), self.Wk(losses), self.Wv(losses)
        attn_scores = self.scale * torch.outer(q, k)
        weights = F.softmax(attn_scores, dim=-1) @ v
        return F.softmax(weights, dim=0)

    def compute_objective(self, weights: Tensor, losses: Tensor) -> Tensor:
        """Return Peacock's loss to learn the weights, see Eq. 14."""
        decrease_rates = self.ema(losses) - losses
        gradient_estimates = torch.sqrt(F.relu(decrease_rates) + self.eps)
        return (weights @ gradient_estimates).square()

    def forward(  # noqa: D102
        self, losses: Tensor, return_weights: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        if losses.ndim != 1 or losses.size(0) != self.n_losses:
            raise ValueError(
                f"Losses must have a shape of ({self.n_losses},); found {losses.size()}"
            )
        weights = self.compute_weights(losses.detach())
        weighted_loss = weights.detach() @ losses
        internal_loss = self.compute_objective(weights, losses.detach())
        if return_weights:
            return weighted_loss + internal_loss, weights.detach()
        return weighted_loss + internal_loss
