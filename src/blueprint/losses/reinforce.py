# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Implementation of Leave-one-out REINFORCE algorithm.

See [1] for information about REINFORCE, and [2] for the LOO - variance reduction trick.

[1] Williams, R.J. Simple statistical gradient-following algorithms for connectionist
reinforcement learning. Mach Learn 8, 229-256 (1992). https://doi.org/10.1007/BF00992696

[2] Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a
baseline for free! ICLR, 2019. https://openreview.net/forum?id=r1lgTGL5DE
"""

import torch
from torch import Tensor, nn

from blueprint.utils.ema import ExpMovingAverage


class BaselineReinforce(nn.Module):
    """REINFORCE loss function with the ema-baseline variance stabilizing trick [1]."""

    def __init__(
        self,
        lambd: float = 0.01,
        eps: float = 1e-9,
        ensure_detached_loss: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.ensure_detached_loss = ensure_detached_loss
        self.ema_mean = ExpMovingAverage(lambd=lambd, n_elements=1)
        self.ema_var = ExpMovingAverage(lambd=lambd, n_elements=1)

    def forward(self, loss: Tensor, log_prob: Tensor) -> Tensor:  # noqa: D102
        if self.ensure_detached_loss:
            loss = loss.detach()

        reward = loss.mean()

        if self.ema_mean.is_initialized:
            mean = self.ema_mean.value()
            std = self.ema_var.value().sqrt()
            advantage = (loss - mean) / (std + self.eps)
        else:
            advantage = loss - loss.mean()

        self.ema_mean.update(reward)
        residual_sq = (reward - self.ema_mean.value()).pow(2)
        self.ema_var.update(residual_sq)

        return (advantage * log_prob).mean()


class LOOReinforce(nn.Module):
    """REINFORCE loss function with the leave-one-out variance stabilizing trick [2]."""

    def __init__(
        self,
        ensure_detached_loss: bool = True,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.ensure_detached_loss = ensure_detached_loss
        self.eps = eps

    def forward(self, losses: Tensor, log_prob: Tensor):  # noqa: D102
        if losses.ndim != 1:
            raise ValueError(f"Expect 1D tensor for losses, got {losses.shape}")
        if losses.size(0) < 3:
            raise ValueError(f"Expect at least 3 losses, got {losses.size(0)}")
        N = losses.size(0)

        if self.ensure_detached_loss:
            losses = losses.detach()

        rolls = losses.repeat(2).as_strided((N, N), (1, 1))
        loo = rolls[:, 1:]
        std, mean = torch.std_mean(loo, dim=1)
        losses = (losses - mean) / (std + self.eps) * log_prob
        return losses.mean()
