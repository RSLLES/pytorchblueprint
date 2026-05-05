# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Implementation of Leave-one-out REINFORCE algorithm.

See [1] for information about REINFORCE, and [2] for the LOO - variance reduction trick.

[1] Williams, R.J. Simple statistical gradient-following algorithms for connectionist
reinforcement learning. Mach Learn 8, 229-256 (1992). https://doi.org/10.1007/BF00992696

[2] Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a
baseline for free! ICLR, 2019. https://openreview.net/forum?id=r1lgTGL5DE
"""

from torch import Tensor, nn

from blueprint.utils.ema import ExpMovingAverage


class BaselineRLLossFunc(nn.Module):
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


class LOORLossFunc(nn.Module):
    """REINFORCE loss function with the leave-one-out variance stabilizing trick [2]."""

    def __init__(self, ensure_detached_loss: bool = True):
        super().__init__()
        self.ensure_detached_loss = ensure_detached_loss

    def forward(self, loss: Tensor, log_prob: Tensor):  # noqa: D102
        if self.ensure_detached_loss:
            loss = loss.detach()
        loss = leave_one_out(loss) * log_prob
        return loss.mean()


def leave_one_out(y: Tensor) -> Tensor:
    """Return y_k minus the mean of all other elements."""
    if y.ndim != 1 or y.size(0) < 2:
        raise ValueError("y must be a 1D tensor with at least two elements.")
    return (y.size(0) * y - y.sum()) / (y.size(0) - 1)
