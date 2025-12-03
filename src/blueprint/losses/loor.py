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


class LOORLossFunc(nn.Module):
    """REINFORCE loss function with the leave-one-out variance stabilizing trick."""

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
