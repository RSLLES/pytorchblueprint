# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Implementation of Maximum Mean Discrepancy (MMD) as a Pytorch loss.

See [1] for details about (linear) MMD.
Code based on https://github.com/yiftachbeer/mmd_loss_pytorch/tree/master

[1] Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
A kernel two-sample test. The journal of machine learning research, 13(1), 723-773.
https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
"""

import torch
from torch import Tensor, nn


class RBF(nn.Module):
    """Radial Base function kernel."""

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.register_buffer("bandwidth_multipliers", bandwidth_multipliers)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        """Return the median of the L2 distances matrix by default."""
        if self.bandwidth is not None:
            return self.bandwidth
        return L2_distances.median()

    def forward(self, z1, z2):  # noqa: D102
        L2_distances = torch.cdist(z1, z2, p=2.0).square()
        sigma2 = self.get_bandwidth(L2_distances) * self.bandwidth_multipliers
        return torch.exp(-L2_distances[..., None] / sigma2).sum(dim=-1)


class MMDLinearLossFunc(nn.Module):
    """Implementation of Linear MMD."""

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    @torch.compiler.disable
    def forward(self, x: Tensor, y: Tensor, mask_x: Tensor, mask_y: Tensor):  # noqa: D102
        x, y = x[mask_x], y[mask_y]
        bs = min(x.size(0), y.size(0))
        bs = bs - (bs % 2)
        x, y = x[:bs], y[:bs]

        if y.size(0) == 0 or x.size(0) == 0:
            return torch.tensor((0.0), device=x.device)

        x = x.reshape(bs // 2, 2, -1)
        y = y.reshape(bs // 2, 2, -1)
        z1 = torch.stack([x[:, 0], y[:, 0]], dim=1)
        z2 = torch.stack([x[:, 1], y[:, 1]], dim=1)

        K = self.kernel(z1, z2)
        MMD = K[:, 0, 0] + K[:, 1, 1] - K[:, 0, 1] - K[:, 1, 0]
        return MMD.mean()
