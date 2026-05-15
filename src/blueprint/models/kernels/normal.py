# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Normal distribution module with data-adaptive bandwidth selection."""

import torch
from torch import Tensor, nn

from blueprint.utils.ema import ExpMovingAverage


class NormalDistribution(nn.Module):
    """Normal sampler with multi-bandwidth scaling for RFF features.

    Bandwidths follow the same quantile-EMA recipe as ``ExponentialKernel``,
    but the pairwise distances are estimated on a random subsample of size
    ``n_subsample`` per side instead of the full N x N matrix.
    """

    uniform_grid: Tensor
    quantiles: Tensor

    def __init__(
        self,
        n_kernels: int = 5,
        n_subsample: int = 128,
        ema_lambd: float = 1e-3,
        eps: float = 1e-9,
    ):
        super().__init__()
        self.n_subsample = n_subsample
        self.eps = eps
        self.register_buffer("uniform_grid", torch.linspace(0.0, 1.0, n_kernels))
        self.register_buffer("quantiles", torch.tensor([0.05, 0.95]))
        self.ema_q05 = ExpMovingAverage(ema_lambd)
        self.ema_q95 = ExpMovingAverage(ema_lambd)

    def _subsample(self, z: Tensor) -> Tensor:
        N = z.size(-2)
        n = min(self.n_subsample, N)
        idx = torch.randperm(N, device=z.device)[:n]
        return z.index_select(-2, idx)

    @torch.no_grad
    def compute_bandwidths(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Interpolate bandwidths between 0.5*q05 and 2.0*q95 on subsampled dists."""
        z1_sub = self._subsample(z1)
        z2_sub = self._subsample(z2)
        dists = torch.cdist(z1_sub, z2_sub, p=2.0)
        dists_no_zeros = torch.where(dists > self.eps, dists, float("nan"))
        q05, q95 = torch.nanquantile(dists_no_zeros, self.quantiles)
        q05, q95 = self.ema_q05(q05), self.ema_q95(q95)
        low, high = 0.5 * q05, 2.0 * q95
        return (high - low) * self.uniform_grid + low

    def forward(self, z1: Tensor, z2: Tensor, n_features: int) -> Tensor:
        """Sample RFF frequencies of shape ``[n_features, D, n_kernels]``."""
        D = z1.size(-1)
        bandwidths = self.compute_bandwidths(z1, z2)
        w = torch.randn(
            n_features, D, bandwidths.numel(), device=z1.device, dtype=z1.dtype
        )
        return w / bandwidths
