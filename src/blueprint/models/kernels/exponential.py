# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""Implement a general kernel of the form exp(- ||x - y||^\alpha)."""

import torch
from torch import Tensor, nn

from blueprint.utils.ema import ExpMovingAverage
from blueprint.utils.torch import reduce


class ExponentialKernel(nn.Module):
    """Kernel that includes both Laplace Kernel (p=1) and Gaussian Kernel (p=2)."""

    uniform_grid: Tensor
    quantiles: Tensor

    def __init__(self, p: float | int, n_kernels: int = 5, eps: float = 1e-9):
        super().__init__()
        self.p = p
        self.eps = eps
        self.register_buffer("uniform_grid", torch.linspace(0.0, 1.0, n_kernels))
        self.register_buffer("quantiles", torch.tensor([0.05, 0.95]))
        self.ema_q05 = ExpMovingAverage(1e-3)
        self.ema_q95 = ExpMovingAverage(1e-3)

    def compute_bandwidths(self, dists: Tensor) -> Tensor:
        """Interpolates bandwidths between 0.5*q05 and 2.0*q95., see [3]."""
        dists = dists.detach()
        # classic masking resizes dists_no_zeros dynamically which crashes the compiler.
        dists_no_zeros = torch.where(dists > self.eps, dists, float("nan"))
        q05, q95 = torch.nanquantile(dists_no_zeros, self.quantiles)
        q05, q95 = self.ema_q05(q05), self.ema_q95(q95)
        low, high = 0.5 * q05, 2.0 * q95
        return (high - low) * self.uniform_grid + low

    def forward(self, z1: Tensor, z2: Tensor, reduction: str = "mean") -> Tensor:  # noqa: D102
        dists = torch.cdist(z1, z2, p=self.p).pow(self.p)
        kernels = torch.exp(-dists[..., None] / self.compute_bandwidths(dists))
        return reduce(kernels, dim=-1, mode=reduction)


class LaplaceGaussianKernel(nn.Module):
    """Mixed kernel combining Laplace (L1) and Gaussian (squared L2)."""

    uniform_grid: Tensor
    quantiles: Tensor

    def __init__(self, n_kernels: int = 5, ema_lambd: float = 1e-4, eps: float = 1e-9):
        super().__init__()
        self.eps = eps
        self.register_buffer("uniform_grid", torch.linspace(0.0, 1.0, n_kernels))
        self.register_buffer("quantiles", torch.tensor([0.05, 0.95]))
        self.ema_l1_q05 = ExpMovingAverage(ema_lambd)
        self.ema_l1_q95 = ExpMovingAverage(ema_lambd)
        self.ema_l2_q05 = ExpMovingAverage(ema_lambd)
        self.ema_l2_q95 = ExpMovingAverage(ema_lambd)

    def _bandwidths(
        self, dists: Tensor, ema_q05: ExpMovingAverage, ema_q95: ExpMovingAverage
    ) -> Tensor:
        dists_no_zeros = torch.where(dists > self.eps, dists, float("nan"))
        q05, q95 = torch.nanquantile(dists_no_zeros, self.quantiles)
        q05, q95 = ema_q05(q05), ema_q95(q95)
        low, high = 0.5 * q05, 2.0 * q95
        return (high - low) * self.uniform_grid + low

    def forward(self, z1: Tensor, z2: Tensor, reduction: str = "mean") -> Tensor:  # noqa: D102
        l1 = torch.cdist(z1, z2, p=1)
        l2_sq = torch.cdist(z1, z2, p=2).square()

        bw_l1 = self._bandwidths(l1.detach(), self.ema_l1_q05, self.ema_l1_q95)
        bw_l2 = self._bandwidths(l2_sq.detach(), self.ema_l2_q05, self.ema_l2_q95)

        k_laplace = torch.exp(-l1[..., None] / bw_l1)
        k_gaussian = torch.exp(-l2_sq[..., None] / bw_l2)
        kernels = torch.cat([k_laplace, k_gaussian], dim=-1)
        return reduce(kernels, dim=-1, mode=reduction)
