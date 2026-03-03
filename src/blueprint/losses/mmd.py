# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Implementation of Maximum Mean Discrepancy (MMD).

See [1] for details about MMD, and [2] for Fuse-MMD.
Code based on https://github.com/yiftachbeer/mmd_loss_pytorch/tree/master

[1] Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
A kernel two-sample test.
The journal of machine learning research, 13(1), 723-773.
https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

[2] Biggs, Felix and Schrab, Antonin and Gretton, Arthur (2023).
MMD-Fuse: Learning and Combining Kernels for Two-Sample Testing Without Data Splitting.
Advances in Neural Information Processing Systems 36 (NeurIPS 2023)
https://proceedings.neurips.cc/paper_files/paper/2023/file/edd00cead3425393baf13004de993017-Paper-Conference.pdf

[3] Schrab, Antonin and Kim, Ilmun and Albert, Melisande and Laurent,
Beatrice and Guedj, Benjamin and Gretton, Arthur (2023).
MMD Aggregated Two-Sample Test. Journal of Machine Learning Research
https://www.jmlr.org/papers/volume24/21-1289/21-1289.pdf
"""

import torch
import torch.distributed as dist
from torch import Tensor, nn

from blueprint.utils.torch import reduce


class ExpMovingAverage(nn.Module):
    """Exponential moving average with warmup for a 1D Tensor."""

    mean: Tensor
    counter: Tensor
    is_initialized: Tensor

    def __init__(self, lambd: float, n_values: int = 1) -> None:
        super().__init__()
        self.lambd = lambd
        self.register_buffer("mean", torch.zeros((n_values,)))
        self.register_buffer("counter", torch.tensor(0, dtype=torch.long))
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
        self.counter.fill_(1)
        self.is_initialized.fill_(True)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Update the current ema and return the new value."""
        if not self.is_initialized:
            self._initialize_buffer(x)
            return self.mean
        x_synced = self._sync_values(x)
        self.counter.add_(1)
        lambd = max(self.lambd, 1.0 / self.counter.item())
        self.mean.mul_(1.0 - lambd).add_(x_synced, alpha=lambd)
        return self.mean


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


def mmd(x: Tensor, y: Tensor, kernel: nn.Module, reduction: str) -> Tensor:
    """Return the (biased) MMD between x and y given a kernel."""
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expected 3D tensors [B, N, D], got {x.shape},{y.shape}.")
    if x.size(0) != y.size(0) or x.size(-1) != y.size(-1):
        raise ValueError(f"Batch or feature dim size mismatch: {x.shape} vs {y.shape}.")
    if x.size(1) < 2 or y.size(1) < 2:
        raise ValueError(
            f"MMD needs at least 2 samples per batch, got x:{x.size(1)}, y:{y.size(1)}."
        )

    N = x.size(1)
    xy = torch.cat([x, y], dim=1)
    K = kernel(xy, xy, reduction="mean")

    K_xx = K[:, :N, :N].mean(dim=(1, 2))
    K_yy = K[:, N:, N:].mean(dim=(1, 2))
    K_xy = K[:, :N, N:].mean(dim=(1, 2))
    mmd_sq = K_xx + K_yy - 2.0 * K_xy
    return reduce(mmd_sq, dim=0, mode=reduction)


def mmd_max(x: Tensor, y: Tensor, kernel: nn.Module, reduction: str) -> Tensor:
    """Return the maximal (biased) MMD between x and y given multiple kernels."""
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expected 3D tensors [B, N, D], got {x.shape},{y.shape}.")
    if x.size(0) != y.size(0) or x.size(-1) != y.size(-1):
        raise ValueError(f"Batch or feature dim size mismatch: {x.shape} vs {y.shape}.")
    if x.size(1) < 2 or y.size(1) < 2:
        raise ValueError(
            f"MMD needs at least 2 samples per batch, got x:{x.size(1)}, y:{y.size(1)}."
        )

    N = x.size(1)
    xy = torch.cat([x, y], dim=1)
    K = kernel(xy, xy, reduction="none")

    K_xx = K[:, :N, :N].mean(dim=(1, 2))
    K_yy = K[:, N:, N:].mean(dim=(1, 2))
    K_xy = K[:, :N, N:].mean(dim=(1, 2))
    mmd_sq = K_xx + K_yy - 2.0 * K_xy
    mmd_sq = mmd_sq.amax(dim=-1)
    return reduce(mmd_sq, dim=0, mode=reduction)


def mmd_fuse(x: Tensor, y: Tensor, kernel: nn.Module, reduction: str) -> Tensor:
    """Implement MMD-Fuse, see [2]."""
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expected 3D tensors [B, N, D], got {x.shape},{y.shape}.")
    if x.size(0) != y.size(0) or x.size(-1) != y.size(-1):
        raise ValueError(f"Batch or feature dim size mismatch: {x.shape} vs {y.shape}.")
    if x.size(1) < 2 or y.size(1) < 2:
        raise ValueError(
            f"MMD needs at least 2 samples per batch, got x:{x.size(1)}, y:{y.size(1)}."
        )

    N = x.size(1)
    xy = torch.cat([x, y], dim=1)
    K = kernel(xy, xy, reduction="none")
    n_kernels = torch.tensor(K.size(-1), device=K.device, dtype=K.dtype)
    lambd = torch.sqrt(n_kernels * (n_kernels - 1.0))

    K_xx = K[:, :N, :N].mean(dim=(1, 2))
    K_yy = K[:, N:, N:].mean(dim=(1, 2))
    K_xy = K[:, :N, N:].mean(dim=(1, 2))
    mmd_sq = K_xx + K_yy - 2.0 * K_xy

    std = K.square().mean(dim=(1, 2)).sqrt()
    mmd_norm = mmd_sq / (std + 1e-9)
    mmd_fused = (torch.logsumexp(lambd * mmd_norm, dim=-1) - n_kernels.log()) / lambd
    return reduce(mmd_fused, dim=0, mode=reduction)


def linear_mmd(x: Tensor, y: Tensor, kernel: nn.Module, reduction: str) -> Tensor:
    """Return the mean MMD between x and y given multiple kernels."""
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expected 3D tensors [B, N, D], got {x.shape},{y.shape}.")
    if x.size(0) != y.size(0) or x.size(-1) != y.size(-1):
        raise ValueError(f"Batch or feature dim size mismatch: {x.shape} vs {y.shape}.")
    if x.size(1) < 2 or y.size(1) < 2:
        raise ValueError(
            f"MMD needs at least 2 samples per batch, got x:{x.size(1)}, y:{y.size(1)}."
        )

    N = min(x.size(1), y.size(1))
    N = N - (N % 2)
    x, y = x[:, :N], y[:, :N]

    x = x.reshape(x.size(0), N // 2, 2, -1)
    y = y.reshape(y.size(0), N // 2, 2, -1)
    z1 = torch.stack([x[:, :, 0], y[:, :, 0]], dim=2)
    z2 = torch.stack([x[:, :, 1], y[:, :, 1]], dim=2)

    K = kernel(z1, z2, reduction="none")
    mmd_linear_sq = K[:, :, 0, 0] + K[:, :, 1, 1] - K[:, :, 0, 1] - K[:, :, 1, 0]
    mmd_linear_sq = mmd_linear_sq.mean(dim=1)  # N//2 dimension
    mmd_linear_sq = mmd_linear_sq.amax(dim=-1)  # kernels dimension
    return reduce(mmd_linear_sq, dim=0, mode=reduction)


class MMD(nn.Module):
    """Batchified MMD."""

    def __init__(self, kernel: nn.Module, reduction: str = "mean"):
        super().__init__()
        self.kernel = kernel
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: D102
        return mmd(x=x, y=y, kernel=self.kernel, reduction=self.reduction)


class MMDMax(nn.Module):
    """Batchified MMD."""

    def __init__(self, kernel: nn.Module, reduction: str = "mean"):
        super().__init__()
        self.kernel = kernel
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: D102
        return mmd_max(x=x, y=y, kernel=self.kernel, reduction=self.reduction)


class MMDFuse(nn.Module):
    """Batchified Fuse-MMD."""

    def __init__(self, kernel: nn.Module, reduction: str = "mean"):
        super().__init__()
        self.kernel = kernel
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: D102
        return mmd_fuse(x=x, y=y, kernel=self.kernel, reduction=self.reduction)


class MMDLinear(nn.Module):
    """Linear MMD."""

    def __init__(self, kernel: nn.Module, reduction: str = "mean"):
        super().__init__()
        self.kernel = kernel
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # noqa: D102
        return linear_mmd(x=x, y=y, kernel=self.kernel, reduction=self.reduction)
