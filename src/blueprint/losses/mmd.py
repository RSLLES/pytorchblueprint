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
from torch import Tensor, nn

from blueprint.utils.torch import reduce


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
