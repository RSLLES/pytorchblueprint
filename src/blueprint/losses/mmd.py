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

from blueprint.utils.torch import reduce


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
        dists = torch.cdist(z1, z2, p=2.0).square()
        sigma2 = self.get_bandwidth(dists) * self.bandwidth_multipliers
        return torch.exp(-dists[..., None] / sigma2).sum(dim=-1)


def mmd(x: Tensor, y: Tensor, kernel: nn.Module, reduction: str) -> Tensor:
    """Batchified (biased) MMD loss function."""
    if x.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expected 3D tensors [B, N, D], got {x.shape},{y.shape}.")
    if x.size(0) != y.size(0) or x.size(-1) != y.size(-1):
        raise ValueError(f"Batch or feature dim size mismatch: {x.shape} vs {y.shape}.")
    if x.size(1) < 2 or y.size(1) < 2:
        raise ValueError(
            f"MMD needs at least 2 samples per batch, got x:{x.size(1)}, y:{y.size(1)}."
        )

    K_xx = kernel(x, x).mean(dim=(1, 2))
    K_yy = kernel(y, y).mean(dim=(1, 2))
    K_xy = kernel(x, y).mean(dim=(1, 2))
    mmd_sq = K_xx + K_yy - 2.0 * K_xy
    return reduce(mmd_sq, dim=0, mode=reduction)


def block_mmd(
    x: Tensor, y: Tensor, kernel: nn.Module, block_size: int | None, reduction: str
) -> Tensor:
    """Block-based MMD implementation for scalable computation."""
    if x.dim() == 2:
        x = x.unsqueeze(0)  # [1, Total, D]
    if y.dim() == 2:
        y = y.unsqueeze(0)  # [1, Total, D]

    B, N, D = x.shape

    if block_size is None or block_size <= 0:
        return mmd(x, y, kernel=kernel, reduction=reduction)
    if block_size == 1:
        raise ValueError(f"MMD requires at least block_size >=2, got {block_size}.")

    x_flat = x.reshape(-1, D)
    y_flat = y.reshape(-1, D)
    n_samples = (x_flat.size(0) // block_size) * block_size
    if n_samples == 0:
        n_samples = x_flat.size(0)

    x_flat = x_flat[:n_samples]
    y_flat = y_flat[:n_samples]
    x_pack = x_flat.view(-1, block_size, D)
    y_pack = y_flat.view(-1, block_size, D)
    return mmd(x_pack, y_pack, kernel=kernel, reduction=reduction)


class MMD(nn.Module):
    """Batchified MMD loss function."""

    def __init__(self, kernel: nn.Module, reduction: str = "mean"):
        super().__init__()
        self.kernel = kernel
        self.reduction = reduction

    def forward(self, x: Tensor, y: Tensor):  # noqa: D102
        return mmd(x=x, y=y, kernel=self.kernel, reduction=self.reduction)


class BlockdMMD(nn.Module):
    """Block-based MMD implementation for scalable computation."""

    def __init__(
        self, kernel: nn.Module, block_size: int | None, reduction: str = "mean"
    ):
        super().__init__()
        self.block_size = block_size
        self.kernel = kernel
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor):  # noqa: D102
        return block_mmd(
            x=x,
            y=y,
            kernel=self.kernel,
            block_size=self.block_size,
            reduction=self.reduction,
        )
