# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from blueprint.losses import MMDFuse
from blueprint.models.kernels import ExponentialKernel


def test_mmd_fuse_single_kernel():
    """MMD-Fuse with a single kernel is finite and positive between shifted samples."""
    torch.manual_seed(0)
    x = torch.randn(1, 64, 2)
    y = torch.randn(1, 64, 2) + 1.0
    loss = MMDFuse(ExponentialKernel(p=2, n_kernels=1))(x, y)
    assert torch.isfinite(loss)
    assert loss > 0
