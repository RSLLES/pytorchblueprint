# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convinient math functions."""

import torch
from torch import Tensor


def logsumexp(
    x: Tensor, dim: int, weights: Tensor, epsilon: float, keepdim: bool = False
) -> Tensor:
    """Return the weighted log-sum-exp of x along dim."""
    if torch.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    weights = weights.expand_as(x)
    log_w = torch.log(weights + epsilon)
    x = x + log_w
    return torch.logsumexp(x, dim=dim, keepdim=keepdim)


def interpolate_tensor(t: Tensor, h: int, w: int, mode="bilinear"):
    """Resize t to shape (B, C, h, w) using the specified interpolation mode."""
    if t.shape[-1] == w and t.shape[-2] == h:
        return t
    return torch.nn.functional.interpolate(t, mode=mode, antialias=False, size=(h, w))
