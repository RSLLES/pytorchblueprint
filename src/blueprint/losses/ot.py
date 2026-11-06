# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor


def _check_shapes(a: Tensor, b: Tensor, M: Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2 or M.ndim != 3:
        raise ValueError(
            "Expected marginals a [B, N], b [B, K] and costs M [B, N, K], "
            f"got {a.shape}, {b.shape}, {M.shape}."
        )


def sinkhorn_knopp(
    a: Tensor, b: Tensor, M: Tensor, reg: float, n_iters: int = 20, eps: float = 1e-12
) -> Tensor:
    """Return the entropic transport plan between a and b, by Sinkhorn iterations."""
    _check_shapes(a, b, M)

    u = torch.ones_like(a)
    v = torch.ones_like(b)
    M = torch.exp(-M / reg)

    for _ in range(n_iters):
        v = b / (torch.einsum("bij, bi -> bj", M, u) + eps)
        u = a / (torch.einsum("bij, bj -> bi", M, v) + eps)

    P = u[:, :, None] * M * v[:, None, :]
    return P


def sinkhorn_knopp_log(
    a: Tensor,
    b: Tensor,
    M: Tensor,
    reg: float | Tensor,
    n_iters: int = 20,
    eps: float = 1e-12,
) -> Tensor:
    """Return the entropic transport plan, by Sinkhorn iterations in log space."""
    _check_shapes(a, b, M)

    log_a = torch.log(a.clip(min=eps))
    log_b = torch.log(b.clip(min=eps))
    u = torch.zeros_like(a)
    v = torch.zeros_like(b)
    log_M = M / (-reg)

    for _ in range(n_iters):
        v = log_b - torch.logsumexp(log_M + u[:, :, None], dim=-2)
        u = log_a - torch.logsumexp(log_M + v[:, None, :], dim=-1)

    log_P = u[:, :, None] + log_M + v[:, None, :]
    return torch.exp(log_P)
