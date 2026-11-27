# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import override

from torch import Tensor, nn
from torch.func import jacrev, vmap

from blueprint.utils.torch import reduce


class HyvarinenLoss(nn.Module):
    """Hyvarinen Score Matching loss, L = (1/2) * ||f(x)||^2 + Tr(nabla_x(f(x)))."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    @override
    def forward(self, score_net: nn.Module, x: Tensor) -> Tensor:
        return hyvarinen_loss(score_net=score_net, x=x, reduction=self.reduction)


def hyvarinen_loss(score_net: nn.Module, x: Tensor, reduction: str = "mean") -> Tensor:
    """Compute the Hyvarinen Score Matching loss.

    L = (1/2) * ||f(x)||^2 + Tr(nabla_x(f(x))).
    """
    batch_shape = x.shape[:-1]
    input_dim = x.shape[-1]
    x = x.view(-1, input_dim)

    def single_score_fn(x_i: Tensor) -> tuple[Tensor, Tensor]:
        score = score_net(x_i.unsqueeze(0)).squeeze(0)
        return score, score

    jacobians, score = vmap(jacrev(single_score_fn, has_aux=True))(x)
    norm_sq = 0.5 * (score**2).sum(dim=1)
    trace = jacobians.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    loss = norm_sq + trace
    loss = loss.view(batch_shape)
    return reduce(loss, mode=reduction, dim=0)
