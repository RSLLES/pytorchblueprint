# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import override

from torch import Tensor, nn


class ResnetBlock(nn.Module):
    """Two-linear residual block with optional norm and configurable activation."""

    def __init__(
        self,
        dim: int,
        norm_layer: type[nn.Module] = nn.Identity,
        act_layer: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            norm_layer(dim),
            act_layer(),
            nn.Linear(dim, dim),
            norm_layer(dim),
        )
        self.act = act_layer()

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.block(x))


class ResnetBlocks(nn.Module):
    """Stack of MLP residual blocks at fixed width."""

    def __init__(
        self,
        dim: int,
        depth: int = 2,
        norm_layer: type[nn.Module] = nn.Identity,
        act_layer: type[nn.Module] = nn.SiLU,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *(
                ResnetBlock(dim, norm_layer=norm_layer, act_layer=act_layer)
                for _ in range(depth)
            )
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)
