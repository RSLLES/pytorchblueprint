# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor, nn

from .basics import FourierEmbeddings


class ResnetBlock(nn.Module):
    """ResNet block with SiLU activation."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.act = nn.SiLU()

    def forward(self, x):  # noqa: D102
        return self.act(x + self.block(x))


class MomentNet(nn.Module):
    """Toy model that learns a input_dim-D  function."""

    def __init__(self, input_dim: int, inner_dim: int, embed_scale: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            FourierEmbeddings(input_dim, inner_dim),
            nn.Linear(inner_dim, inner_dim),
            nn.SiLU(),
            ResnetBlock(inner_dim),
            ResnetBlock(inner_dim),
            nn.Linear(inner_dim, input_dim),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x_shape = x.shape
        x_flat = x.reshape(-1, x.size(-1))
        outputs_flat = self.net(x_flat)
        outputs = outputs_flat.view(*x_shape)
        return outputs
