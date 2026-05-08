# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor, nn

from .basics import FourierEmbeddings, ResnetBlocks


class MomentNet(nn.Module):
    """Toy model that learns a input_dim-D  function."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        output_dim: int | None = None,
        embed_scale: float = 1.0,
    ):
        super().__init__()
        self.output_dim = input_dim if output_dim is None else output_dim
        self.net = nn.Sequential(
            FourierEmbeddings(input_dim, inner_dim, scale=embed_scale),
            nn.Linear(inner_dim, inner_dim),
            nn.SiLU(),
            ResnetBlocks(inner_dim, depth=2),
            nn.Linear(inner_dim, self.output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        output_shape = x.shape[:-1] + (self.output_dim,)
        x_flat = x.reshape(-1, x.size(-1))
        outputs_flat = self.net(x_flat)
        outputs = outputs_flat.view(*output_shape)
        return outputs
