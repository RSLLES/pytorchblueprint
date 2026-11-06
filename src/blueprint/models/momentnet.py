# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch import Tensor, nn

from .basics import FourierEmbeddings, ResnetBlocks


class MomentNet(nn.Module):
    """Toy model that maps input_dim-D points to output_dim-D points."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        output_dim: int | None = None,
        embed_scale: float = 1.0,
    ):
        super().__init__()
        self.output_dim = input_dim if output_dim is None else output_dim
        self.embed = FourierEmbeddings(input_dim, inner_dim, scale=embed_scale)
        self.body = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.SiLU(),
            ResnetBlocks(inner_dim, depth=2),
        )
        self.head = nn.Linear(inner_dim, self.output_dim)

    def hidden_matrices(self) -> list[Tensor]:
        """Return the weight matrices between the input and output layers."""
        return [m.weight for m in self.body.modules() if isinstance(m, nn.Linear)]

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        output_shape = x.shape[:-1] + (self.output_dim,)
        x_flat = x.reshape(-1, x.size(-1))
        outputs_flat = self.head(self.body(self.embed(x_flat)))
        return outputs_flat.view(*output_shape)
