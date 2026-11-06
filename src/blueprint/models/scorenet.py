# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import torch
from torch import Tensor, nn

from .basics import FourierEmbeddings, ResnetBlocks


def langevin_step(
    score_fn: Callable[[Tensor], Tensor], x: Tensor, step_size: float
) -> Tensor:
    """Perform one step of Unadjusted Langevin Dynamics."""
    score = score_fn(x)
    noise = torch.randn_like(x)
    return x + (step_size / 2) * score + (step_size**0.5) * noise


def langevin_dynamics(
    score_fn: Callable[[Tensor], Tensor], x0: Tensor, n_steps: int, step_size: float
) -> Tensor:
    """Integrate using Langevin Dynamics starting from x0."""
    x = x0
    for _ in range(n_steps):
        x = langevin_step(score_fn=score_fn, x=x, step_size=step_size)
    return x


class ScoreNet(nn.Module):
    """Toy model that learns an input_dim-D score function."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        n_steps: int,
        step_size: float,
        embed_scale: float = 1.0,
    ):
        super().__init__()
        self.embed = FourierEmbeddings(input_dim, inner_dim, scale=embed_scale)
        self.body = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.SiLU(),
            ResnetBlocks(inner_dim, depth=2),
        )
        self.head = nn.Linear(inner_dim, input_dim)
        self.n_steps = n_steps
        self.step_size = step_size

    def forward_single_pass(self, x: Tensor) -> Tensor:
        """Return the predicted score at x."""
        x_shape = x.shape
        x_flat = x.reshape(-1, x.size(-1))
        outputs_flat = self.head(self.body(self.embed(x_flat)))
        return outputs_flat.view(*x_shape)

    def forward(self, x: Tensor) -> Tensor:
        """Perform either a single step when training, or integrate during inference."""
        if self.training:
            return self.forward_single_pass(x)
        return langevin_dynamics(
            score_fn=self.forward_single_pass,
            x0=x,
            n_steps=self.n_steps,
            step_size=self.step_size,
        )
