# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn


class GaussianFourierProjection(nn.Module):
    """Project data to sin and cosine embeddings with normaly sampled weights."""

    def __init__(self, input_dim, embed_dim, scale=1.0):
        super().__init__()
        self.register_buffer("B", torch.randn(input_dim, embed_dim // 2) * scale)

    def forward(self, x):  # noqa: D102
        x_proj = (2 * torch.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResnetBlock(nn.Module):
    """ResNet block with SiLU activation."""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.act = nn.SiLU()

    def forward(self, x):  # noqa: D102
        return self.act(x + self.block(x))


def langevin_step(score_fn: callable, x: Tensor, step_size: float) -> Tensor:
    """Perform one step of Unadjusted Langevin Dynamics."""
    score = score_fn(x)
    noise = torch.randn_like(x)
    return x + (step_size / 2) * score + (step_size**0.5) * noise


def langevin_dynamics(
    score_fn: callable, x0: Tensor, n_steps: int, step_size: float
) -> Tensor:
    """Integrate using Langevin Dynamics starting from x0."""
    x = x0
    for _ in range(n_steps):
        x = langevin_step(score_fn=score_fn, x=x, step_size=step_size)
    return x


class ScoreNet(nn.Module):
    """Toy model that learns a input_dim-D score function."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        n_steps: int,
        step_size: float,
        embed_scale: float = 1.0,
    ):
        super().__init__()
        self.embed = GaussianFourierProjection(input_dim, inner_dim, scale=embed_scale)
        self.net = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.SiLU(),
            ResnetBlock(inner_dim),
            ResnetBlock(inner_dim),
            nn.Linear(inner_dim, input_dim),
        )
        self.n_steps = n_steps
        self.step_size = step_size

    def forward_single_pass(self, x: Tensor) -> Tensor:
        """Perform a single step."""
        x_shape = x.shape
        x_flat = x.reshape(-1, x.size(-1))
        outputs_flat = self.net(self.embed(x_flat))
        outputs = outputs_flat.view(*x_shape)
        return outputs

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
