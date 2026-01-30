# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn


class FourierEmbeddings(nn.Module):
    """Project data to sin and cosine embeddings with normaly sampled weights."""

    angular_freqs: Tensor

    def __init__(self, input_dim: int, embed_dim: int, scale: float = 1.0):
        super().__init__()
        assert embed_dim % 2 == 0
        angular_freqs = scale * 2 * torch.pi * torch.randn(input_dim, embed_dim // 2)
        self.register_buffer("angular_freqs", angular_freqs)

    def forward(self, t: Tensor) -> Tensor:  # noqa: D102
        angles = t @ self.angular_freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
