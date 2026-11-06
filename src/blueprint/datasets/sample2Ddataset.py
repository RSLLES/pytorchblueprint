# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Generator, Tensor
from torch.utils.data import Dataset, default_collate

import blueprint


class Sample2DDataset(Dataset):
    """Samples 2D points from various distributions."""

    def __init__(self, length: int, seed: int, dist: str):
        super().__init__()
        self.seed = seed
        self.length = length
        dist = blueprint.utils.format.format_string(dist)
        if dist not in _SAMPLERS:
            raise ValueError(f"Unsupported distribution: {dist}")
        self.sample = _SAMPLERS[dist]
        self.collate_fn = default_collate

    def increment_seed(self):
        """Advance internal seed."""
        self.seed = blueprint.utils.random.derive_new_seed(self.seed)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tensor:
        if idx < 0 or idx >= self.__len__():
            raise IndexError(f"Index {idx} out of range for length {self.length}.")
        seed = blueprint.utils.random.derive_new_seed(self.seed + idx + 1)
        gen = blueprint.utils.random.get_generator(seed)
        return self.sample(n=1, gen=gen).squeeze(0)


def sample_gaussian(n: int, gen: Generator) -> Tensor:
    """Sample n points from a standard 2D Gaussian distribution."""
    return torch.randn((n, 2), generator=gen)


def sample_spiral(
    n: int, gen: Generator, fullangle: float = 6 * torch.pi, noise: float = 0.1
) -> Tensor:
    """Sample n points from a noisy spiral distribution."""
    theta = fullangle * torch.rand(n, generator=gen)
    rot = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    r = theta / (2 * torch.pi)
    eps = sample_gaussian(n, gen=gen)
    x = r[:, None] * rot + noise * eps
    return x


def sample_cardioid(n: int, gen: Generator, noise: float = 0.2) -> Tensor:
    """Sample n points from a cardioid distribution."""
    diameter = 2.0
    theta = 2 * torch.pi * torch.rand(n, generator=gen)
    rot = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
    r = diameter * (1 - rot[:, 0])
    eps = sample_gaussian(n, gen=gen)
    x = r[:, None] * rot + noise * eps
    x[:, 0] += diameter
    return x


_SAMPLERS = {
    "gaussian": sample_gaussian,
    "spiral": sample_spiral,
    "cardioid": sample_cardioid,
}
