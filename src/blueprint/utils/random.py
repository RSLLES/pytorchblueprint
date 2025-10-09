# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for reproducible random number generation."""

import torch
from torch import Generator, Tensor

_MAX_SEED = torch.iinfo(torch.int64).max


@torch.compiler.disable()
def get_generator(seed: int | Tensor, device=None) -> Generator:
    """Return a torch.Generator seeded with `seed` on `device`."""
    if isinstance(seed, Tensor):
        device = seed.device if device is None else device
        seed = seed.item()
    gen = torch.Generator(device).manual_seed(seed)
    return gen


@torch.compiler.disable()
def derive_new_seed(seed: int | Tensor) -> int | Tensor:
    """Derive a new seed from `seed` using a dedicated generator."""
    gen = get_generator(seed)
    new_seed = torch.randint(
        0, _MAX_SEED, size=(), device=gen.device, dtype=torch.int64, generator=gen
    )
    new_seed = new_seed.item() if isinstance(seed, int) else new_seed
    return new_seed
