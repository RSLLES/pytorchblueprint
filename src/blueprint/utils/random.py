# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for reproducible random number generation."""

import torch
from torch import Generator, Tensor

_MAX_SEED = torch.iinfo(torch.int64).max
_RANK_STRIDE = 0x9E3779B97F4A7C15 % _MAX_SEED


@torch.compiler.disable()
def get_generator(
    seed: int | Tensor, device: torch.device | str | None = None
) -> Generator:
    """Return a torch.Generator seeded with `seed` on `device`."""
    if isinstance(seed, Tensor):
        device = seed.device if device is None else device
        seed = int(seed.item())
    gen = torch.Generator(device).manual_seed(seed)
    return gen


@torch.compiler.disable()
def get_rank_generator(
    seed: int | Tensor, rank: int, device: torch.device | str | None = None
) -> Generator:
    """Return a generator whose stream differs across data-parallel ranks."""
    if isinstance(seed, Tensor):
        device = seed.device if device is None else device
        seed = int(seed.item())
    return get_generator((seed + rank * _RANK_STRIDE) % _MAX_SEED, device)


@torch.compiler.disable()
def derive_new_seed(seed: int) -> int:
    """Derive a new seed from `seed` using a dedicated generator."""
    new_seed = torch.tensor(seed)
    derive_new_seed_(new_seed)
    return int(new_seed)


@torch.compiler.disable()
def derive_new_seed_(seed: Tensor):
    """In-place version of derive_new_seed."""
    gen = get_generator(seed)
    seed.copy_(
        torch.randint(
            0, _MAX_SEED, size=(), device=gen.device, dtype=torch.int64, generator=gen
        )
    )
