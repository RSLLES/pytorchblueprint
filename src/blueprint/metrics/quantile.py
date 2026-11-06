# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil
from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric


def _gather_and_concatenate(x: Tensor, group: Any = None) -> list[Tensor]:
    """Gather a state whose length differs per rank, as one already-merged tensor."""
    world_size = torch.distributed.get_world_size(group)
    x = x.contiguous()
    size = torch.tensor([x.size(0)], device=x.device)
    sizes = [torch.empty_like(size) for _ in range(world_size)]
    torch.distributed.all_gather(sizes, size, group=group)
    padded = x.new_zeros(int(torch.cat(sizes).max()), *x.shape[1:])
    padded[: x.size(0)] = x
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, padded, group=group)
    return [torch.cat([g[: int(n)] for g, n in zip(gathered, sizes)])]


class KLLOnlineQuantile(Metric):
    """Approximate per-channel quantile via the KLL sketch (Karnin-Lang-Liberty)."""

    full_state_update = False
    values: Tensor
    levels: Tensor

    def __init__(
        self,
        channel_dim: int = 1,
        q: float | tuple[float, ...] | list[float] = 0.5,
        k: int = 256,
        c: float = 2 / 3,
    ):
        super().__init__(dist_sync_fn=_gather_and_concatenate)
        self.q = q
        self.k = k
        self.c = c
        self.add_state(
            "values", default=torch.empty(0, channel_dim), dist_reduce_fx="cat"
        )
        self.add_state(
            "levels", default=torch.empty(0, dtype=torch.long), dist_reduce_fx="cat"
        )

    def update(self, x: Tensor) -> None:  # noqa: D102
        x = x.reshape(-1, 1) if x.ndim == 1 else x.reshape(-1, x.size(-1))
        self.values = torch.cat([self.values, x])
        self.levels = torch.cat([self.levels, x.new_zeros(x.size(0), dtype=torch.long)])
        while self.values.size(0) >= self._max_size():
            self._compact(self._lowest_full_level())

    def compute(self) -> Tensor:
        """Quantile per channel: (C,) for a scalar q, (len(q), C) for a tuple."""
        values = self.values.reshape(-1, self.values.size(-1))
        weights = torch.pow(2.0, self.levels.reshape(-1)).to(values)
        order = values.argsort(dim=0)
        values = values.gather(0, order)
        cumw = weights[:, None].expand_as(values).gather(0, order).cumsum(0)
        levels = torch.as_tensor(self.q, device=values.device)
        idx = (cumw < levels.reshape(-1, 1, 1) * cumw[-1]).sum(1)
        idx = idx.clamp(max=values.size(0) - 1)
        rows = values.expand(idx.size(0), -1, -1)
        quantiles = rows.gather(1, idx[:, None]).squeeze(1)
        return quantiles if levels.ndim else quantiles.squeeze(0)

    def _n_levels(self) -> int:
        return int(self.levels.max()) + 1 if self.levels.numel() else 1

    def _capacity(self, h: int) -> int:
        return max(2, ceil(self.k * self.c ** (self._n_levels() - 1 - h)))

    def _max_size(self) -> int:
        return sum(self._capacity(h) for h in range(self._n_levels()))

    def _lowest_full_level(self) -> int:
        counts = self.levels.bincount()
        return next(h for h in range(len(counts)) if counts[h] >= self._capacity(h))

    def _compact(self, h: int) -> None:
        at_h = self.levels == h
        block = self.values[at_h].sort(dim=0).values
        m = block.size(0) - (block.size(0) & 1)
        offset = torch.randint(2, ()).item()
        survivors, leftover = block[offset:m:2], block[m:]
        self.values = torch.cat([self.values[~at_h], survivors, leftover])
        self.levels = torch.cat(
            [
                self.levels[~at_h],
                self.levels.new_full((survivors.size(0),), h + 1),
                self.levels.new_full((leftover.size(0),), h),
            ]
        )
