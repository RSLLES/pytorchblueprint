# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Median of means estimator as a torchmetric."""

import math
import warnings

import torch
from torch import Tensor
from torchmetrics import Metric

from blueprint.utils.random import get_generator


class MedianOfMeans(Metric):
    """Median of means estimator over a stream of scalar values."""

    full_state_update: bool = False
    sums: Tensor
    counts: Tensor

    def __init__(self, n_groups: int, seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        if n_groups < 1:
            raise ValueError(f"n_groups must be >= 1, got {n_groups}")
        if n_groups > 1000:
            warnings.warn(
                f"n_groups={n_groups} is large; typical usage is O(log n).",
                stacklevel=2,
            )

        self._gen = None
        self.n_groups = n_groups
        self.seed = seed
        sums_init = torch.zeros(n_groups, dtype=torch.float32)
        self.add_state("sums", default=sums_init, dist_reduce_fx="sum")
        counts_init = torch.zeros(n_groups, dtype=torch.long)
        self.add_state("counts", default=counts_init, dist_reduce_fx="sum")

    @staticmethod
    def recommended_n_groups(n_elements: int) -> int:
        """Group count such that failure probability is 1/n_elements."""
        return max(1, math.ceil(8.0 * math.log(n_elements)))

    def update(self, value: float | Tensor) -> None:  # noqa: D102
        if self._gen is None:
            self._gen = get_generator(self.seed, device=self.device)
        value = torch.as_tensor(value, device=self.device).flatten()
        groups = torch.randint(
            0, self.n_groups, (value.numel(),), device=self.device, generator=self._gen
        )
        self.sums += torch.bincount(groups, weights=value, minlength=self.n_groups)
        self.counts += torch.bincount(groups, minlength=self.n_groups)

    def compute(self) -> Tensor:  # noqa: D102
        mask = self.counts > 0
        if not mask.any():
            return torch.tensor(float("nan"), device=self.device)
        return (self.sums[mask] / self.counts[mask]).median()
