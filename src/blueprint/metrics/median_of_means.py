# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Median of means estimator as a torchmetric."""

import math
import warnings
from typing import Any, override

import torch
from torch import Tensor
from torchmetrics import Metric

from blueprint.utils.random import get_generator


class MedianOfMeans(Metric):
    """Median of means estimator over a stream of scalar values."""

    full_state_update = False
    sums: Tensor
    counts: Tensor

    def __init__(self, n_groups: int, seed: int = 0, **kwargs: Any):
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

    @override
    def update(self, value: float | Tensor):
        if self._gen is None:
            self._gen = get_generator(self.seed, device=self.device)
        # updated every training step, so it must not wait for the GPU: a blocking
        # copy or torch.bincount (which reads groups.max()) would
        value = torch.as_tensor(value).to(self.device, non_blocking=True).flatten()
        groups = torch.randint(
            0, self.n_groups, (value.numel(),), device=self.device, generator=self._gen
        )
        self.sums.index_add_(0, groups, value.to(self.sums.dtype))
        self.counts.index_add_(0, groups, torch.ones_like(groups))

    @override
    def compute(self) -> Tensor:
        mask = self.counts > 0
        if not mask.any():
            return torch.tensor(float("nan"), device=self.device)
        return (self.sums[mask] / self.counts[mask]).median()
