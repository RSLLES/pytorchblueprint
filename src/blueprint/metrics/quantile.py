# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torchmetrics import Metric


class ReservoirOnlineQuantile(Metric):
    """Approximate per-channel quantile via reservoir sampling."""

    full_state_update = False
    reservoir: Tensor

    def __init__(self, channel_dim: int, q: float = 0.05, reservoir_size: int = 512):
        super().__init__()
        self.q = q
        self.reservoir_size = reservoir_size

        self.add_state(
            "reservoir",
            default=torch.empty(0, channel_dim),
            dist_reduce_fx="cat",
        )
        self.add_state(
            "count",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, x: Tensor) -> None:  # noqa: D102
        x = x.reshape(-1, x.size(-1))

        for sample in x:
            self.count += 1
            k = int(self.count)

            if self.reservoir.size(0) < self.reservoir_size:
                self.reservoir = torch.cat([self.reservoir, sample.unsqueeze(0)], dim=0)
            else:
                j = torch.randint(0, k, (1,), device=x.device)
                if j < self.reservoir_size:
                    self.reservoir[j] = sample

    def compute(self) -> Tensor:  # noqa: D102
        # after DDP sync, reservoir = concatenation of all local reservoirs
        if self.reservoir.size(0) > self.reservoir_size:
            idx = torch.randperm(self.reservoir.size(0), device=self.reservoir.device)
            subset_idx = idx[: self.reservoir_size]
            self.reservoir = self.reservoir[subset_idx]

        return torch.quantile(self.reservoir, self.q, dim=0)
