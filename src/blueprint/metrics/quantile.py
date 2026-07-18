# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torchmetrics import Metric


class ReservoirOnlineQuantile(Metric):
    """Approximate per-channel quantile via key-based reservoir sampling.

    Each item is tagged with a uniform random key and the reservoir keeps the
    items with the largest keys. This yields a uniform sample of the stream
    (Efraimidis & Spirakis) that is order-independent and merges correctly
    across processes, since the global top-k by key is a uniform sample of the
    union of the local reservoirs.
    """

    full_state_update = False
    reservoir: Tensor
    keys: Tensor

    def __init__(self, channel_dim: int = 1, q: float = 0.5, reservoir_size: int = 512):
        super().__init__()
        self.q = q
        self.reservoir_size = reservoir_size
        self.add_state(
            "reservoir", default=torch.empty(0, channel_dim), dist_reduce_fx="cat"
        )
        self.add_state("keys", default=torch.empty(0), dist_reduce_fx="cat")

    def update(self, x: Tensor) -> None:  # noqa: D102
        x = x.reshape(-1, 1) if x.ndim == 1 else x.reshape(-1, x.size(-1))
        keys = torch.rand(x.size(0), device=x.device)
        self.reservoir, self.keys = self._keep_largest(
            torch.cat([self.reservoir, x]), torch.cat([self.keys, keys])
        )

    def compute(self) -> Tensor:  # noqa: D102
        reservoir, _ = self._keep_largest(self.reservoir, self.keys)
        return torch.quantile(reservoir, self.q, dim=0)

    def _keep_largest(self, values: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
        if keys.size(0) <= self.reservoir_size:
            return values, keys
        top = keys.topk(self.reservoir_size).indices
        return values[top], keys[top]
