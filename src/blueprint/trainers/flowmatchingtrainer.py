# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Flowmatching training step."""

import torch
from torch import Tensor, nn

from blueprint.utils import distributed, random


class FlowMatchingTrainer(nn.Module):
    """Train a model with flowmatching."""

    seed: Tensor

    def __init__(self, model: nn.Module, seed: int):
        super().__init__()
        self.model = model
        self.register_buffer("seed", torch.tensor(seed, dtype=torch.int64))
        # a plain attribute, not a buffer: DDP broadcasts buffers from rank 0 on
        # every forward, which would undo the per-rank offset
        self.rank = distributed.global_rank()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute one training step loss for a batch of distributions."""
        x0 = batch["sourcedist"]
        x1 = batch["targetdist"]

        random.derive_new_seed_(self.seed)
        gen = random.get_rank_generator(self.seed, self.rank)
        t = torch.rand((x0.size(0), 1), device=x0.device, dtype=x0.dtype, generator=gen)

        xt = t * x1 + (1.0 - t) * x0
        dx = x1 - x0
        dxpred = self.model(xt, t)
        loss = torch.nn.functional.mse_loss(dxpred, dx)
        return {"loss": loss}
