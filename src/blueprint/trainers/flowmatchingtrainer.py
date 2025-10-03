# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from blueprint.utils import random


class FlowMatchingTrainer(nn.Module):
    """Train a model with flow-matching."""

    def __init__(self, model: nn.Module, seed: int):
        super().__init__()
        self.model = model
        self.register_buffer("seed", torch.tensor(seed))

    def forward(self, x):
        """Compute one training step loss for a batch of distributions."""
        x0 = x["sourcedist"]
        x1 = x["targetdist"]

        gen = random.get_generator(self.seed)
        self.seed = random.derive_new_seed(self.seed)
        t = torch.rand((x0.size(0), 1), device=x0.device, dtype=x0.dtype, generator=gen)

        xt = t * x1 + (1.0 - t) * x0
        dx = x1 - x0
        dxpred = self.model(xt, t)
        loss = torch.nn.functional.mse_loss(dxpred, dx)
        return {"loss": loss}
