# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Moment matching training step."""

from torch import Tensor, nn

from blueprint.losses import MMDBlock


class MomentMatchingTrainer(nn.Module):
    """Train a model with Hyvarinen Score Matching."""

    def __init__(self, model: nn.Module, kernel: nn.Module):
        super().__init__()
        self.model = model
        self.loss_func = MMDBlock(block_size=128, kernel=kernel)

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute one training step loss for a batch of distributions."""
        x_source = x["sourcedist"]
        x_target = x["targetdist"]
        x_pred = self.model(x_source)
        loss = self.loss_func(x_pred, x_target)
        return {"loss": loss}
