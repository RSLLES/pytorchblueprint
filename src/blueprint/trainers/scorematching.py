# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Flowmatching training step."""

from torch import Tensor, nn

from blueprint.losses import HyvarinenLoss


class ScoreMatchingTrainer(nn.Module):
    """Train a model with Hyvarinen Score Matching."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.loss_func = HyvarinenLoss()

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Compute one training step loss for a batch of distributions."""
        loss = self.loss_func(score_net=self.model, x=x)
        return {"loss": loss}
