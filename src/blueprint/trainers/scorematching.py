# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Score matching training step."""

from torch import Tensor, nn

from blueprint.losses import HyvarinenLoss


class ScoreMatchingTrainer(nn.Module):
    """Train a model with Hyvarinen Score Matching."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.loss_func = HyvarinenLoss()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute the score matching loss on a batch of target samples."""
        loss = self.loss_func(score_net=self.model, x=batch["targetdist"])
        return {"loss": loss}
