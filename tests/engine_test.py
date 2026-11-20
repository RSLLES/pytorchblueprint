# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest
import torch
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ConstantLR

from blueprint import engine, utils
from blueprint.datasets import Sample2DDataset
from blueprint.datasets.utils import StackDataset
from blueprint.models import MomentNet


class _ScriptedLoss(nn.Module):
    """Training module whose loss follows a per-epoch script, at two steps per epoch."""

    def __init__(self, model: nn.Module, epoch_losses: list[float]):
        super().__init__()
        self.model = model
        self.epoch_losses = epoch_losses

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        scale = self.model(batch["sourcedist"]).sum() * 0 + 1
        return {"loss": scale * self.epoch_losses[int(batch["step"]) // 2]}


@pytest.mark.parametrize(("strategy", "expected_epoch"), [("min", 1), ("max", 0)])
def test_best_checkpoint_follows_strategy(
    tmp_path: Path, strategy: str, expected_epoch: int
):
    """The best checkpoint holds the epoch that is best under the watched strategy."""
    fabric = Fabric(accelerator="cpu", devices=1)
    model = MomentNet(input_dim=2, inner_dim=8)
    scripted = _ScriptedLoss(model, [3.0, 1.0, 2.0])
    optimizer = SGD(scripted.parameters(), lr=0.0)
    scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)
    training_module, optimizer = fabric.setup(scripted, optimizer)
    ds = StackDataset(
        sourcedist=Sample2DDataset(8, seed=0, dist="gaussian"),
        targetdist=Sample2DDataset(8, seed=1, dist="spiral"),
    )
    dl = utils.dataloader.build_dl(
        ds, batch_size=4, n_workers=0, shuffle=False, world_size=1
    )
    engine.train(
        fabric=fabric,
        model=model,
        training_module=training_module,
        optimizer=optimizer,
        scheduler=scheduler,
        dl_train=dl,
        dl_val=dl,
        watched_metric="loss",
        watched_metric_strategy=strategy,
        log_dir=str(tmp_path),
        cfg_str="",
        n_epochs=3,
        detect_divergence=False,
    )
    best = torch.load(tmp_path / "best.tar", weights_only=False)
    assert best["epoch"] == expected_epoch
