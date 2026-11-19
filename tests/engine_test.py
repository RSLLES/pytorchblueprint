# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch
from lightning.fabric import Fabric
from torch import Tensor, nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader

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
        self.progress: list[float] = []

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        self.progress.append(float(batch["progress"]))
        scale = self.model(batch["sourcedist"]).sum() * 0 + 1
        return {"loss": scale * self.epoch_losses[int(batch["step"]) // 2]}


def _dl(length: int) -> DataLoader:
    ds = StackDataset(
        sourcedist=Sample2DDataset(length, seed=0, dist="gaussian"),
        targetdist=Sample2DDataset(length, seed=1, dist="spiral"),
    )
    return utils.dataloader.build_dl(
        ds, batch_size=4, n_workers=0, shuffle=False, world_size=1
    )


def _setup(epoch_losses: list[float]):
    fabric = Fabric(accelerator="cpu", devices=1)
    model = MomentNet(input_dim=2, inner_dim=8)
    scripted = _ScriptedLoss(model, epoch_losses)
    optimizer = SGD(scripted.parameters(), lr=0.0)
    scheduler = ConstantLR(optimizer, factor=1.0, total_iters=0)
    training_module, optimizer = fabric.setup(scripted, optimizer)
    return fabric, model, scripted, training_module, optimizer, scheduler


def _train(
    tmp_path,
    epoch_losses: list[float],
    watched_metric: str = "loss",
    watched_metric_strategy: str = "min",
    val_stride: int = 1,
    detect_divergence: bool = False,
    grad_clip_norm: float | None = None,
    total_steps: int = -1,
) -> _ScriptedLoss:
    fabric, model, scripted, training_module, optimizer, scheduler = _setup(
        epoch_losses
    )
    dl = _dl(8)
    engine.train(
        fabric=fabric,
        model=model,
        training_module=training_module,
        optimizer=optimizer,
        scheduler=scheduler,
        dl_train=dl,
        dl_val=dl,
        watched_metric=watched_metric,
        watched_metric_strategy=watched_metric_strategy,
        log_dir=str(tmp_path),
        cfg_str="",
        n_epochs=3,
        val_stride=val_stride,
        total_steps=total_steps,
        grad_clip_norm=grad_clip_norm,
        detect_divergence=detect_divergence,
    )
    return scripted


@pytest.mark.parametrize(("strategy", "expected_epoch"), [("min", 1), ("max", 0)])
def test_best_checkpoint_follows_strategy(tmp_path, strategy, expected_epoch):
    """The best checkpoint holds the epoch that is best under the watched strategy."""
    _train(tmp_path, [3.0, 1.0, 2.0], watched_metric_strategy=strategy)
    best = torch.load(tmp_path / "best.tar", weights_only=False)
    assert best["epoch"] == expected_epoch


@pytest.mark.parametrize("val_stride", [1, 0])
def test_misspelled_watched_metric_raises(tmp_path, val_stride):
    """A misspelled watched metric must raise, not silently skip best checkpoints."""
    with pytest.raises(ValueError, match="lsos"):
        _train(tmp_path, [1.0, 1.0, 1.0], watched_metric="lsos", val_stride=val_stride)


def test_validation_metric_can_be_watched_across_skipped_epochs(tmp_path):
    """Epochs that skip validation lack its metrics, which is no reason to fail."""
    _train(tmp_path, [1.0, 1.0, 1.0], watched_metric="earth_mover_dist", val_stride=2)
    best = torch.load(tmp_path / "best.tar", weights_only=False)
    assert best["epoch"] in (1, 2)


def test_second_run_in_the_same_log_dir_raises(tmp_path):
    """Two runs that picked the same log directory must not overwrite each other."""
    _train(tmp_path, [1.0, 1.0, 1.0])
    with pytest.raises(FileExistsError):
        _train(tmp_path, [1.0, 1.0, 1.0])


@pytest.mark.parametrize("precision", ["bf16-mixed", "bf16-true", "64-true"])
def test_validation_runs_at_the_training_precision(precision):
    """Validation runs the model with the dtypes and autocast state of training."""
    fabric = Fabric(accelerator="cpu", devices=1, precision=precision)
    model = MomentNet(input_dim=2, inner_dim=8)
    training_module = fabric.setup(nn.Sequential(model))
    calls = []

    def record(module: nn.Module, inputs: tuple[Tensor], output: Tensor) -> None:
        autocast = torch.is_autocast_enabled("cpu")
        calls.append((inputs[0].dtype, torch.get_default_dtype(), autocast))

    model.register_forward_hook(record)
    training_module(torch.randn(4, 2))
    engine.validate(fabric=fabric, dl=_dl(4), model=model)
    assert calls[1] == calls[0]


def test_nan_gradients_revert_to_best_checkpoint(tmp_path):
    """NaN gradients reach the divergence check, which restores the best weights."""
    scripted = _train(
        tmp_path, [3.0, 1.0, math.nan], grad_clip_norm=1.0, detect_divergence=True
    )
    assert all(torch.isfinite(p).all() for p in scripted.parameters())


def test_epoch_without_optimizer_step_returns_nan_loss():
    """An epoch shorter than one accumulation reports a NaN loss and takes no step."""
    fabric, _, _, training_module, optimizer, scheduler = _setup([1.0, 1.0, 1.0])
    metrics, step = engine.train_one_epoch(
        fabric=fabric,
        dl=_dl(4),
        n_accum_steps=2,
        optimizer=optimizer,
        scheduler=scheduler,
        step=5,
        total_steps=-1,
        clock=None,
        grad_clip_norm=None,
        training_module=training_module,
        enable_profiling=False,
    )
    assert math.isnan(metrics["loss"])
    assert step == 5


def test_progress_spans_training(tmp_path):
    """Progress runs from 0 towards 1 over the whole run, one value per step."""
    scripted = _train(tmp_path, [1.0, 1.0, 1.0], total_steps=6)
    assert scripted.progress == pytest.approx([s / 6 for s in range(6)])
