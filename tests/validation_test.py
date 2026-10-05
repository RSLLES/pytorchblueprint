# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from lightning.fabric import Fabric
from matplotlib.figure import Figure
from torch import nn

from blueprint import engine


def _batches(with_target: bool) -> list[dict]:
    gen = torch.Generator().manual_seed(0)
    batches = []
    for _ in range(2):
        batch = {"sourcedist": torch.randn(8, 2, generator=gen)}
        if with_target:
            batch["targetdist"] = torch.randn(8, 2, generator=gen)
        batches.append(batch)
    return batches


def _validate(with_target: bool) -> dict:
    return engine.validate(
        fabric=Fabric(accelerator="cpu", devices=1),
        dl=_batches(with_target),  # ty: ignore[invalid-argument-type]
        model=nn.Identity(),
        evaluators=[engine.WassersteinEvaluator(), engine.ScatterPlotEvaluator()],
    )


def test_validate_merges_evaluator_metrics_under_val_prefix():
    """Each evaluator contributes its metrics, namespaced for the logs."""
    metrics = _validate(with_target=True)
    assert set(metrics) == {"val/earth_mover_dist", "val/fig"}
    assert isinstance(metrics["val/fig"], Figure)


def test_validate_skips_evaluators_whose_keys_are_missing():
    """A dataset without targets runs only the evaluators that do not need them."""
    assert _validate(with_target=False) == {}
