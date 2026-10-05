# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import pytest

from blueprint import engine
from blueprint.engine.training import should_validate


@pytest.mark.parametrize(
    ("val_stride", "expected_epochs"),
    [(1, [0, 1, 2, 3, 4]), (2, [1, 3, 4]), (0, []), (-1, [4])],
)
def test_should_validate_follows_the_stride_and_keeps_the_last_epoch(
    val_stride, expected_epochs
):
    """A positive stride validates periodically and at the end; negative, at the end."""
    epochs = [e for e in range(5) if should_validate(val_stride, epoch=e, n_epochs=5)]
    assert epochs == expected_epochs


def _train_with(**kwargs):
    defaults: dict[str, Any] = dict(
        fabric=None,
        model=None,
        training_module=None,
        optimizer=[],
        scheduler=[],
        dl_train=None,
        dl_val=None,
        evaluators=[],
        log_dir="",
        cfg_str="",
        watched_metric="loss",
        watched_metric_strategy="min",
        detect_divergence=False,
    )
    return engine.train(**(defaults | kwargs))


def test_train_rejects_an_unknown_watched_metric_strategy():
    """Only min and max define which checkpoint is the best."""
    with pytest.raises(ValueError, match="watched_metric_strategy"):
        _train_with(watched_metric_strategy="median")


@pytest.mark.parametrize("options", [{"detect_divergence": True}, {"patience": 3}])
def test_train_rejects_best_checkpoint_options_without_watched_metric(options):
    """Divergence recovery and early stopping both rely on a best checkpoint."""
    with pytest.raises(ValueError, match="watched_metric"):
        _train_with(watched_metric=None, **options)
