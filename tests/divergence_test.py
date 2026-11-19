# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch

from blueprint.utils.divergence import detect_divergence


@pytest.mark.parametrize("last", [math.nan, math.inf, -math.inf])
def test_non_finite_loss_diverges_before_window_fills(last):
    """A non-finite loss diverges even before the history fills the window."""
    assert detect_divergence([torch.tensor(1.0), torch.tensor(last)])


def test_flat_history_tolerates_tiny_increase():
    """A tiny increase after a flat history is not a divergence."""
    history = [torch.tensor(0.5)] * 6 + [torch.tensor(0.5001)]
    assert not detect_divergence(history)


def test_spike_after_flat_history_diverges():
    """A large spike after a flat history is a divergence."""
    history = [torch.tensor(0.5)] * 6 + [torch.tensor(1.0)]
    assert detect_divergence(history)
