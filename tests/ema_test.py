# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from blueprint.utils.ema import ExpMovingAverage


def test_early_estimate_is_the_running_average():
    """Long before 1/lambd updates, every update weighs about the same."""
    ema = ExpMovingAverage(lambd=1e-3)
    for x in torch.arange(1.0, 6.0):
        ema.update(x)
    torch.testing.assert_close(ema.value(), torch.tensor([3.0]), atol=1e-2, rtol=0)


def test_update_keeps_a_single_compiled_graph():
    """The EMA must not split the graph of a compiled model."""
    ema = ExpMovingAverage(lambd=1e-3)
    ema.update(torch.ones(1))
    explanation = torch._dynamo.explain(ema)(torch.ones(1))
    assert explanation.graph_break_count == 0
