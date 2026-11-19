# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from blueprint.metrics import MedianOfMeans


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_update_does_not_wait_for_the_gpu():
    """Updated every training step, the metric must not stall the CPU on the GPU."""
    metric = MedianOfMeans(n_groups=1).to("cuda")
    value = torch.ones((), device="cuda")
    torch.cuda.set_sync_debug_mode("error")
    try:
        metric.update(value)
        metric.update(2.0)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert metric.compute() == 1.5
