# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from blueprint.losses import BaselineReinforce


@pytest.mark.parametrize("batch_size", [1, 64])
def test_baseline_reinforce_gradient_stays_bounded(batch_size: int):
    """The first calls, with little history in the EMAs, yield moderate gradients."""
    torch.manual_seed(0)
    reinforce = BaselineReinforce()
    for _ in range(5):
        log_prob = torch.zeros(batch_size, requires_grad=True)
        reinforce(torch.randn(batch_size), log_prob).backward()
        assert log_prob.grad is not None
        assert log_prob.grad.norm() < 10.0
