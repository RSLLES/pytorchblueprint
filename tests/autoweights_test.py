# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from blueprint.losses.autoweights import PeacockWeighting


def test_peacock_weighting_returns_a_finite_loss():
    """Peacock builds its loss EMA and weighs three losses into a finite total."""
    weighting = PeacockWeighting(n_losses=3)
    assert torch.isfinite(weighting(torch.tensor([1.0, 2.0, 3.0]))).all()
