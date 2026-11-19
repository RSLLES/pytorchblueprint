# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from blueprint.models.basics import UNet


def test_unet_accepts_activation_without_inplace():
    """A UNet builds and runs with an activation that takes no inplace argument."""
    model = UNet(1, 2, depth=1, init_features=4, act_layer=nn.GELU)
    assert model(torch.randn(1, 1, 8, 8)).shape == (1, 2, 8, 8)
