# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import blueprint
from blueprint.datasets import Sample2DDataset
from blueprint.datasets.sample2Ddataset import sample_cardioid, sample_spiral


def test_sample_gaussian(n: int = 10000, seed: int = 0):
    """Gaussian samples are standard normal."""
    gen = blueprint.utils.random.get_generator(seed)
    x = blueprint.datasets.sample2Ddataset.sample_gaussian(n, gen=gen)
    mean, std = x.mean(dim=0), x.std(dim=0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=2e-2, rtol=0.0)
    assert torch.allclose(std, torch.ones_like(mean), atol=2e-2, rtol=0.0)


@pytest.mark.parametrize("sampler", [sample_spiral, sample_cardioid])
def test_sample_shape_and_finite(sampler, n: int = 10000, seed: int = 0):
    """Spiral and cardioid samples are finite 2D points."""
    gen = blueprint.utils.random.get_generator(seed)
    x = sampler(n, gen=gen)
    assert x.shape == (n, 2)
    assert torch.isfinite(x).all()


def test_sample2D_dataset_rejects_an_unknown_distribution():
    """An unsupported distribution name is a ValueError naming it."""
    with pytest.raises(ValueError, match="moon"):
        Sample2DDataset(1, seed=0, dist="moon")
