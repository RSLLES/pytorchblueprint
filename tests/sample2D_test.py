# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import torch

import blueprint


def test_sample_gaussian(n: int = 10000, atol=2e-2, seed: int = 0):
    gen = blueprint.utils.random.get_generator(seed)
    x = blueprint.datasets.sample2Ddataset.sample_gaussian(n, gen=gen)
    mean, std = x.mean(dim=0), x.std(dim=0)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=atol, rtol=0.0)
    assert torch.allclose(std, torch.ones_like(mean), atol=atol, rtol=0.0)
    plt.scatter(x[..., 0], x[..., 1])
    plt.savefig("/tmp/out.png")


def test_sample_spiral(n: int = 10000, atol=1e-2, seed: int = 0):
    gen = blueprint.utils.random.get_generator(seed)
    x = blueprint.datasets.sample2Ddataset.sample_spiral(n, gen=gen)
    plt.scatter(x[..., 0], x[..., 1])
    plt.savefig("/tmp/out.png")


def test_sample_cardioid(n: int = 10000, atol=1e-2, seed: int = 0):
    gen = blueprint.utils.random.get_generator(seed)
    x = blueprint.datasets.sample2Ddataset.sample_cardioid(n, gen=gen)
    plt.scatter(x[..., 0], x[..., 1])
    plt.savefig("/tmp/out.png")
