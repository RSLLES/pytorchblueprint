# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from blueprint import utils


@pytest.mark.parametrize("world_size", [1, 2])
def test_total_steps_matches_dataloader_steps(world_size):
    """Total steps count the optimizer steps the per-rank dataloaders really take."""
    n_samples = 1000
    cfg = OmegaConf.create(
        {"runtime": {"n_epochs": 2, "batch_size": 64, "n_accum_steps": 1}}
    )
    cfg = utils.config.add_total_steps(cfg, n_samples=n_samples, world_size=world_size)
    samples = TensorDataset(torch.arange(n_samples))
    sampler = DistributedSampler(
        samples, num_replicas=world_size, rank=0, shuffle=False
    )
    dl = DataLoader(samples, batch_size=64, sampler=sampler)
    assert cfg.runtime.total_steps == 2 * len(dl)
