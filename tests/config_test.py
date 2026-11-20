# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from blueprint import utils


@pytest.mark.parametrize("world_size", [1, 2])
def test_total_steps_matches_dataloader_steps(world_size: int):
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


@pytest.mark.parametrize(
    "config_name",
    ["flowmatching", "momentmatching", "scorematching", "wassersteinmatching"],
)
def test_datasets_get_distinct_seeds(config_name: str):
    """Source and target of a split, and the val and test sets, never share a seed."""
    config_dir = Path(__file__).parents[1] / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = utils.config.initialize_config(compose(config_name=config_name))
    train, val, test = cfg.ds_train, cfg.ds_val, cfg.ds_test
    assert train.sourcedist.seed != train.targetdist.seed
    seeds = [
        ds.seed
        for ds in (val.sourcedist, val.targetdist, test.sourcedist, test.targetdist)
    ]
    assert len(set(seeds)) == 4
