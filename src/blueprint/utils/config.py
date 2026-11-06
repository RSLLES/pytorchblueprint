# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for hydra configuration."""

import math

from omegaconf import DictConfig, OmegaConf

from blueprint.utils.git import get_head_commit_hash


def initialize_config(cfg: DictConfig):
    """Initialize configuration with resolvers and git commit hash."""
    OmegaConf.set_struct(cfg, False)
    if not OmegaConf.has_resolver("pow2"):
        OmegaConf.register_new_resolver("pow2", lambda n: 2 ** int(n))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg.runtime.git_commit_hash = get_head_commit_hash()
    return cfg


def add_eff_batch_size(cfg: DictConfig, world_size: int):
    """Compute effective batch size and assign to configuration."""
    cfg.runtime.eff_batch_size = (
        cfg.runtime.batch_size * world_size * cfg.runtime.n_accum_steps
    )
    return cfg


def add_total_steps(cfg: DictConfig, n_samples: int, world_size: int) -> DictConfig:
    """Count the optimizer steps each rank takes over training, -1 if unbounded."""
    cfgr = cfg.runtime
    if cfgr.n_epochs <= 0:
        cfgr.total_steps = -1
        return cfg
    n_batches = math.ceil(math.ceil(n_samples / world_size) / cfgr.batch_size)
    cfgr.total_steps = cfgr.n_epochs * (n_batches // cfgr.n_accum_steps)
    return cfg
