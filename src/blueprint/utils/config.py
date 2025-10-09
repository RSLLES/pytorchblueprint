# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for hydra configuration."""

import yaml
from omegaconf import DictConfig, OmegaConf

from blueprint.utils.git import get_head_commit_hash


def initialize_config(cfg: DictConfig):
    """Initialize configuration with resolvers and git commit hash."""
    OmegaConf.set_struct(cfg, False)
    if not OmegaConf.has_resolver("pow2"):
        OmegaConf.register_new_resolver("pow2", lambda n: 2 ** int(n))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    cfg.git_commit_hash = get_head_commit_hash()
    return cfg


def add_git_commit_hash(cfg: DictConfig):
    """Add git commit hash to configuration."""
    cfg.git_commit_hash = get_head_commit_hash()
    return cfg


def add_eff_batch_size(cfg: DictConfig, world_size: int):
    """Compute effective batch size and assign to configuration."""
    cfg.eff_batch_size = cfg.batch_size * world_size * cfg.n_accum_steps
    return cfg


def add_total_steps(cfg: DictConfig, step_per_epoch: int):
    """Compute total training steps and assign to configuration."""
    if cfg.n_epochs > 0:
        cfg.total_steps = cfg.n_epochs * step_per_epoch // cfg.eff_batch_size
    else:
        cfg.total_steps = -1
    return cfg


def dump_config(cfg: DictConfig) -> str:
    """Dump configuration to YAML string."""
    return yaml.dump(OmegaConf.to_container(cfg), default_flow_style=False)
