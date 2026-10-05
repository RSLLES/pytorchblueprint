# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the collective out-of-memory vote that keeps a retry in step."""

import pytest
from lightning.fabric import Fabric
from omegaconf import OmegaConf

from blueprint import utils


def _vote_with_one_rank_out_of_memory(fabric: Fabric) -> None:
    with pytest.raises(utils.oom.DistributedOutOfMemory):
        utils.oom.raise_on_any_rank_oom(fabric, fabric.global_rank == 1)


def _vote_with_every_rank_healthy(fabric: Fabric) -> None:
    utils.oom.raise_on_any_rank_oom(fabric, False)


def test_one_rank_out_of_memory_raises_on_every_rank():
    """A rank that retries alone desyncs DDP, so the vote must reach the other ranks."""
    Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn").launch(
        _vote_with_one_rank_out_of_memory
    )


def test_a_healthy_epoch_raises_nothing():
    """The vote runs every epoch; it must be silent when there is nothing to report."""
    Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn").launch(
        _vote_with_every_rank_healthy
    )


def test_handle_oom_retries_the_collective_error():
    """The vote's exception must reach the decorator that halves the batch size."""
    cfg = OmegaConf.create({"runtime": {"batch_size": 8, "n_accum_steps": 1}})
    attempts = []

    @utils.oom.handle_oom()
    def run(cfg):
        attempts.append(cfg.runtime.batch_size)
        if len(attempts) == 1:
            raise utils.oom.DistributedOutOfMemory("out of memory on 1 of the ranks.")

    run(cfg)
    assert attempts == [8, 4]
    assert cfg.runtime.n_accum_steps == 2
