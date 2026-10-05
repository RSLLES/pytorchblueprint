# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Under DDP, every rank must draw its own random stream, as one larger batch would."""

import torch
from lightning.fabric import Fabric
from torch import Tensor, nn

from blueprint.trainers import FlowMatchingTrainer


class _RecordTime(nn.Module):
    """Velocity model stub that keeps the last flow time it was given."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.t = torch.empty(0)

    def forward(self, xt: Tensor, t: Tensor) -> Tensor:
        self.t = t
        return self.linear(xt)


def _draw_flow_times(fabric: Fabric | None) -> Tensor:
    recorder = _RecordTime()
    trainer = FlowMatchingTrainer(recorder, seed=0)
    if fabric is not None:
        trainer = fabric.setup(trainer)
    batch = {"sourcedist": torch.zeros(16, 2), "targetdist": torch.ones(16, 2)}
    trainer(batch)
    return recorder.t.detach()


def _draw_on_each_rank(fabric: Fabric) -> list[float]:
    t = _draw_flow_times(fabric)
    t_all = fabric.all_gather(t)
    assert not torch.equal(t_all[0], t_all[1]), "every rank drew the same flow times"
    return t.flatten().tolist()  # a tensor would not outlive the spawned process


def test_ranks_draw_distinct_flow_times_and_rank_zero_matches_one_device():
    """DDP adds fresh samples per rank, while rank 0 keeps the single-device stream."""
    fabric = Fabric(accelerator="cpu", devices=2, strategy="ddp_spawn")
    t_rank0 = fabric.launch(_draw_on_each_rank)
    assert t_rank0 == _draw_flow_times(fabric=None).flatten().tolist()
