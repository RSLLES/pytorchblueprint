# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools

import pytest
import torch
from torch import nn
from torch.optim import AdamW, Muon
from torch.optim.lr_scheduler import CosineAnnealingLR

from blueprint.models import MomentNet
from blueprint.optimizers import MuonAdamW, split_param_groups


def _muon_adamw(model: nn.Module) -> MuonAdamW:
    return MuonAdamW(
        split_param_groups(model, model),
        muon=functools.partial(Muon, lr=0.02),
        adamw=functools.partial(AdamW, lr=0.003),
    )


def test_split_param_groups_covers_each_parameter_once():
    """Every parameter lands in exactly one group; Muon gets the hidden matrices."""
    model = MomentNet(input_dim=2, inner_dim=8)
    groups = split_param_groups(model, model)
    grouped = [id(p) for g in groups for p in g["params"]]
    assert sorted(grouped) == sorted(id(p) for p in model.parameters())
    assert len(grouped) == len(set(grouped))
    muon_group = next(g for g in groups if g["use_muon"])
    assert [id(p) for p in muon_group["params"]] == [
        id(p) for p in model.hidden_matrices()
    ]


def test_split_param_groups_single_group_without_hidden_matrices():
    """A model that does not declare hidden matrices needs nothing from the split."""
    model = nn.Linear(2, 2)
    groups = split_param_groups(model, model)
    assert len(groups) == 1
    assert not groups[0]["use_muon"]


def test_muon_adamw_requires_hidden_matrices():
    """Without hidden matrices, Muon has nothing to optimize."""
    with pytest.raises(ValueError, match="hidden_matrices"):
        _muon_adamw(nn.Linear(2, 2))


def test_muon_adamw_scheduler_drives_inner_lrs_after_resume():
    """A resumed scheduler still sets the learning rates the inner optimizers use."""
    model = MomentNet(input_dim=2, inner_dim=8)
    optimizer = _muon_adamw(model)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    for _ in range(2):
        model(torch.randn(16, 2)).square().mean().backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    resumed = _muon_adamw(model)
    resumed_scheduler = CosineAnnealingLR(resumed, T_max=10)
    resumed.load_state_dict(optimizer.state_dict())
    resumed_scheduler.load_state_dict(scheduler.state_dict())
    lrs_before = [g["lr"] for g in resumed.param_groups]
    resumed_scheduler.step()

    inner_groups = resumed.muon.param_groups + resumed.adamw.param_groups
    for outer, inner, lr_before in zip(
        resumed.param_groups, inner_groups, lrs_before, strict=True
    ):
        assert inner["lr"] == outer["lr"]
        assert inner["lr"] != lr_before
