# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Muon on a model's hidden matrices, AdamW on every other parameter.

The pattern follows nanochat (https://github.com/karpathy/nanochat), Dion
(https://github.com/microsoft/dion) and Keller Jordan's Muon
(https://github.com/KellerJordan/Muon): input and output layers stay on AdamW.
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW, Muon, Optimizer


def split_param_groups(module: nn.Module, model: nn.Module) -> list[dict[str, Any]]:
    """Split module's parameters by whether model lists them in hidden_matrices().

    Defining hidden_matrices() is optional; only models trained with MuonAdamW need it.
    """
    if not hasattr(model, "hidden_matrices"):
        return [{"params": list(module.parameters()), "use_muon": False}]
    hidden = list(model.hidden_matrices())
    hidden_ids = {id(p) for p in hidden}
    others = [p for p in module.parameters() if id(p) not in hidden_ids]
    return [
        {"params": hidden, "use_muon": True},
        {"params": others, "use_muon": False},
    ]


class MuonAdamW(Optimizer):
    """Step Muon on the groups flagged use_muon and AdamW on the others."""

    def __init__(
        self,
        params: Iterable[dict[str, Any]],
        muon: Callable[..., Muon],
        adamw: Callable[..., AdamW],
    ) -> None:
        groups = list(params)
        muon_groups = [g for g in groups if g["use_muon"]]
        if not muon_groups:
            raise ValueError("MuonAdamW needs the model to define hidden_matrices().")
        self.muon = muon(muon_groups)
        self.adamw = adamw([g for g in groups if not g["use_muon"]])
        # sharing the inner group dicts lets one scheduler drive both optimizers
        super().__init__(self.muon.param_groups + self.adamw.param_groups, {})

    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # noqa: D102
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon.step()
        self.adamw.step()
        return loss

    def state_dict(self) -> dict[str, Any]:  # noqa: D102
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # noqa: D102
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])
        # the inner loads replace their group dicts, which the scheduler must reach
        self.param_groups = self.muon.param_groups + self.adamw.param_groups
