# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions to inspect models."""

from datetime import datetime

import torch
from lightning_fabric import Fabric
from torch import nn


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def present_model(model: nn.Module) -> None:
    """Print the model name and its trainable parameter count."""
    print(f"Model '{model._get_name()}'; {count_parameters(model):,} parameters.")


def explain_model(model: nn.Module, dataloader, fabric: Fabric) -> None:
    """Run torch._dynamo.explain to inspect Dynamo's compilation process."""
    if fabric.is_global_zero:
        print("Running torch._dynamo.explain...")
    batch = next(iter(dataloader))
    explanation = torch._dynamo.explain(model)(batch)
    if fabric.is_global_zero:
        filename = "explain_" + datetime.now().strftime("%FT%X") + ".log"
        with open(filename, "w") as f:
            f.write(str(explanation))
        print(f"Explanation saved at '{filename}'.")
