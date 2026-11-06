# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""General utility functions for PyTorch."""

import logging

import torch
from torch import Tensor

from .format import format_string


def initialize_torch(detect_anomaly: bool = False):
    """Set PyTorch's precision, control detect_anomaly, improve logs."""
    if detect_anomaly:
        print("Warning: detect_anomaly is enabled.")
    torch.autograd.set_detect_anomaly(detect_anomaly)
    torch.set_float32_matmul_precision("high")
    torch.set_printoptions(linewidth=160)
    torch._logging.set_logs(all=logging.WARNING)
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)


def reduce(x: Tensor, dim: int, mode: str) -> Tensor:
    """Reduce a tensor. Supports 'none' | 'mean' | 'sum' | 'min' | 'max'."""
    mode = format_string(mode)
    if mode == "none":
        return x
    if mode == "sum":
        return x.sum(dim=dim)
    if mode == "mean":
        return x.mean(dim=dim)
    if mode == "min":
        return x.amin(dim=dim)
    if mode == "max":
        return x.amax(dim=dim)
    raise ValueError(
        f"Unsupported reduction mode: {mode}."
        "Use 'none' | 'mean' | 'sum' | 'min' | 'max'."
    )
