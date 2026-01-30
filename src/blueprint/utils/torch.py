# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""General utility functions for PyTorch."""

import logging
import os

import torch
from torch import Tensor
from torch.nn.modules.utils import _pair, _quadruple, _single, _triple

from .format import format_size, format_string


def initialize_torch(detect_anomaly: bool = False):
    """Set PyTorch's precision, control detect_anomaly, improve logs."""
    if detect_anomaly:
        print("Warning: detect_anomaly is enabled.")
    torch.autograd.set_detect_anomaly(detect_anomaly)
    torch.set_printoptions(linewidth=160)
    torch._logging.set_logs(all=logging.WARNING)
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)  # weird


def hash_tensor(tensor: Tensor):
    """Hash a tensor (in a trivial / non-cryptographic way)."""
    if "PYTHONHASHSEED" in os.environ:
        if os.environ["PYTHONHASHSEED"] != "1":
            raise ValueError("Environment variable 'PYTHONHASHSEED' must be set to 1.")
    else:
        os.environ["PYTHONHASHSEED"] = "1"
    return hash(tuple(tensor.flatten().tolist()))


def are_broadcastable(shape1, shape2):
    """Determine whether two shapes are broadcastable."""
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a != 1 and b != 1 and a != b:
            return False
    return True


def get_memory_size(tensor: Tensor, as_string: bool = True) -> int | str:
    """Return the size occupied by the tensor in memory in bytes."""
    size = tensor.element_size() * tensor.nelement()
    if not as_string:
        return size
    return format_size(size)


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


"""Helper functions to convert element to tuples"""
to_single = _single
to_pair = _pair
to_triple = _triple
to_quadruple = _quadruple
