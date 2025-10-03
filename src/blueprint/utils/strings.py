# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for formatting mutliple objects to strings."""

import re

import numpy as np
import torch
from torch import Tensor


def format_metrics(metrics: dict) -> str:
    """Format metrics for loging in the console."""
    return "; ".join(f"{name}: {format_numeric(x)}" for name, x in metrics.items())


def format_numeric(x: Tensor | np.ndarray | float | int, n_digits: int = 3) -> str:
    """Format all sorts of numeric object with given significant digits."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, Tensor):
        if x.nelement() > 1:
            strings = [format_numeric(e, n_digits=n_digits) for e in x]
            return "[" + ",".join(strings) + "]"
        x = x.item()
    if isinstance(x, int):
        x = float(x)
    if isinstance(x, float):
        return format_float(x, n_digits=n_digits)
    return str(x)  # other types are printed as they are...


def format_float(value: float, n_digits: int = 3) -> str:
    """Format a float with given significant digits."""
    size = n_digits + 5  # dot, e, +-, 2 digits in mantissa
    # value = f"{value:{n_digits}g}"
    # return value
    # if len(value) > n_digits:
    #     raise NotImplementedError()
    # if len(value) < n_digits
    return f"{value:>{size}.{n_digits}g}"


def format_string(s: str) -> str:
    """Format a string to lowercase with only alphanumeric characters."""
    s = s.lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s
