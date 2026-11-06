# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for formatting objects to strings."""

import re
import warnings
from math import isinf, isnan
from typing import Any

import numpy as np
import sigfig
import torch
from torch import Tensor


def format_metrics(metrics: dict[str, Any]) -> str:
    """Format metrics in a text-logging format."""
    return "; ".join(f"{name}:{format_object(x)}" for name, x in metrics.items())


def format_object(x: Any) -> str:
    """Format all sorts of objects to strings."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, Tensor):
        if x.nelement() > 1:
            strings = [format_object(e) for e in x]
            return "[" + ",".join(strings) + "]"
        x = x.item()
    if isinstance(x, float | int):
        return format_number(x)
    return f" {x}"  # other types are printed as they are...


def format_number(x: float | int, n_digits: int = 3) -> str:
    """Format a number with a given number of significant digits, fixed width."""
    if isinstance(x, int) or isnan(x) or isinf(x):
        return _pad_sign(str(x))
    target_width = n_digits + 5
    with warnings.catch_warnings():  # sigfig warns when x has fewer than n_digits
        warnings.filterwarnings("ignore", category=UserWarning)
        display = _pad_sign(sigfig.round(x, sigfigs=n_digits, type=str))
        if len(display) >= target_width:
            sci = sigfig.round(x, sigfigs=n_digits, type=str, notation="sci")
            display = _pad_sign(sci)
    return display.ljust(target_width)


def _pad_sign(s: str) -> str:
    """Prefix non-negative numbers with a space, so that signs align."""
    return s if s.startswith("-") else f" {s}"


def format_string(s: str) -> str:
    """Format a string to lowercase with only alphanumeric characters."""
    s = s.lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def format_duration(seconds: float) -> str:
    """Format a duration with its two largest units (3h42m, 12m30s, 45s)."""
    units = [("d", 86400), ("h", 3600), ("m", 60), ("s", 1)]
    remaining = int(seconds)
    parts = []
    for name, size in units:
        value, remaining = divmod(remaining, size)
        if value or parts:
            parts.append(f"{value}{name}")
        if len(parts) == 2:
            break
    return "".join(parts) or "0s"
