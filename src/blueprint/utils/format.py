# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for formatting objects to strings."""

import re
import warnings
from math import isinf, isnan

import numpy as np
import torch
from sigfig import round
from torch import Tensor


def format_metrics(metrics: dict) -> str:
    """Format metrics in a text-logging format."""
    return "; ".join(f"{name}:{format_object(x)}" for name, x in metrics.items())


def format_object(x: any) -> str:
    """Format all sorts of objects to strings."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x, Tensor):
        if x.nelement() > 1:
            strings = [format_object(e) for e in x]
            return "[" + ",".join(strings) + "]"
        x = x.item()
    if isinstance(x, float) or isinstance(x, int):
        return format_number(x)
    return f" {x}"  # other types are printed as they are...


def format_number(x: float | int, n_digits: int = 3) -> str:
    """Format a number with a given number of significant digits, fixed width."""
    target_width = n_digits + 5

    if isinstance(x, int):
        s = str(x)
        return f" {s}" if s[0] != "-" else s

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        if isnan(x) or isinf(x):
            s = str(x)
            return f" {s}" if s[0] != "-" else s
        s = round(x, sigfigs=n_digits, type=str)

    display = f" {s}" if s[0] != "-" else s

    if len(display) >= target_width:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            s = round(x, sigfigs=n_digits, type=str, notation="sci")
        display = f" {s}" if s[0] != "-" else s

    return display.ljust(target_width)


def format_string(s: str) -> str:
    """Format a string to lowercase with only alphanumeric characters."""
    s = s.lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


def format_size(size: int, n_digits: int = 3) -> str:
    """Format a number of bytes into a readable string with size unit (KiB, ...)."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if size < 1024:
            break
        size //= 1024
    with warnings.catch_warnings():  # sigfig warns when x has less than n_digits
        warnings.filterwarnings("ignore", category=UserWarning)
        return round(size, sigfigs=n_digits, type=str) + unit
