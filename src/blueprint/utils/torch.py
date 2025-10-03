# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import torch
from torch import Tensor


def initialize_torch():
    """Set PyTorch to high matmul precision and suppress debug logs."""
    torch.set_float32_matmul_precision("high")
    torch._logging.set_logs(all=logging.WARNING)


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
