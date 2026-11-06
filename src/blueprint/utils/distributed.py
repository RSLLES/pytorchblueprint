# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Turn a per-rank batch statistic into the one every rank shares.

Only the collectives themselves are hidden from the compiler: the single-device path
returns before reaching them, so a compiled graph stays unbroken on one GPU.
"""

import torch
import torch.distributed as dist
from torch import Tensor


def is_distributed() -> bool:
    """Report whether collectives are available on a live process group."""
    return dist.is_available() and dist.is_initialized()


def global_rank() -> int:
    """Return this process's rank, 0 when running on a single device."""
    return dist.get_rank() if is_distributed() else 0


@torch.compiler.disable()
def _all_reduce(x: Tensor, op: dist.ReduceOp) -> Tensor:
    x = x.clone()
    dist.all_reduce(x, op=op)
    return x


def reduce_across_ranks(x: Tensor, op: dist.ReduceOp = dist.ReduceOp.AVG) -> Tensor:
    """Combine a gradient-free statistic so every rank reads the same value."""
    return _all_reduce(x, op) if is_distributed() else x
