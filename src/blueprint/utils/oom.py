# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility decorator to detect OOM and restart training with a reduced batch size."""

from collections.abc import Callable
from functools import wraps

import torch
from lightning.fabric import Fabric
from omegaconf import DictConfig


class DistributedOutOfMemory(RuntimeError):
    """At least one rank ran out of memory."""


def is_oom(error: BaseException) -> bool:
    """Report whether an exception is an out-of-memory failure."""
    return "out of memory" in str(error) or "OOM" in str(error)


def raise_on_any_rank_oom(fabric: Fabric, local_oom: bool):
    """Turn a one-rank OOM into an every-rank one, so the retry stays collective."""
    votes = torch.tensor(float(local_oom), device=fabric.device)
    n_oom = int(fabric.all_reduce(votes, reduce_op="sum"))  # ty:ignore[invalid-argument-type]
    if n_oom > 0:
        raise DistributedOutOfMemory(f"out of memory on {n_oom} of the ranks.")


def handle_oom[**P, R](
    cfg_attr_name: str = "cfg",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Handle out-of-memory errors by retrying with reduced batch size."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            cfg = kwargs.get(cfg_attr_name)
            if cfg is None:
                for arg in args:
                    if isinstance(arg, DictConfig):
                        cfg = arg
                        break
            if not isinstance(cfg, DictConfig):
                raise ValueError("DictConfig not found in arguments.")

            cfgr = cfg.runtime
            while True:
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    if not is_oom(e):
                        raise
                    if cfgr.batch_size == 1:
                        print(
                            "OOM encountered. Batch size is already 1. "
                            "Cannot reduce further."
                        )
                        raise
                    cfgr.batch_size = max(1, cfgr.batch_size // 2)
                    cfgr.n_accum_steps = 2 * cfgr.n_accum_steps
                    print(
                        f"OOM encountered. Retry with batch_size={cfgr.batch_size} "
                        f"and n_accum_steps={cfgr.n_accum_steps}."
                    )
                    torch.cuda.empty_cache()
                    torch._dynamo.reset()

        return wrapper

    return decorator
