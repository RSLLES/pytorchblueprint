# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import wraps

import torch
from omegaconf import DictConfig


def handle_oom(cfg_attr_name="cfg"):
    """Handle out-of-memory errors by retrying with reduced batch size."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find the cfg object from args or kwargs
            cfg = kwargs.get(cfg_attr_name)
            if cfg is None:
                for arg in args:
                    if isinstance(arg, DictConfig):
                        cfg = arg
                        break
            if cfg is None:
                raise ValueError("DictConfig not found in arguments.")

            while True:
                try:
                    return func(*args, **kwargs)
                except RuntimeError as e:
                    oom = "out of memory" in str(e)
                    oom = oom or "OOM" in str(e)
                    if oom:
                        if cfg.batch_size > 1:
                            cfg.batch_size = max(1, cfg.batch_size // 2)
                            cfg.n_accum_steps = 2 * cfg.n_accum_steps
                            print(
                                f"OOM encountered. Retry with batch_size={cfg.batch_size} and n_accum_steps={cfg.n_accum_steps}."
                            )
                            torch.cuda.empty_cache()
                        else:
                            print(
                                "OOM encountered. Batch size is already 1. Cannot reduce further."
                            )
                            raise
                    else:
                        raise

        return wrapper

    return decorator
