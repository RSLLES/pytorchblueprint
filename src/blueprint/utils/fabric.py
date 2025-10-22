# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for fabric."""

import contextlib

from lightning_fabric import Fabric


def initialize_fabric(seed: int, devices: str | int = "auto"):
    """Initialize a Fabric instance with the given seed and devices."""
    fabric = Fabric(devices=devices)
    fabric.seed_everything(seed)
    fabric.launch()
    return fabric


@contextlib.contextmanager
def global_zero_context(ctx_manager_factory: callable, fabric: Fabric):
    """Wrap a context manager so it only activates on global zero device."""
    if fabric.is_global_zero:
        with ctx_manager_factory() as ctx:
            yield ctx
    else:
        yield None
