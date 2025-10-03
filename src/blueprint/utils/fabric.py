# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from lightning_fabric import Fabric


def initialize_fabric(seed: int, devices: str | int = "auto"):
    """Initialize a Fabric instance with the given seed and devices."""
    fabric = Fabric(devices=devices)
    fabric.seed_everything(seed)
    fabric.launch()
    return fabric
