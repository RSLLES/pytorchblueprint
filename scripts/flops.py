# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Count the number of flops of SHOT"""

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from ptflops import get_model_complexity_info

import blueprint


@hydra.main(version_base=None, config_path="../config", config_name="shot")
def main(cfg: DictConfig):
    blueprint.utils.torch.initialize_torch()
    fabric = blueprint.utils.fabric.initialize_fabric(cfg.seed)
    if fabric.world_size > 1:
        raise ValueError("Export only works with one gpu.")

    with fabric.init_module(empty_init=False):
        model = instantiate(cfg.model)

    macs, params = get_model_complexity_info(
        model.encoder_net,
        (1, 64, 64),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    print(f"Encoder: {macs}, Params={params}")

    macs, params = get_model_complexity_info(
        model.decoder_net,
        (model.dim, 64, 64),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    print(f"Decoder: {macs}, Params={params}")

    macs, params = get_model_complexity_info(
        model.residual_net,
        (model.dim * (model.n_frames + 2), 64, 64),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    print(f"Residual Net: {macs}, Params={params}")

    macs, params = get_model_complexity_info(
        model.renderer,
        (32 * 32, 4),
        input_constructor=lambda shape: {
            "x": torch.ones((1,) + shape),
            "bg": torch.rand((1, 1, 64, 64)),
        },
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )
    print(f"Renderer: {macs}, Params={params}")


if __name__ == "__main__":
    main()
