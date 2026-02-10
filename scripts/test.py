# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from blueprint import engine, utils


@hydra.main(version_base=None, config_path="../configs")
@torch.no_grad()
def test(cfg: DictConfig):
    """Perform an evaluation pass over the test dataset."""
    utils.torch.initialize_torch()
    fabric = utils.fabric.initialize_fabric(cfg.seed)
    cfg = utils.config.initialize_config(cfg)

    if fabric.is_global_zero:
        print(OmegaConf.to_yaml(cfg))

    # model
    with fabric.init_module(empty_init="weights_path" in cfg):
        model = instantiate(cfg.model)
    if fabric.is_global_zero:
        utils.model.present_model(model)

    if "weights_path" in cfg:
        epoch, step = utils.checkpoint.load_weights(
            fabric=fabric, ckpt_path=cfg.weights_path, model=model
        )
        if fabric.is_global_zero:
            print(f"Weights loaded from {cfg.weights_path}: epoch {epoch} step {step}.")
    else:
        if fabric.is_global_zero:
            print("Warning: no weights loaded.")

    # data
    ds_test = instantiate(cfg.ds_test)
    dl_test = utils.dataloader.build_dl_from_config(fabric=fabric, cfg=cfg, ds=ds_test)

    # acceleration
    model = fabric.setup_module(model)
    dl_test = fabric.setup_dataloaders(dl_test)

    # evaluation
    metrics = engine.validate(fabric=fabric, model=model, dl=dl_test)

    # log
    if fabric.is_global_zero:
        print(utils.format.format_metrics(metrics))
    return metrics


if __name__ == "__main__":
    test()
