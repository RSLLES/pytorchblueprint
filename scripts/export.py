# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from blueprint import engine, utils


@hydra.main(version_base=None, config_path="../config")
@torch.no_grad()
def export(cfg: DictConfig):
    utils.torch.initialize_torch()
    fabric = utils.fabric.initialize_fabric(cfg.seed, devices=1)
    cfg = utils.config.initialize_config(cfg)

    if fabric.is_global_zero:
        print(OmegaConf.to_yaml(cfg))

    # writer
    writer = instantiate(cfg.writer)
    if fabric.is_global_zero:
        print(f"Output file will be written with {writer.__class__.__name__}.")

    # data
    ds_test = instantiate(cfg.ds_test)
    dl_test = utils.dataloader.build_dl_from_config(fabric=fabric, cfg=cfg, ds=ds_test)

    # model
    with fabric.init_module(empty_init=True):
        model = instantiate(cfg.model)
    utils.checkpoint.load_weights(
        fabric=fabric, ckpt_path=cfg.weights_path, model=model
    )
    if fabric.is_global_zero:
        utils.model.present_model(model)
        print(f"Weights loaded from {cfg.weights_path}")

    # data for calibration if needed
    model_needs_calib = hasattr(model, "apply_thresholds")
    if model_needs_calib:
        ds_calib = instantiate(cfg.ds_calib)
        dl_calib = utils.dataloader.build_dl_from_config(
            fabric=fabric, cfg=cfg, ds=ds_calib
        )
    else:
        dl_calib = None

    # acceleration
    model = fabric.setup_module(model)
    if model_needs_calib:
        model.mark_forward_method("apply_thresholds")
        dl_calib, dl_test = fabric.setup_dataloaders(dl_calib, dl_test)
    else:
        dl_test = fabric.setup_dataloaders(dl_test)

    # evaluation
    if model_needs_calib:
        thresholds = engine.calibration_loop(
            fabric=fabric, model=model, dl=dl_calib, metric=cfg.watched_metric
        )
        model.set_thresholds(thresholds)
    engine.export_loop(fabric=fabric, model=model, dl=dl_test, writer=writer)


if __name__ == "__main__":
    export()
