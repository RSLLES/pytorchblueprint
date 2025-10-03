# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from blueprint import engine, utils


@hydra.main(version_base=None, config_path="../config")
@torch.no_grad()
def eval(cfg: DictConfig):
    utils.torch.initialize_torch()
    fabric = utils.fabric.initialize_fabric(cfg.seed)
    cfg = utils.config.initialize_config(cfg)

    if fabric.is_global_zero:
        print(OmegaConf.to_yaml(cfg))

    # weights
    weights_paths = glob.glob(cfg.weights_path, recursive=True)
    if len(weights_paths) == 0:
        raise ValueError(f"Did not find any file matching pattern '{cfg.weights_path}'")
    weights_paths.sort()
    if fabric.is_global_zero:
        print("Discovered weights:")
        for weights_path in weights_paths:
            print(f"- {weights_path}")

    # model
    with fabric.init_module(empty_init=True in cfg):
        model = instantiate(cfg.model)
    if fabric.is_global_zero:
        utils.model.present_model(model)
    model_needs_calib = hasattr(model, "apply_thresholds")

    # data
    ds_test = instantiate(cfg.ds_test)
    dl_test = utils.dataloader.build_dl_from_config(fabric=fabric, cfg=cfg, ds=ds_test)
    if model_needs_calib:
        ds_calib = instantiate(cfg.ds_calib)
        dl_calib = utils.dataloader.build_dl_from_config(
            fabric=fabric, cfg=cfg, ds=ds_calib
        )

    # acceleration
    model = fabric.setup_module(model)
    if model_needs_calib:
        model.mark_forward_method("apply_thresholds")
        dl_calib, dl_test = fabric.setup_dataloaders(dl_calib, dl_test)
    else:
        dl_test = fabric.setup_dataloaders(dl_test)

    # multi-evaluation
    combined_metrics = {}
    for weights_path in weights_paths:
        # weights
        utils.checkpoint.load_weights(
            fabric=fabric, ckpt_path=weights_path, model=model
        )
        if fabric.is_global_zero:
            print(f"Weights loaded from {weights_path}")

        # calibration
        if model_needs_calib:
            thresholds = engine.calibration_loop(
                fabric=fabric,
                model=model,
                dl=dl_calib,
                metric=cfg.watched_metric,
            )
            model.set_thresholds(thresholds)
            ds_calib.increment_seed()

        # evaluation
        metrics = engine.validation_loop(
            fabric=fabric, model=model, dl=dl_test, wooblecorr=True
        )
        if fabric.is_global_zero:
            print(utils.strings.format_metrics(metrics))
        for key, value in metrics.items():
            if key not in combined_metrics:
                combined_metrics[key] = []
            combined_metrics[key].append(value)

    # log
    combined_metrics.pop("offset")  # tensor would break formatting
    combined_metrics.pop("wooblecorr")
    if fabric.is_global_zero:
        print("Aggregated results:")
        print(utils.format.metrics2string(**combined_metrics))


if __name__ == "__main__":
    eval()
