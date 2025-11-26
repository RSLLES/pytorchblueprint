# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from blueprint import engine, utils


@hydra.main(version_base=None, config_path="../configs")
@torch.no_grad()
def multitest(cfg: DictConfig):
    """Perform an evaluation pass over the test dataset for multiple sets of weights."""
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

    # data
    ds_test = instantiate(cfg.ds_test)
    dl_test = utils.dataloader.build_dl_from_config(fabric=fabric, cfg=cfg, ds=ds_test)

    # acceleration
    model = fabric.setup_module(model)
    dl_test = fabric.setup_dataloaders(dl_test)

    # multi-evaluation
    metrics = {}
    for weights_path in weights_paths:
        utils.checkpoint.load_weights(
            fabric=fabric, ckpt_path=weights_path, model=model
        )
        if fabric.is_global_zero:
            print(f"Weights loaded from {weights_path}")
        run_metrics = engine.validate(fabric=fabric, model=model, dl=dl_test)
        if fabric.is_global_zero:
            print(utils.format.format_metrics(run_metrics))
        for key, value in run_metrics.items():
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)

    metrics.pop("offset")  # tensor would break formatting
    metrics.pop("wobble_corr")
    if fabric.is_global_zero:
        print("Aggregated results:")
        print(utils.format.format_metrics_with_uncertainties(metrics))


if __name__ == "__main__":
    multitest()
