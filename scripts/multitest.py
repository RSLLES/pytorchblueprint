# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from blueprint import engine, utils


@hydra.main(version_base=None, config_path="../configs")
@torch.no_grad()
def multitest(cfg: DictConfig):
    """Perform an evaluation pass over the test dataset for multiple sets of weights."""
    cfgr = cfg.runtime
    if "weights_path" not in cfgr:
        raise ValueError("Provide runtime.weights_path as a glob pattern to test.")
    utils.torch.initialize_torch(detect_anomaly=cfgr.detect_anomaly)
    fabric = utils.fabric.initialize_fabric(
        cfgr.seed, precision=cfgr.get("precision", "32-true"), devices=cfgr.devices
    )
    cfg = utils.config.initialize_config(cfg)

    if fabric.is_global_zero:
        print(OmegaConf.to_yaml(cfg))

    # weights
    weights_paths = glob.glob(cfgr.weights_path, recursive=True)
    if len(weights_paths) == 0:
        raise ValueError(
            f"Did not find any file matching pattern '{cfgr.weights_path}'"
        )
    weights_paths.sort()
    if fabric.is_global_zero:
        print("Discovered weights:")
        for weights_path in weights_paths:
            print(f"- {weights_path}")

    # model: checkpoints hold the whole training module
    with fabric.init_module(empty_init=cfgr.strict_loading):
        model = instantiate(cfg.model)
        training_module = instantiate(cfg.trainer, model=model)
    if fabric.is_global_zero:
        utils.model.present_model(model)

    # data
    ds_test = instantiate(cfg.ds_test)
    dl_test = utils.dataloader.build_dl(
        ds_test,
        batch_size=cfgr.batch_size,
        n_workers=cfgr.n_workers,
        world_size=fabric.world_size,
        shuffle=False,
    )

    # acceleration
    model = fabric.setup_module(model)
    dl_test = fabric.setup_dataloaders(dl_test)

    # multi-evaluation
    evaluators = [engine.WassersteinEvaluator(), engine.ScatterPlotEvaluator()]
    metrics = {}
    for weights_path in weights_paths:
        utils.checkpoint.load_training(
            fabric=fabric,
            ckpt_path=weights_path,
            model=training_module,
            optimizer=None,
            scheduler=None,
            strict=cfgr.strict_loading,
        )
        if fabric.is_global_zero:
            print(f"Weights loaded from {weights_path}")
        run_metrics = engine.validate(
            fabric=fabric, model=model, dl=dl_test, evaluators=evaluators
        )
        if fabric.is_global_zero:
            print(utils.format.format_metrics(run_metrics))
        for key, value in run_metrics.items():
            if not isinstance(value, Tensor) or value.numel() != 1:
                continue  # only scalars aggregate into a mean and a std
            if key not in metrics:
                metrics[key] = []
            metrics[key].append(value)

    if fabric.is_global_zero:
        print("Aggregated results:")
        print(utils.format.format_metrics_with_uncertainties(metrics))


if __name__ == "__main__":
    multitest()
