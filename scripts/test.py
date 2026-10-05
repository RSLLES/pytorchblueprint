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
    cfgr = cfg.runtime
    if "weights_path" not in cfgr:
        raise ValueError("Provide runtime.weights_path to test.")
    utils.torch.initialize_torch(detect_anomaly=cfgr.detect_anomaly)
    fabric = utils.fabric.initialize_fabric(
        cfgr.seed, precision=cfgr.get("precision", "32-true"), devices=cfgr.devices
    )
    cfg = utils.config.initialize_config(cfg)

    if fabric.is_global_zero:
        print(OmegaConf.to_yaml(cfg))

    # model: checkpoints hold the whole training module
    with fabric.init_module(empty_init=cfgr.strict_loading):
        model = instantiate(cfg.model)
        training_module = instantiate(cfg.trainer, model=model)
    if fabric.is_global_zero:
        utils.model.present_model(model)

    epoch, step = utils.checkpoint.load_training(
        fabric=fabric,
        ckpt_path=cfgr.weights_path,
        model=training_module,
        optimizer=None,
        scheduler=None,
        strict=cfgr.strict_loading,
    )
    if fabric.is_global_zero:
        print(f"Weights loaded from {cfgr.weights_path}: epoch {epoch} step {step}.")

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

    # evaluation
    evaluators = [engine.WassersteinEvaluator(), engine.ScatterPlotEvaluator()]
    metrics = engine.validate(
        fabric=fabric, model=model, dl=dl_test, evaluators=evaluators
    )

    # log
    if fabric.is_global_zero:
        print(utils.format.format_metrics(metrics))
    return metrics


if __name__ == "__main__":
    test()
