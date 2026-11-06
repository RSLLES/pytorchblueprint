# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from tempfile import TemporaryDirectory

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from blueprint import engine, utils


@hydra.main(version_base=None, config_path="../configs", config_name="momentmatching")
@utils.oom.handle_oom()
def train(cfg: DictConfig) -> float:
    """Train a configuration."""
    # init
    cfgr = cfg.runtime
    utils.torch.initialize_torch(detect_anomaly=cfgr.detect_anomaly)
    fabric = utils.fabric.initialize_fabric(
        cfgr.seed,
        precision=cfgr.precision,
        devices=cfgr.devices,
        find_unused_parameters=cfgr.find_unused_parameters,
    )
    cfg = utils.config.initialize_config(cfg)

    cfg = utils.config.add_eff_batch_size(cfg, world_size=fabric.world_size)
    cfgr.world_size = fabric.world_size
    if fabric.is_global_zero:
        print(OmegaConf.to_yaml(cfg, sort_keys=True))

    # data
    ds_train = instantiate(cfg.ds_train)
    dl_train = utils.dataloader.build_dl(
        ds_train,
        batch_size=cfgr.batch_size,
        n_workers=cfgr.n_workers,
        world_size=fabric.world_size,
        shuffle=False,
    )
    ds_val = instantiate(cfg.ds_val)
    dl_val = utils.dataloader.build_dl(
        ds_val,
        batch_size=cfgr.batch_size,
        n_workers=cfgr.n_workers,
        world_size=fabric.world_size,
        shuffle=False,
    )
    cfg = utils.config.add_total_steps(
        cfg, n_samples=len(ds_train), world_size=fabric.world_size
    )

    # model
    # a non-strict load leaves missing parameters as they were initialized
    load_all = ("ckpt_path" in cfgr or "weights_path" in cfgr) and cfgr.strict_loading
    with fabric.init_module(empty_init=load_all):
        model = instantiate(cfg.model)
        training_module = instantiate(cfg.trainer, model=model)
        optimizer = instantiate(cfg.optimizer, params=training_module.parameters())
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    if fabric.is_global_zero:
        utils.model.present_model(model)
    explain = cfgr.get("explain", False)
    if cfgr.compile and not explain:
        training_module.compile()
        if fabric.is_global_zero:
            print("Compilation enabled.")

    # acceleration
    training_module, optimizer = fabric.setup(training_module, optimizer)
    dl_train, dl_val = fabric.setup_dataloaders(dl_train, dl_val)
    if explain:
        utils.model.explain_model(training_module, dl_train, fabric=fabric)
        utils.fabric.exit_with_barrier(fabric)

    epoch, step = 0, 0
    if "ckpt_path" in cfgr:
        epoch, step = utils.checkpoint.load_training(
            fabric=fabric,
            ckpt_path=cfgr.ckpt_path,
            model=training_module,
            optimizer=optimizer,
            scheduler=scheduler,
            strict=cfgr.strict_loading,
        )
        epoch += 1
        if fabric.is_global_zero:
            print(f"Training checkpoint loaded from {cfgr.ckpt_path}")
    elif "weights_path" in cfgr:
        utils.checkpoint.load_training(
            fabric=fabric,
            ckpt_path=cfgr.weights_path,
            model=training_module,
            strict=cfgr.strict_loading,
        )
        if fabric.is_global_zero:
            print(f"Weights loaded from {cfgr.weights_path}")

    # logs; an OOM retry reruns this function, so the resolved path stays out of cfg
    log_dir = cfgr.get("log_dir")
    tempdir = None
    if log_dir is None:
        log_dir = utils.logs.get_log_dir(cfgr.name) if fabric.is_global_zero else None
        log_dir = fabric.broadcast(log_dir, src=0)
    elif log_dir == "/tmp":
        if fabric.is_global_zero:
            tempdir = TemporaryDirectory()
        log_dir = fabric.broadcast(tempdir.name if tempdir else None, src=0)
    elif os.path.exists(log_dir):
        raise ValueError(f"log_dir '{log_dir}' exists.")
    if fabric.is_global_zero:
        print(f"Log directory is {log_dir}")

    # training options
    if not cfgr.detect_divergence and fabric.is_global_zero:
        print("Warning: divergence detection is disabled.")

    # train
    best_metric = engine.train(
        fabric=fabric,
        optimizer=optimizer,
        scheduler=scheduler,
        model=model,
        training_module=training_module,
        dl_train=dl_train,
        dl_val=dl_val,
        log_dir=log_dir,
        cfg_str=OmegaConf.to_yaml(cfg, sort_keys=True),
        begin_epoch=epoch,
        begin_step=step,
        n_epochs=cfgr.n_epochs,
        total_steps=cfgr.total_steps,
        n_accum_steps=cfgr.n_accum_steps,
        patience=cfgr.patience,
        val_stride=cfgr.val_stride,
        watched_metric=cfgr.watched_metric,
        watched_metric_strategy=cfgr.watched_metric_strategy,
        detect_divergence=cfgr.detect_divergence,
        enable_profiling=cfgr.profiling,
        grad_clip_norm=cfgr.grad_clip_norm,
    )
    return best_metric


if __name__ == "__main__":
    train()
