# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from tempfile import TemporaryDirectory

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from blueprint import engine, utils


@hydra.main(version_base=None, config_path="../configs")
@utils.oom.handle_oom()
def train(cfg: DictConfig) -> float:
    """Train a configuration."""
    # init
    utils.torch.initialize_torch(detect_anomaly=cfg.detect_anomaly)
    fabric = utils.fabric.initialize_fabric(cfg.seed, precision=cfg.precision)
    cfg = utils.config.initialize_config(cfg)

    cfg = utils.config.add_git_commit_hash(cfg)
    cfg = utils.config.add_eff_batch_size(cfg, world_size=fabric.world_size)
    if fabric.is_global_zero:
        print(OmegaConf.to_yaml(cfg, sort_keys=True))

    # data
    ds_train = instantiate(cfg.ds_train)
    dl_train = utils.dataloader.build_dl_from_config(
        fabric=fabric, cfg=cfg, ds=ds_train
    )
    ds_val = instantiate(cfg.ds_val)
    dl_val = utils.dataloader.build_dl_from_config(fabric=fabric, cfg=cfg, ds=ds_val)
    cfg = utils.config.add_total_steps(cfg, step_per_epoch=len(ds_train))

    # model
    with fabric.init_module(empty_init="ckpt_path" in cfg):
        model = instantiate(cfg.model)
        training_module = instantiate(cfg.trainer, model=model)
        opt = instantiate(cfg.optimizer, params=training_module.parameters())
        scheduler = instantiate(cfg.scheduler, optimizer=opt)
    if fabric.is_global_zero:
        utils.model.present_model(model)
    if cfg.compile:
        training_module.compile()
        if fabric.is_global_zero:
            print("Compilation enabled.")

    epoch, step = 0, 0
    if "ckpt_path" in cfg:
        epoch, step = utils.checkpoint.load_training(
            fabric=fabric,
            ckpt_path=cfg.ckpt_path,
            optimizer=opt,
            scheduler=scheduler,
            training_module=training_module,
        )
        epoch += 1
        if fabric.is_global_zero:
            print(f"Training checkpoint loaded from {cfg.ckpt_path}")
    elif "weights_path" in cfg:
        utils.checkpoint.load_weights(
            fabric=fabric, ckpt_path=cfg.weights_path, model=model
        )
        if fabric.is_global_zero:
            print(f"Weights loaded from {cfg.weights_path}")

    # logs
    if "log_dir" not in cfg:
        log_dir = utils.logs.get_log_dir(cfg.name) if fabric.is_global_zero else None
        cfg.log_dir = fabric.broadcast(log_dir, src=0)
    if cfg.log_dir == "/tmp":
        log_dir = None
        if fabric.is_global_zero:
            tempdir = TemporaryDirectory()
            log_dir = tempdir.name
        cfg.log_dir = fabric.broadcast(log_dir, src=0)
    elif os.path.exists(cfg.log_dir):
        raise ValueError(f"log_dir '{cfg.log_dir}' exists.")
    if fabric.is_global_zero:
        print(f"Log directory is {cfg.log_dir}")

    # acceleration
    training_module, opt = fabric.setup(training_module, opt)
    dl_train, dl_val = fabric.setup_dataloaders(dl_train, dl_val)

    # train
    best_metric = engine.train(
        fabric=fabric,
        opt=opt,
        scheduler=scheduler,
        model=model,
        training_module=training_module,
        dl_train=dl_train,
        dl_val=dl_val,
        log_dir=cfg.log_dir,
        cfg_str=utils.config.dump_config(cfg),
        begin_epoch=epoch,
        begin_step=step,
        n_epochs=cfg.n_epochs,
        n_accum_steps=cfg.n_accum_steps,
        patience=cfg.patience,
        watched_metric=cfg.watched_metric,
        enable_divergence_detection=cfg.divergence_detection,
        enable_profiling=cfg.profiling,
    )
    return best_metric


if __name__ == "__main__":
    train()
