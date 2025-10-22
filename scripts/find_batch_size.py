# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tempfile import TemporaryDirectory

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from blueprint import engine, utils
from blueprint.datasets.utils import SplitDataset


@hydra.main(version_base=None, config_path="../configs")
@utils.oom.handle_oom()
def find_batch_size(cfg: DictConfig) -> float:
    """Find the biggest batch size that does not result in an OOM."""
    # init
    utils.torch.initialize_torch()
    fabric = utils.fabric.initialize_fabric(cfg.seed)
    cfg = utils.config.initialize_config(cfg)
    cfg.n_epochs = 2
    init_batch_size = cfg.batch_size

    # returns True for valid batch sizes, False otherwise
    def func(batch_size: int, cfg=cfg):
        torch.cuda.empty_cache()
        if fabric.is_global_zero:
            print(f"Testing for batch_size={batch_size}.")
        if batch_size <= 0:
            raise ValueError("batch_size can't be <= 0.")
        cfg.batch_size = batch_size
        cfg = utils.config.add_git_commit_hash(cfg)
        cfg = utils.config.add_eff_batch_size(cfg, world_size=fabric.world_size)
        if fabric.is_global_zero and batch_size == init_batch_size:
            print(OmegaConf.to_yaml(cfg, sort_keys=True))

        # data
        ds_train = instantiate(cfg.ds_train)
        if batch_size > len(ds_train):
            raise ValueError("batch_size is larger than the length of ds_train.")
        ds_train = SplitDataset(
            ds_train,
            prop=min(4 * batch_size / len(ds_train), 1.0),
            direction="forward",
        )
        dl_train = utils.dataloader.build_dl_from_config(
            fabric=fabric, cfg=cfg, ds=ds_train
        )

        ds_val = instantiate(cfg.ds_val)
        if batch_size > len(ds_val):
            raise ValueError("batch_size is larger than the length of ds_val.")
        ds_val = SplitDataset(
            ds_val,
            prop=min(4 * batch_size / len(ds_val), 1.0),
            direction="forward",
        )
        dl_val = utils.dataloader.build_dl_from_config(
            fabric=fabric, cfg=cfg, ds=ds_val
        )
        cfg = utils.config.add_total_steps(cfg, step_per_epoch=len(ds_train))

        # model
        with fabric.init_module(empty_init=False):
            model = instantiate(cfg.model)
            training_module = instantiate(cfg.trainer, model=model)
            opt = instantiate(cfg.optimizer, params=training_module.parameters())
            scheduler = instantiate(cfg.scheduler, optimizer=opt)
        if fabric.is_global_zero and batch_size == init_batch_size:
            utils.model.present_model(model)
        if cfg.compile:
            training_module.compile()
            if fabric.is_global_zero and batch_size == init_batch_size:
                print("Compilation enabled.")

        epoch, step = 0, 0

        # logs
        with utils.fabric.global_zero_context(
            lambda: TemporaryDirectory(), fabric=fabric
        ) as tempdir:
            cfg.log_dir = fabric.broadcast(tempdir, src=0)
            if fabric.is_global_zero and batch_size == init_batch_size:
                print(f"Log directory is {cfg.log_dir}")

            # acceleration
            training_module, opt = fabric.setup(training_module, opt)
            dl_train, dl_val = fabric.setup_dataloaders(dl_train, dl_val)

            try:
                engine.train(
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
                )
                return True
            except RuntimeError as e:
                if "out of memory" in str(e) or "OOM" in str(e):
                    return False
                raise e

    # search for best batch size
    batch_size = natural_bisect_unbounded(f=func, n=cfg.batch_size)
    print(f"Biggest batch size that does not raise an OOM error is {batch_size}.")


def natural_bisect_unbounded(f: callable, n: int, low: int = 0):
    """Unbounded version of natural_bisect.

    Expands the search space exponentially until f(n) is False, then applies
    natural_bisect to find the biggest n such that f(n) is True.
    """
    if not f(n):
        return natural_bisect(f, low=low, high=n)
    return natural_bisect_unbounded(f, low=n, n=2 * n)


def natural_bisect(f: callable, low: int, high: int):
    """Binary search over natural numbers; find biggest n such that f(n) is True."""
    if high - low <= 1:
        if low == 0:
            raise ValueError("Could not find a working batch size.")
        return low
    m = (low + high) // 2
    if f(m):
        return natural_bisect(f, low=m, high=high)
    return natural_bisect(f, low=low, high=m)


if __name__ == "__main__":
    find_batch_size()
