# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import wandb
import os
import torch

from pdearena import utils
from pdearena.data.datamodule import PDEDataModule
from pdearena.lr_scheduler import LinearWarmupCosineAnnealingLR  # noqa: F401
from pdearena.models.pdemodel import PDEModel

from pytorch_lightning.plugins.environments import SLURMEnvironment

# setting the precision of the computations, see https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
# torch.set_float32_matmul_precision(precision='medium')  # 'medium' or 'high' or 'highest'


logger = utils.get_logger(__name__)


def setupdir(path):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "tb"), exist_ok=True)
    os.makedirs(os.path.join(path, "ckpts"), exist_ok=True)




class DisabledSLURMEnvironment(SLURMEnvironment):
    """
    See https://github.com/Lightning-AI/lightning/issues/6389. 
    Fix to run PyTorchLightning on GPU.
    """
    def __init__(self):
        super().__init__(auto_requeue=False)

    def detect() -> bool:
        return False

    @staticmethod
    def _validate_srun_used() -> None:
        return

    @staticmethod
    def _validate_srun_variables() -> None:
        return


def main():
    # automatically done within PDECLI (the login). in fact logger is controlled above in l. 21!!!!!!!!!!!!!!!!!!!
    # login to wandb and create run
    # wandb.login(key='a9912be7ba0f93559201dd91894f52f7f3c2db8a')
    # wandb_run = wandb.init(project="multiresdiff", entity="diffmodels", mode='online')
    # if os.environ.get("LOCAL_RANK", None) is None:
    #     print("WANDB_RUN_DIR set.")
    #     os.environ["WANDB_RUN_DIR"] = wandb.run.dir

    cli = utils.PDECLI(
        PDEModel,
        PDEDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    if cli.trainer.default_root_dir is None:
        logger.warning("No default root dir set, using: ")
        cli.trainer.default_root_dir = os.environ.get("AMLT_OUTPUT_DIR", "./outputs")
        logger.warning(f"\t {cli.trainer.default_root_dir}")

    # set wandb environment variable
    # os.environ['WANDB_RUN_DIR'] = wandb.run.dir

    setupdir(cli.trainer.default_root_dir)
    logger.info(f"Checkpoints and logs will be saved in {cli.trainer.default_root_dir}")
    logger.info("Starting training...")
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    if not cli.trainer.fast_dev_run:
        logger.info("Starting testing...")
        cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
