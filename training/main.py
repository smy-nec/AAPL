import os

import torch
from dotenv import load_dotenv
from ignite.utils import manual_seed
from omegaconf import OmegaConf

import aapl.utils.mlflow as mlflow
from aapl.config.io import load_cfg
from aapl.config.schema import Config
from aapl.datasets import get_data_loaders
from aapl.engines import create_trainer_and_evaluator
from aapl.models import build_model
from aapl.optim import get_optimizer
from aapl.utils.logging import setup_logging
from aapl.utils.misc import prepare_directory


def run(cfg: Config) -> None:
    """Run a single training run.

    Args:
        cfg: Config object.
    """
    logger = setup_logging("single_run")

    logger.info("Starting training.")
    logger.info(OmegaConf.to_yaml(cfg))

    manual_seed(cfg.rng_seed)

    # Prepare data loaders
    logger.info("Single-run mode")
    data_loaders = get_data_loaders(cfg)
    logger.info(repr(data_loaders))

    # Prepare a trainer and evaluator
    model = build_model(cfg)
    logger.info(repr(model))

    optimizer = get_optimizer(cfg, model.parameters())
    logger.info(repr(optimizer))

    trainer, evaluator = create_trainer_and_evaluator(
        cfg, model, optimizer, data_loaders["validation"]
    )

    if cfg.checkpoint_path is not None:
        checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.load_state_dict(checkpoint["trainer"])

    # Run training for `cfg.num_epochs` epochs
    model.cuda()

    with mlflow.start_run(run_name="run", cfg=cfg):
        mlflow.log_dict(OmegaConf.to_container(cfg), "cfg.yaml")
        trainer.run(data_loaders["training"], max_epochs=cfg.num_epochs)


def main() -> None:
    cfg = load_cfg()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    manual_seed(cfg.rng_seed)

    prepare_directory(cfg)
    setup_logging(
        with_stream_handler=True,
        filepath=os.path.join(cfg.output_dir, "log.out"),
        reset=True,
    )
    logger = setup_logging("main")
    logger.info("Starting WSTADAC's main script")
    OmegaConf.save(cfg, os.path.join(cfg.output_dir, "cfg.yaml"))
    mlflow.set_experiment(experiment_name=cfg.experiment_name)
    run(cfg)


if __name__ == "__main__":
    load_dotenv()
    main()
