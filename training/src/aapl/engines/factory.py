from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from aapl.engines.evaluator import Evaluator
from aapl.engines.trainer import Trainer
from aapl.utils import mlflow

if TYPE_CHECKING:
    from aapl.config.schema import Config
    from aapl.models.aapl_model import AAPLModel


def create_trainer_and_evaluator(
    cfg: Config, model: AAPLModel, optimizer: Optimizer, eval_data_loader: DataLoader
) -> tuple[Engine, Engine]:
    trainer = Trainer(cfg, model, optimizer)
    evaluator = Evaluator(cfg, trainer, model)

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.evaluate_every))
    def run_evaluator(_: Engine) -> None:
        evaluator.run(eval_data_loader)

    _setup_state_logging(cfg, trainer, evaluator, eval_data_loader)
    to_save = {"model": model, "optimizer": optimizer, "trainer": trainer}
    _setup_mlflow_logger(cfg, trainer, evaluator, model, optimizer)
    if cfg.logging.enable_checkpoints:
        _setup_checkpoint(cfg, trainer, evaluator, to_save)

    ProgressBar(desc="Train").attach(trainer, metric_names=["loss"])
    ProgressBar(desc="Eval").attach(evaluator, metric_names=["loss"])

    return trainer, evaluator


def create_evaluator(cfg: Config, model: AAPLModel) -> Engine:
    evaluator = Evaluator(cfg, None, model)

    @evaluator.on(Events.COMPLETED)
    def log_eval_metrics(_: Engine) -> None:
        state_log = [
            {
                "eval_metrics": {
                    key: value for key, value in evaluator.state.metrics.items()
                }
            }
        ]
        with open(os.path.join(cfg.output_dir, "state_log.json"), "w") as fp:
            json.dump(state_log, fp)
        evaluator.logger.info(
            "Eval metrics: {\n"
            + ",\n".join(
                [
                    f'\t"{key}": {value:.4f}'
                    for key, value in evaluator.state.metrics.items()
                ]
            )
            + "\n}"
        )

    ProgressBar(desc="Eval").attach(evaluator, metric_names=["loss"])

    return evaluator


def _setup_state_logging(
    cfg: Config, trainer: Engine, evaluator: Engine, eval_data_loader: DataLoader
) -> None:
    state_log = []

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_trainer_state(_: Engine) -> None:
        state_log.append(
            {
                "epoch": trainer.state.epoch,
                "iteraton": trainer.state.iteration,
                "epoch_length": trainer.state.epoch_length,
                "max_epochs": trainer.state.max_epochs,
                "metrics": {key: value for key, value in trainer.state.metrics.items()},
            }
        )
        trainer.logger.info(
            "Train metrics: {\n"
            + ",\n".join(
                [
                    f'\t"{key}": {value:.4f}'
                    for key, value in trainer.state.metrics.items()
                ]
            )
            + "\n}"
        )

    @trainer.on(Events.EPOCH_COMPLETED(every=cfg.evaluate_every))
    def log_evaluator_state(_: Engine) -> None:
        state_log[-1]["eval_metrics"] = {
            key: value for key, value in evaluator.state.metrics.items()
        }
        evaluator.logger.info(
            "Eval metrics: {\n"
            + ",\n".join(
                [
                    f'\t"{key}": {value:.4f}'
                    for key, value in evaluator.state.metrics.items()
                ]
            )
            + "\n}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_state_log(_: Engine) -> None:
        with open(os.path.join(cfg.output_dir, "state_log.json"), "w") as fp:
            json.dump(state_log, fp)


def _setup_checkpoint(
    cfg: Config, trainer: Engine, evaluator: Engine, to_save: dict
) -> None:
    chkpt_handler = ModelCheckpoint(
        os.path.join(cfg.output_dir, "chkpts"),
        score_name="Avg-mAP",
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(Events.COMPLETED, chkpt_handler, to_save)


def _setup_mlflow_logger(
    cfg: Config,
    trainer: Engine,
    evaluator: Engine,
    model: AAPLModel,
    optimizer: Optimizer,
) -> None:
    """Setup MLflow logging."""

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_metrics_mlflow(_: Engine) -> None:
        """MLflow logging helper function."""
        metrics = {
            f"training/{key}": value for key, value in trainer.state.metrics.items()
        }
        mlflow.log_metrics(metrics, step=trainer.state.epoch)

    @evaluator.on(Events.COMPLETED)
    def log_eval_metrics_mlflow(_: Engine) -> None:
        """MLflow logging helper function."""
        metrics = {
            f"validation/{key}": value for key, value in evaluator.state.metrics.items()
        }
        mlflow.log_metrics(metrics, step=trainer.state.epoch)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_lr_mlflow(_: Engine) -> None:
        """MLflow logging helper function."""
        lrs = {
            f"lr_{i}": float(params["lr"])
            for i, params in enumerate(optimizer.param_groups)
        }
        mlflow.log_metrics(lrs, step=trainer.state.epoch)
