from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from ignite.engine import Engine, Events
from ignite.handlers import EpochOutputStore
from torch import Tensor
from torch.optim import Optimizer

from aapl.config.schema import Config
from aapl.engines.common_utils import calculate_average_partial_losses, to_device
from aapl.utils.logging import setup_logging

if TYPE_CHECKING:
    from aapl.datasets.structures import WeakSupBatch
    from aapl.models.aapl_model import AAPLModel


def _train_step(
    trainer: Engine,
    instance: WeakSupBatch,
    model: AAPLModel,
    optimizer: Optimizer,
    batch_size: int,
) -> dict[str, Tensor]:
    model.train()
    instance = to_device(instance, non_blocking=True)

    head_out = model(instance, predict_segments=False)
    loss, loss_dict = model.calculate_loss(head_out, instance)
    loss.backward()  # type: ignore[no-untyped-call]

    optimizer.step()
    optimizer.zero_grad()

    return {"loss": loss.detach(), **loss_dict}


def Trainer(cfg: Config, model: AAPLModel, optimizer: Optimizer) -> Engine:
    trainer = Engine(
        partial(
            _train_step,
            model=model,
            optimizer=optimizer,
            batch_size=cfg.optim.batch_size,
        )
    )
    trainer.logger = setup_logging("trainer")

    loss_dict_store = EpochOutputStore()
    loss_dict_store.attach(trainer, "loss_dict_store")
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, partial(calculate_average_partial_losses)
    )

    return trainer
