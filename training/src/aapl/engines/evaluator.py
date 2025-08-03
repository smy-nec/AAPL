from __future__ import annotations

import json
import os
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from ignite.engine import Engine, Events
from ignite.handlers import EpochOutputStore
from ignite.metrics import Average
from torch import Tensor

from aapl.config.schema import Config
from aapl.utils.anet.evaluator import ANETdetection
from aapl.utils.anet.io import predictions_to_anet_format
from aapl.utils.logging import setup_logging

from .common_utils import calculate_average_partial_losses, to_device

if TYPE_CHECKING:
    from aapl.datasets.structures import WeakSupBatch
    from aapl.models.aapl_model import AAPLModel
    from aapl.models.predictors import Predictions

    _eval_step_out_t = tuple[WeakSupBatch, Predictions, Tensor, dict[str, Tensor]]


def _eval_step_video_label(
    _: Engine, instance: WeakSupBatch, model: AAPLModel
) -> _eval_step_out_t:
    model.eval()
    instance = to_device(instance, device="cuda", non_blocking=True)
    with torch.no_grad():
        pred_segments, head_out = model(instance, predict_segments=True)
        loss, loss_dict = model.calculate_loss(head_out, instance)
    return instance, pred_segments, loss, loss_dict


def _evaluate_detection(evaluator: Engine, cfg: Config, trainer: Engine | None) -> None:
    pred_store: list[tuple[str, list]] = evaluator.state.pred_store  # type: ignore
    results = {video_id: pred_anet for video_id, pred_anet in pred_store}

    ground_truth_file = os.path.join(
        cfg.dataset.dataset_root, cfg.dataset.ground_truth_file
    )
    if trainer:
        prediction_file = os.path.join(
            cfg.output_dir, f"prediction-{trainer.state.epoch:d}epoch.json"
        )
    else:
        prediction_file = os.path.join(cfg.output_dir, "predictions.json")
    with open(prediction_file, "w") as fp:
        json.dump({"version": "", "external_data": {}, "results": results}, fp)

    tiou_thresholds = np.array(cfg.inference.tiou_thresholds)
    subset = (
        "validation"
        if cfg.dataset.validation_subset is None
        else cfg.dataset.validation_subset
    )
    anet_evaluator = ANETdetection(
        ground_truth_file,
        prediction_file,
        tiou_thresholds=tiou_thresholds,
        subset=subset,
        verbose=True,
    )
    anet_evaluator.evaluate()

    for tiou, mAP in zip(tiou_thresholds, anet_evaluator.mAP):
        evaluator.state.metrics[f"mAP@{tiou:.2f}"] = mAP
    evaluator.state.metrics["Avg-mAP"] = anet_evaluator.average_mAP
    if cfg.inference.average_mAP_over:
        for tiou_inds in cfg.inference.average_mAP_over:
            avg_mAP = anet_evaluator.mAP[tiou_inds].mean()
            lower_iou = tiou_thresholds[tiou_inds[0]]
            upper_iou = tiou_thresholds[tiou_inds[-1]]
            evaluator.state.metrics[
                f"Avg-mAP[{lower_iou:.2f}:{upper_iou:.2f}]"
            ] = avg_mAP

    if cfg.inference.cleanup_predictions:
        os.remove(prediction_file)


def Evaluator(cfg: Config, trainer: Engine | None, model: AAPLModel) -> Engine:
    evaluator = Engine(partial(_eval_step_video_label, model=model))
    evaluator.logger = setup_logging("evaluator")

    def _store_anet_prediction(x: _eval_step_out_t) -> tuple[str, list]:
        video_id = x[0]["video_id"][0]
        pred_anet = predictions_to_anet_format(x[1], cfg.dataset.action_class_names)
        return (video_id, pred_anet)

    pred_store = EpochOutputStore(_store_anet_prediction)
    pred_store.attach(evaluator, "pred_store")
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, partial(_evaluate_detection, cfg=cfg, trainer=trainer)
    )

    loss_avg = Average(lambda x: x[2])
    loss_avg.attach(evaluator, "loss")
    loss_dict_store = EpochOutputStore(lambda x: x[-1])
    loss_dict_store.attach(evaluator, "loss_dict_store")
    evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, partial(calculate_average_partial_losses)
    )

    return evaluator
