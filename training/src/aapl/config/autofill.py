from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .schema import Config


def autofill(cfg: Config) -> None:
    pred_cfg = cfg.model.predictor
    if pred_cfg.proposal_threshold_range:
        assert pred_cfg.proposal_thresholds is None
        pred_cfg.proposal_thresholds = np.arange(
            *(pred_cfg.proposal_threshold_range)
        ).tolist()

    infer_cfg = cfg.inference
    if infer_cfg.tiou_threshold_range:
        assert infer_cfg.tiou_thresholds is None
        infer_cfg.tiou_thresholds = np.arange(
            *(infer_cfg.tiou_threshold_range)
        ).tolist()
