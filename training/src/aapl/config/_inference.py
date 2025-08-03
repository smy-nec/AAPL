from dataclasses import dataclass


@dataclass
class InferenceCfg:
    tiou_thresholds: list[float] | None = None
    tiou_threshold_range: tuple[float, float, float] | None = None

    average_mAP_over: list[list[int]] | None = None
    primary_metric_key: str | None = None

    cleanup_predictions: bool = True
