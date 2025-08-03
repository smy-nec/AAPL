from dataclasses import dataclass

from omegaconf import II, MISSING


@dataclass
class EmbedderCfg:
    embedder_type: str = MISSING
    kernel_size: int | None = None
    groups: int = 1
    dropout_input: bool = False


@dataclass
class HeadCfg:
    head_type: str = MISSING
    topk_denoms: list[float] | None = None

    two_branch_embedder: bool = True
    kernel_size: int = 1
    fg_threshold: float = 0.5
    bg_threshold: float = 0.5
    video_loss_weight: float = 1.0
    positive_only_video_loss: bool = False
    contrastive_loss_weight: float = 0.0
    prototype_momentum: float = 0.1
    contrastive_temperature: float = 0.1
    contrastive_version: int = 2


@dataclass
class PredictorCfg:
    predictor_type: str = MISSING
    proposal_thresholds: list[float] | None = None
    proposal_threshold_range: tuple[float, float, float] | None = None
    video_cls_threshold: float | None = None
    zero_fill_below_thresh: bool | None = None

    # Upsampling
    upsampling_rate: float | None = None
    aligned_upsampling: bool = False

    # Outer-Inner Contrastive
    oic_outer_margin: float | None = None
    oic_offset: float = 0.0

    # Proposal fuser
    fuser_type: str | None = None
    fuser_iou_threshold: float | None = None
    fuser_temperature: float | None = None
    fuser_sigma: float | None = None


@dataclass
class ModelCfg:
    dim_input_features: int = MISSING
    dim_embedded_features: int = II(".dim_input_features")
    dropout_rate: float | None = None

    embedder: EmbedderCfg = EmbedderCfg()
    head: HeadCfg = HeadCfg()
    predictor: PredictorCfg = PredictorCfg()
