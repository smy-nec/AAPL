from __future__ import annotations

import numpy as np

from aapl.config.schema import Config
from aapl.datasets.video_dataset import VideoDataset
from aapl.models.embedders import OneConvLayerEmbedder
from aapl.models.heads import AAPLHead
from aapl.models.predictors import OICPredictor

from .aapl_model import AAPLModel


def build_model(cfg: Config) -> AAPLModel:
    embedder = OneConvLayerEmbedder.from_cfg(cfg)
    head = AAPLHead.from_cfg(cfg)
    predictor = OICPredictor.from_cfg(cfg)

    def _aapl_head_out_postprocess(
        head_out: AAPLHead.HeadOut,
    ) -> tuple[np.ndarray, np.ndarray]:
        video_logits, _ = head.topk_pooling(
            head_out["cas_logits"], head_out["fused_cas"]
        )
        return (
            head_out["fused_cas"].mean(dim=2)[0, :].detach().cpu().numpy(),
            video_logits.sigmoid().mean(dim=1)[0].detach().cpu().numpy(),
        )

    model = AAPLModel(
        embedder, head, predictor, head_out_postprocess=_aapl_head_out_postprocess
    )
    model.head.init_with_train_data(  # type: ignore
        VideoDataset.from_cfg(cfg, "training", "validation"), model
    )
    return model
