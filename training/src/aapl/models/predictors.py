from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from functools import partial
from itertools import product
from typing import Type

import numpy as np
import torch

from aapl.config.schema import Config
from aapl.utils.proposals import (
    FuserOut,
    calculate_oic_scores,
    mask_to_segments,
    soft_nms,
)
from aapl.utils.temporal_sampling import linear_upsampling

from .aapl_model import Predictions, PredictorBase


class OICPredictor(PredictorBase):
    def __init__(
        self,
        proposal_thresholds: float | Iterable[float],
        video_threshold: float,
        proposal_scorer: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        proposal_fuser: Callable[[np.ndarray, np.ndarray], FuserOut],
        upsampler: Callable[[np.ndarray], np.ndarray] | None = None,
        upsampling_rate: float | None = None,
        zero_fill: bool = False,
        oic_offset: float = 0.0,
    ) -> None:
        if upsampler:
            assert upsampling_rate is not None
        if not isinstance(proposal_thresholds, Iterable):
            proposal_thresholds = [proposal_thresholds]
        self.proposal_threshold = proposal_thresholds
        self.video_threshold = video_threshold
        self.proposal_scorer = proposal_scorer
        self.proposal_fuser = proposal_fuser
        self.upsampler = upsampler
        self.upsampling_rate = upsampling_rate
        self.oic_offset = oic_offset

        self.zero_fill = zero_fill

    @classmethod
    def from_cfg(cls: Type[OICPredictor], cfg: Config) -> OICPredictor:
        pred_cfg = cfg.model.predictor

        assert pred_cfg.proposal_thresholds is not None
        thresholds = pred_cfg.proposal_thresholds

        scorer, fuser = cls._get_scorer_and_fuser(cfg)
        assert pred_cfg.video_cls_threshold is not None
        if pred_cfg.upsampling_rate:
            upsampler: Callable | None = partial(
                linear_upsampling,
                scale_factor=pred_cfg.upsampling_rate,
                align=pred_cfg.aligned_upsampling,
            )
        else:
            upsampler = None

        assert pred_cfg.zero_fill_below_thresh is not None

        return cls(
            proposal_thresholds=thresholds,
            video_threshold=pred_cfg.video_cls_threshold,
            proposal_scorer=scorer,
            proposal_fuser=fuser,
            upsampler=upsampler,
            upsampling_rate=pred_cfg.upsampling_rate,
            zero_fill=pred_cfg.zero_fill_below_thresh,
            oic_offset=pred_cfg.oic_offset,
        )

    @staticmethod
    def _get_scorer_and_fuser(
        cfg: Config,
    ) -> tuple[
        Callable[[np.ndarray, np.ndarray, float], np.ndarray],
        Callable[[np.ndarray, np.ndarray], FuserOut],
    ]:
        pred_cfg = cfg.model.predictor

        iou_threshold = pred_cfg.fuser_iou_threshold
        outer_margin = pred_cfg.oic_outer_margin
        assert iou_threshold is not None
        assert outer_margin is not None
        assert pred_cfg.fuser_type is not None

        def scorer(
            _scores: np.ndarray, _segments: np.ndarray, _offset: float
        ) -> np.ndarray:
            return calculate_oic_scores(
                _scores,
                _segments,
                outer_margin=outer_margin,
                return_inner=False,
                offset=_offset,
            )

        fuser = partial(
            soft_nms, sigma=pred_cfg.fuser_sigma, iou_threshold=iou_threshold
        )
        return scorer, fuser

    @torch.no_grad()
    def __call__(
        self, cas: np.ndarray, video_fg_scores: np.ndarray, t_stride: float
    ) -> Predictions:
        # cas: [T, C+1]
        # video_fg_scores: [C]
        if video_fg_scores.max() >= self.video_threshold:
            pred_video_cls = np.nonzero(video_fg_scores >= self.video_threshold)[0]
        else:
            pred_video_cls = np.array([video_fg_scores.argmax()])

        if self.upsampler and self.upsampling_rate is not None:
            cas = self.upsampler(cas)
            t_stride = t_stride / self.upsampling_rate

        proposal_dict: dict[int, list] = defaultdict(list)
        for threshold, video_cls_np in product(self.proposal_threshold, pred_video_cls):
            video_cls = video_cls_np.item()
            cas_one_class = cas[:, video_cls].copy()
            if self.zero_fill:
                cas_one_class[cas_one_class < threshold] = 0.0

            out = self._generate_scored_proposals(
                cas_one_class,
                threshold,
                offset=self.oic_offset * video_fg_scores[video_cls],
            )
            if out is None:
                continue
            proposal_dict[video_cls].append(out)

        labels, segments, scores = self._fuse_and_concat_proposals(proposal_dict)

        return Predictions(labels, segments * t_stride, scores)

    def _generate_scored_proposals(
        self,
        seg_criterion: np.ndarray,  # [T]
        threshold: float,
        score_sequence: np.ndarray | None = None,
        offset: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        score_sequence = seg_criterion if score_sequence is None else score_sequence

        # generate raw proposals by thresholding `seg_criterion`
        above_thresh = seg_criterion >= threshold
        segments = mask_to_segments(above_thresh)  # [N, 2], list of start and end idx
        segments = segments[segments[:, 1] - segments[:, 0] > 1]
        if len(segments) == 0:
            return None

        # oic_scores: np.ndarray = calculate_oic_scores(
        scores = self.proposal_scorer(score_sequence, segments, offset)
        return segments, scores

    def _fuse_and_concat_proposals(self, proposal_dict: dict[int, list]) -> Predictions:
        proposals_all = []
        for video_cls in proposal_dict:
            segments_cls, scores_cls = [
                np.concatenate(arrays, axis=0)
                for arrays in zip(*(proposal_dict[video_cls]))
            ]
            segs_fused, scores_fused, kept_idx = self.proposal_fuser(
                segments_cls, scores_cls
            )
            if len(kept_idx) == 0:
                continue
            labels_cls = np.full_like(kept_idx, video_cls)
            proposals_all.append((labels_cls, segs_fused, scores_fused))

        if len(proposals_all) == 0:
            return Predictions(np.zeros(0), np.zeros((0, 2)), np.zeros(0))
        return Predictions(
            *[np.concatenate(arrays, axis=0) for arrays in zip(*proposals_all)]
        )
