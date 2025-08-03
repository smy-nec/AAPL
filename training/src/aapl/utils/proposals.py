from typing import Literal, NamedTuple, overload

import numpy as np


def mask_to_segments(mask: np.ndarray) -> np.ndarray:
    assert mask.ndim == 1 and mask.dtype == np.bool_
    end_points = np.nonzero(np.diff(mask, prepend=False, append=False))[0]
    return end_points.reshape((-1, 2))


@overload
def calculate_oic_scores(
    scores: np.ndarray,
    segments: np.ndarray,
    outer_margin: float,
    return_inner: Literal[False] = False,
    offset: float | np.ndarray = 0.0,
) -> np.ndarray:
    ...


@overload
def calculate_oic_scores(
    scores: np.ndarray,
    segments: np.ndarray,
    outer_margin: float,
    return_inner: Literal[True],
    offset: float | np.ndarray = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    ...


def calculate_oic_scores(
    scores: np.ndarray,
    segments: np.ndarray,
    outer_margin: float,
    return_inner: bool = False,
    offset: float | np.ndarray = 0.0,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    len_sequence = len(scores)
    oic_scores = np.zeros(len(segments))
    if return_inner:
        inner_score_all = np.zeros(len(segments))

    inner_s, inner_e = segments[:, 0], segments[:, 1]
    outer = np.empty_like(segments, dtype=int)
    len_segment = inner_e - inner_s
    outer[:, 0] = np.maximum(0, (inner_s - len_segment * outer_margin).astype(int))
    outer[:, 1] = np.minimum(
        len_sequence, (inner_e + len_segment * outer_margin).astype(int)
    )
    for idx_seg, ((inner_s, inner_e), (outer_s, outer_e)) in enumerate(
        zip(segments, outer)
    ):
        outer_indices: np.ndarray = np.concatenate(
            (np.arange(outer_s, inner_s), np.arange(inner_e, outer_e))
        )
        if len(outer_indices) > 0:
            outer_score = scores[outer_indices].mean()
        else:
            outer_score = 0.0
        inner_score = scores[inner_s:inner_e].mean()
        oic_scores[idx_seg] = inner_score - outer_score + offset
        if return_inner:
            inner_score_all[idx_seg] = inner_score

    if return_inner:
        return oic_scores, inner_score_all
    return oic_scores


class FuserOut(NamedTuple):
    segments_fused: np.ndarray
    scores_fused: np.ndarray
    kept_indices: np.ndarray


def soft_nms(
    segments: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    sigma: float,
    method: str = "gaussian",
    threshold: float = 0.001,
) -> FuserOut:
    segments_copy, scores_copy = segments.copy(), scores.copy()
    kept_idx_np = np.arange(len(scores_copy))

    for i in range(len(segments_copy) - 1):
        if i >= len(scores_copy):
            break
        max_idx = np.argmax(scores_copy[i:]).item() + i

        segments_copy[[i, max_idx]] = segments_copy[[max_idx, i]]
        scores_copy[[i, max_idx]] = scores_copy[[max_idx, i]]
        kept_idx_np[[i, max_idx]] = kept_idx_np[[max_idx, i]]

        start, end = segments_copy[i:, 0], segments_copy[i:, 1]

        inter_s = np.maximum(start[0], start[1:])
        inter_e = np.minimum(end[0], end[1:])
        inter = np.clip(inter_e - inter_s + 1, a_min=0, a_max=None)
        union = (end[0] - start[0] + 1) + (end[1:] - start[1:] + 1) - inter

        iou = inter / union
        if method == "linear":
            scores_copy[i + 1 :] *= np.where(iou > iou_threshold, 1 - iou, 1)
        elif method == "gaussian":
            scores_copy[i + 1 :] *= np.exp(-(iou**2) / sigma)
        else:
            raise ValueError(f"Unknown method: {method}")

        retained = scores_copy[i + 1 :] > threshold

        idx_s, idx_e = i + 1, i + 1 + retained.sum()
        scores_copy[idx_s:idx_e] = scores_copy[i + 1 :][retained]
        segments_copy[idx_s:idx_e] = segments_copy[i + 1 :][retained]
        kept_idx_np[idx_s:idx_e] = kept_idx_np[i + 1 :][retained]

        scores_copy = scores_copy[:idx_e]
        segments_copy = segments_copy[:idx_e]
        kept_idx_np = kept_idx_np[:idx_e]
    return FuserOut(segments[kept_idx_np], scores_copy, kept_idx_np)
