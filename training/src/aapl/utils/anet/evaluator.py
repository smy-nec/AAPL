"""ActivityNet-style evaluator.
The code below was adapted from ActivityNet's official repo and optimized by SMY.

* Original repo: https://github.com/activitynet/ActivityNet/tree/ebc8aebb4018a758c4efb35bb7ab547d31b1eb1e
* Original license:
    ---
    Copyright (c) 2015 ActivityNet

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    ---
"""
import json
from collections.abc import Iterable
from os import PathLike

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.api.typing import DataFrameGroupBy


class ANETdetection:
    GROUND_TRUTH_FIELDS = ["database", "taxonomy", "version"]
    PREDICTION_FIELDS = ["results", "version", "external_data"]

    def __init__(
        self,
        ground_truth_filename: PathLike | str,
        prediction_filename: PathLike | str,
        ground_truth_fields: Iterable[str] = GROUND_TRUTH_FIELDS,
        prediction_fields: Iterable[str] = PREDICTION_FIELDS,
        tiou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),
        subset: str = "validation",
        n_jobs: int = 10,
        fast_computation: bool = True,
        verbose: bool = True,
        check_format: bool = False,
    ) -> None:
        if not ground_truth_filename:
            raise IOError("Please input a valid ground truth file.")
        if not prediction_filename:
            raise IOError("Please input a valid prediction file.")
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap: np.ndarray | None = None
        # Import ground truth and predictions.
        self.ground_truth, self.activity_index = self._import_ground_truth(
            ground_truth_filename, check_format
        )
        self.prediction = self._import_prediction(prediction_filename, check_format)
        if fast_computation:
            self.compute_average_precision_detection = (
                compute_average_precision_detection_fast
            )
        else:
            self.compute_average_precision_detection = (
                compute_average_precision_detection
            )

        if self.verbose:
            print("[INIT] Loaded annotations from {} subset.".format(subset))
            nr_gt = len(self.ground_truth)
            print("\tNumber of ground truth instances: {}".format(nr_gt))
            nr_pred = len(self.prediction)
            print("\tNumber of predictions: {}".format(nr_pred))
            print("\tFixed threshold for tiou score: {}".format(self.tiou_thresholds))

    def _import_ground_truth(
        self, ground_truth_filename: PathLike | str, check_format: bool
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, "r") as fobj:
            data = json.load(fobj)
        # Checking format
        if check_format:
            if not all([field in data.keys() for field in self.gt_fields]):
                raise IOError("Please input a valid ground truth file.")

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data["database"].items():
            if self.subset != v["subset"]:
                continue
            for ann in v["annotations"]:
                if ann["label"] not in activity_index:
                    activity_index[ann["label"]] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann["segment"][0]))
                t_end_lst.append(float(ann["segment"][1]))
                label_lst.append(activity_index[ann["label"]])

        ground_truth = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
            }
        )
        return ground_truth, activity_index

    def _import_prediction(
        self, prediction_filename: PathLike | str, check_format: bool
    ) -> pd.DataFrame:
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, "r") as fobj:
            data = json.load(fobj)
        # Checking format...
        if check_format:
            if not all([field in data.keys() for field in self.pred_fields]):
                raise IOError("Please input a valid prediction file.")

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid, v in data["results"].items():
            for result in v:
                label = self.activity_index[result["label"]]
                video_lst.append(videoid)
                t_start_lst.append(float(result["segment"][0]))
                t_end_lst.append(float(result["segment"][1]))
                label_lst.append(label)
                score_lst.append(result["score"])
        prediction = pd.DataFrame(
            {
                "video-id": video_lst,
                "t-start": t_start_lst,
                "t-end": t_end_lst,
                "label": label_lst,
                "score": score_lst,
            }
        )
        return prediction

    def _get_predictions_with_label(
        self, prediction_by_label: DataFrameGroupBy, label_name: str, cidx: int
    ) -> pd.DataFrame:
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def wrapper_compute_average_precision(self) -> np.ndarray:
        """Computes average precision for each class in the subset."""
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby("label")
        prediction_by_label = self.prediction.groupby("label")

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(
                    drop=True
                ),
                prediction=self._get_predictions_with_label(
                    prediction_by_label, label_name, cidx
                ),
                tiou_thresholds=self.tiou_thresholds,
            )
            for label_name, cidx in self.activity_index.items()
        )

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:, cidx] = results[i]

        return ap

    def evaluate(self) -> None:
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print("[RESULTS] Performance on ActivityNet detection task.")
            print("\tAverage-mAP: {}".format(self.average_mAP))


def compute_average_precision_detection(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    tiou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),
) -> np.ndarray:
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():
        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred["video-id"])
        except Exception:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(
            this_pred[["t-start", "t-end"]].values, this_gt[["t-start", "t-end"]].values
        )
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]["index"]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]["index"]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(
            precision_cumsum[tidx, :], recall_cumsum[tidx, :]
        )

    return ap


def compute_average_precision_detection_fast(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    tiou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),
) -> np.ndarray:
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction["score"].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby("video-id")
    prediction_gbvn = prediction.groupby("video-id")

    for video_id, pred_indices in prediction_gbvn.indices.items():
        try:
            ground_truth_videoid = ground_truth_gbvn.get_group(video_id)
        except Exception:
            fp[:, pred_indices] = 1
            continue
        this_gt = ground_truth_videoid.reset_index()
        gt_index = this_gt["index"]
        this_preds = prediction_gbvn.get_group(video_id)

        tiou_arr = segment_iou_batched(  # [N_pred, N_gt]
            this_preds[["t-start", "t-end"]].values,
            this_gt[["t-start", "t-end"]].values,
        )
        tiou_sorted_idx = tiou_arr.argsort(axis=1)[:, ::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for k, idx in enumerate(pred_indices):
                for jdx in tiou_sorted_idx[k]:
                    if tiou_arr[k, jdx] < tiou_thr:
                        fp[tidx, idx] = 1
                        break
                    if lock_gt[tidx, gt_index[jdx]] >= 0:
                        continue
                    tp[tidx, idx] = 1
                    lock_gt[tidx, gt_index[jdx]] = idx
                    break
                if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                    fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(
            precision_cumsum[tidx, :], recall_cumsum[tidx, :]
        )

    return ap


def interpolated_prec_rec(prec: np.ndarray, rec: np.ndarray) -> float:
    """Interpolated AP - VOCdevkit from VOC 2011."""
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap: float = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


def segment_iou(
    target_segment: np.ndarray, candidate_segments: np.ndarray
) -> np.ndarray:
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (
        (candidate_segments[:, 1] - candidate_segments[:, 0])
        + (target_segment[1] - target_segment[0])
        - segments_intersection
    )
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU: np.ndarray = segments_intersection.astype(float) / segments_union
    return tIoU


def segment_iou_batched(
    target_segments: np.ndarray, candidate_segments: np.ndarray
) -> np.ndarray:
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 2d array
        Temporal target segments containing N x [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 2d array
        Temporal intersection over union score of the N target segments against
        the N candidates.
    """
    tt1 = np.maximum(target_segments[:, None, 0], candidate_segments[None, :, 0])
    tt2 = np.minimum(target_segments[:, None, 1], candidate_segments[None, :, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (
        (candidate_segments[None, :, 1] - candidate_segments[None, :, 0])
        + (target_segments[:, None, 1] - target_segments[:, None, 0])
        - segments_intersection
    )
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU: np.ndarray = segments_intersection.astype(float) / segments_union
    return tIoU
