from __future__ import annotations

import json
from collections.abc import Container
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from aapl.models.predictors import Predictions


def load_anet_ground_truth(
    path: str, excluded_videos: Container[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open(path, "r") as fp:
        data = json.load(fp)["database"]

    ground_truth: dict[str, list] = {
        "label": [],
        "t_start": [],
        "t_end": [],
        "video_id": [],
    }

    entry = next(iter(data.values()))
    metadata_keys = [key for key in entry if key != "annotations"]
    if "video_id" not in metadata_keys:
        metadata_keys = ["video_id"] + metadata_keys
    metadata: dict[str, list] = {key: [] for key in metadata_keys}

    if not excluded_videos:
        excluded_videos = []
    for video_id in sorted(data):
        if video_id in excluded_videos:
            continue
        video_entry = data[video_id]
        metadata["video_id"].append(video_id)
        for key, value in video_entry.items():
            if key in metadata:
                metadata[key].append(value)
        for anno_entry in video_entry["annotations"]:
            ground_truth["video_id"].append(video_id)
            ground_truth["label"].append(anno_entry["label"])
            t_start, t_end = anno_entry["segment"]
            ground_truth["t_start"].append(t_start)
            ground_truth["t_end"].append(t_end)

    return pd.DataFrame(ground_truth), pd.DataFrame(metadata)


def load_anet_predictions(path: str) -> pd.DataFrame:
    with open(path, "r") as fp:
        data = json.load(fp)["results"]

    results: dict[str, list] = {
        "label": [],
        "t_start": [],
        "t_end": [],
        "score": [],
        "video_id": [],
    }

    for video_id, video_entry in data.items():
        for result_entry in video_entry:
            results["video_id"].append(video_id)
            results["label"].append(result_entry["label"])
            results["score"].append(result_entry["score"])
            t_start, t_end = result_entry["segment"]
            results["t_start"].append(t_start)
            results["t_end"].append(t_end)

    return pd.DataFrame(results)


def predictions_to_anet_format(
    predictions: Predictions, action_classes: list[str]
) -> list[dict[str, Any]]:
    results = []
    for label, segment, score in zip(*predictions):
        results.append(
            {
                "label": action_classes[label.item()],
                "segment": segment.tolist(),
                "score": score.item(),
            }
        )
    return results
