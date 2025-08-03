from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from typing import Type

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from aapl.config.schema import Config
from aapl.datasets.feature_loaders import get_feature_loader, npy_loader
from aapl.utils.anet.io import load_anet_ground_truth

from .structures import SparseSupInstance, VideoInstance, WeakSupInstance
from .temporal_samplers import get_temporal_sampler


class VideoDataset(Dataset):
    def __init__(
        self,
        video_metadata: Sequence[tuple[str, float, float, int, str]],
        action_classes: Sequence[str],
        video_labels: Sequence[Tensor] | Tensor | None = None,
        sparse_labels: Sequence[Iterable[tuple[int, float]]] | None = None,
        feature_loader: Callable[[str], Tensor] = npy_loader,
        transform: Callable[[VideoInstance], VideoInstance] | None = None,
    ) -> None:
        assert video_labels is None or len(video_labels) == len(video_metadata)
        assert (
            video_labels is None
            or sparse_labels is None
            or len(video_labels) == len(sparse_labels)
        )
        super().__init__()

        self._video_metadata = video_metadata
        self._video_labels = video_labels
        self._sparse_labels = sparse_labels

        self.action_classes = action_classes
        self.class_name_to_index_dict = {
            name: idx for idx, name in enumerate(action_classes)
        }
        self.class_name_to_index_dict["background"] = len(action_classes)
        self.num_action_classes = len(action_classes)

        self._feature_loader = feature_loader
        self._transform = transform

    def _get_video_instance(self, index: int) -> VideoInstance:
        video_id, duration, frame_rate, stride, path = self._video_metadata[index]
        input_feature = self._feature_loader(path)
        return {
            "video_id": video_id,
            "index": index,
            "orig_length": input_feature.size(0),
            "duration": duration,
            "frame_rate": frame_rate,
            "stride": stride,
            "input": input_feature,
        }

    def _get_video_labels(self, video_inst: VideoInstance) -> WeakSupInstance:
        assert self._video_labels is not None
        video_labels = self._video_labels[video_inst["index"]]
        return {**video_inst, "video_labels": video_labels}

    def _get_sparse_labels(self, video_labeled: VideoInstance) -> SparseSupInstance:
        assert self._sparse_labels

        sparse_label_list = self._sparse_labels[video_labeled["index"]]
        sparse_labels = torch.zeros(
            (video_labeled["input"].size(0), self.num_action_classes + 1)
        )
        sec_to_idx = video_labeled["frame_rate"] / video_labeled["stride"]
        for class_idx, timestamp in sparse_label_list:
            try:
                sparse_labels[round(timestamp * sec_to_idx), class_idx] = 1
            except IndexError:
                pass

        fg_t_indices = sparse_labels[:, :-1].any(dim=1).nonzero(as_tuple=True)[0]
        fg_labels = sparse_labels[fg_t_indices, :-1]
        bg_t_indices = sparse_labels[:, -1].nonzero(as_tuple=True)[0]

        if "video_labels" not in video_labeled:
            if len(fg_t_indices) > 0:
                video_labels = fg_labels.max(dim=0)[0]
            else:
                video_labels = fg_labels.new_zeros((self.num_action_classes,))
            video_wl: WeakSupInstance = {**video_labeled, "video_labels": video_labels}
        else:
            video_wl = video_labeled  # type: ignore

        return {
            **video_wl,
            "fg_labels": fg_labels,
            "fg_t_indices": fg_t_indices,
            "bg_t_indices": bg_t_indices,
        }

    @torch.no_grad()
    def __getitem__(self, index: int) -> VideoInstance:
        video_instance = self._get_video_instance(index)
        if self._video_labels is not None:
            video_instance = self._get_video_labels(video_instance)
        if self._sparse_labels:
            video_instance = self._get_sparse_labels(video_instance)

        if self._transform:
            video_instance = self._transform(video_instance)

        return video_instance

    @classmethod
    def from_cfg(
        cls: Type[VideoDataset],
        cfg: Config,
        split: str,
        split_sampler: str | None = None,
    ) -> VideoDataset:
        if split == "training":
            subset = cfg.dataset.training_subset
        elif split == "validation":
            subset = cfg.dataset.validation_subset
        else:
            raise ValueError(f"split name: {split}")
        assert subset
        video_labels, metadata, action_classes = _parse_anet_groundtruth(cfg, subset)

        if not cfg.dataset.with_video_labels:
            video_labels = None  # type: ignore

        split_sampler = split_sampler if split_sampler else split
        temporal_sampler = get_temporal_sampler(cfg, split_sampler)

        if cfg.dataset.with_sparse_labels:
            assert cfg.dataset.sparse_label_file
            sparse_labels = _load_sparse_labels(cfg, metadata)
        else:
            sparse_labels = None

        feature_loader = get_feature_loader(cfg.dataset.feature_type, cfg)

        return cls(
            video_labels=video_labels,
            video_metadata=metadata,
            action_classes=action_classes,
            transform=temporal_sampler,
            sparse_labels=sparse_labels,
            feature_loader=feature_loader,
        )

    def __len__(self) -> int:
        return len(self._video_metadata)


def _parse_anet_groundtruth(
    cfg: Config, subset: str
) -> tuple[Tensor, list[tuple[str, float, float, int, str]], list[str]]:
    data_cfg = cfg.dataset
    gt_path = os.path.join(data_cfg.dataset_root, data_cfg.ground_truth_file)

    if data_cfg.excluded_video_list:
        excluded_list_path = os.path.join(
            data_cfg.dataset_root, data_cfg.excluded_video_list
        )
        with open(excluded_list_path, "r") as fp:
            excluded_videos = [line.strip() for line in fp]
    else:
        excluded_videos = None
    # load anet ground-truth file
    full_gts_df, video_metadata_df = load_anet_ground_truth(gt_path, excluded_videos)
    video_metadata_df = video_metadata_df[video_metadata_df["subset"] == subset]

    name_to_idx_dict = {
        name: idx for idx, name in enumerate(data_cfg.action_class_names)
    }
    full_gts_df["class_index"] = (
        full_gts_df["label"].map(name_to_idx_dict).astype("int")
    )
    full_gt_grp = full_gts_df.groupby(by="video_id")

    # Variables to return
    video_labels = torch.zeros(
        (len(video_metadata_df), len(data_cfg.action_class_names))
    )
    parsed_metadata = []

    # iterate over videos in the subset
    for idx, video_meta in enumerate(video_metadata_df.itertuples(index=False)):
        video_gt = full_gt_grp.get_group(video_meta.video_id)
        video_label_set = set(video_gt["class_index"])
        for class_idx in video_label_set:
            video_labels[idx, class_idx] = 1

        # Frame rate given in cfg takes precedence because for feature extraction,
        # one may use a fixed frame rate different from the original data
        if data_cfg.frame_rate:
            frame_rate = data_cfg.frame_rate
        elif "frame_rate" in video_meta._fields:
            frame_rate = video_meta.frame_rate
        elif "fps" in video_meta._fields:
            frame_rate = video_meta.fps
        else:
            raise ValueError("Frame rate not found in metadata")

        if "duration" in video_meta._fields:
            duration = video_meta.duration
        else:
            duration = 0.0
        parsed_metadata.append(
            (
                video_meta.video_id,
                duration,
                frame_rate,
                data_cfg.snippet_stride,
                data_cfg.feature_path_format.format(
                    root=data_cfg.dataset_root,
                    subset=subset,
                    video_id=video_meta.video_id,
                ),
            )
        )
    return video_labels, parsed_metadata, data_cfg.action_class_names


def _load_sparse_labels(
    cfg: Config, metadata: list[tuple[str, float, float, int, str]]
) -> list[list[tuple[int, float]]]:
    assert cfg.dataset.sparse_label_file
    sparse_labels_df = pd.read_csv(
        os.path.join(cfg.dataset.dataset_root, cfg.dataset.sparse_label_file)
    )
    name_to_idx_dict = {
        name: idx for idx, name in enumerate(cfg.dataset.action_class_names)
    }
    name_to_idx_dict["background"] = len(name_to_idx_dict)
    sparse_labels_df["class_index"] = (
        sparse_labels_df["action_label"].map(name_to_idx_dict).astype(int)
    )
    sparse_label_grp = sparse_labels_df.groupby(by="video_id")

    sparse_label_iters = []
    for video_id, *_ in metadata:
        try:
            group = sparse_label_grp.get_group(video_id)
            sparse_label_iters.append(
                group[["class_index", "timestamp"]].itertuples(index=False)
            )
        except KeyError:
            sparse_label_iters.append([])

    sparse_label_data = [
        [(class_idx, timestamp) for class_idx, timestamp in iterable]
        for iterable in sparse_label_iters
    ]
    return sparse_label_data
