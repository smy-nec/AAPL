from typing import TypedDict

from torch import Tensor


class VideoInstance(TypedDict):
    video_id: str
    index: int
    orig_length: int  # temporal length before temporal sampling
    duration: float  # duration of the video in seconds
    frame_rate: float  # fps of the inputs to feature extractor
    stride: int  # num of frames in a snippet
    #
    input: Tensor  # [temporal length, dim features]


class WeakSupInstance(VideoInstance):
    video_labels: Tensor  # [num action classes]


class SparseSupInstance(WeakSupInstance):
    fg_labels: Tensor  # [num action labels, num action classes]
    fg_t_indices: Tensor  # [num action labels]
    bg_t_indices: Tensor  # [num background labels]


class VideoBatch(TypedDict):
    batch_size: int
    video_id: list[str]
    index: list[int]
    orig_length: list[int]  # temporal length before temporal sampling
    duration: list[float]  # duration of the video in seconds
    frame_rate: list[float]  # fps of the inputs to feature extractor
    stride: list[int]  # num of frames in a snippet
    #
    input: Tensor  # [temporal length, dim features]


class WeakSupBatch(VideoBatch):
    video_labels: Tensor  # [num action classes]


class SparseSupBatch(WeakSupBatch):
    fg_labels: list[Tensor]  # [num action labels, num action classes]
    fg_t_indices: list[Tensor]  # [num action labels]
    bg_t_indices: list[Tensor]  # [num background labels]
