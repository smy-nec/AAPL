from collections.abc import Callable
from functools import partial
from typing import Any, overload

import numpy as np
import scipy.interpolate as interpolate
import torch

from aapl.config import Config
from aapl.datasets.structures import SparseSupInstance, VideoInstance


def get_temporal_sampler(cfg: Config, split: str) -> Callable | None:
    if split == "training":
        sampler = cfg.dataset.temporal_sampler_train
    else:
        sampler = cfg.dataset.temporal_sampler_eval

    if sampler is None:
        return None

    target_length = cfg.dataset.temporal_sampler_target_length
    if sampler == "uniform":
        base_sampler: Callable = uniformly_sample_long_sequence
        kwargs: dict[str, Any] = {"max_length": target_length}
    elif sampler == "perturbed_uniform":
        base_sampler = perturbed_uniform_sampling
        kwargs = {"num_snippets": target_length}
    elif sampler == "perturbed_uniform_acmtype":
        base_sampler = perturbed_uniform_sampling_acmtype
        kwargs = {"num_snippets": target_length}
    elif sampler == "jittering":
        base_sampler = temporal_jittering
        kwargs = {
            "max_length": target_length,
            "half_width": cfg.dataset.temporal_jittering_half_width,
        }
    else:
        raise ValueError(f"unknown temporal sampler: {sampler}")

    return partial(_sampler_wrapper, base_sampler=base_sampler, kwargs=kwargs)


def _sampler_wrapper(
    inst: SparseSupInstance, base_sampler: Callable, kwargs: Any
) -> VideoInstance:
    feature_np: np.ndarray = inst["input"].numpy()
    if "fg_labels" in inst:
        sparse_labels_np = np.zeros(
            (feature_np.shape[0], inst["video_labels"].shape[0] + 1), dtype=np.int64
        )
        sparse_labels_np[inst["fg_t_indices"], :-1] = inst["fg_labels"]
        sparse_labels_np[inst["bg_t_indices"], -1] = 1
        sampled_feature, sparse_labels_np = base_sampler(
            feature_np, sparse_labels_np, **kwargs
        )
        fg_t_indices_np = np.any(sparse_labels_np[:, :-1], axis=-1)
        fg_labels_np = sparse_labels_np[fg_t_indices_np, :-1]
        inst["fg_labels"] = torch.from_numpy(fg_labels_np).float()
        inst["fg_t_indices"] = torch.from_numpy(np.nonzero(fg_t_indices_np)[0])
        inst["bg_t_indices"] = torch.from_numpy(np.nonzero(sparse_labels_np[:, -1])[0])
    else:
        sampled_feature = base_sampler(feature_np, **kwargs)  # type: ignore

    inst["input"] = torch.from_numpy(sampled_feature).float()
    return inst


@overload
def uniformly_sample_long_sequence(  # type: ignore[overload-overlap]
    feature_array: np.ndarray, *, max_length: int
) -> np.ndarray:
    ...


@overload
def uniformly_sample_long_sequence(
    feature_array: np.ndarray, *labels: np.ndarray, max_length: int
) -> tuple[np.ndarray, ...]:
    ...


def uniformly_sample_long_sequence(
    feature_array: np.ndarray, *labels: np.ndarray, max_length: int
) -> np.ndarray | tuple[np.ndarray, ...]:
    input_length = feature_array.shape[0]
    assert input_length > 0 and max_length > 0
    if input_length > max_length:
        indices_fp = np.arange(max_length) * input_length / max_length
        indices = np.floor(indices_fp).astype(int)
        if not labels:
            return feature_array[indices]  # type: ignore
        return (feature_array[indices],) + tuple(label[indices] for label in labels)
    if input_length == 1:
        if not labels:
            return feature_array[[0, 0]]
        return (feature_array[[0, 0]],) + tuple(label[[0, 0]] for label in labels)
    if not labels:
        return feature_array
    return (feature_array,) + labels


@overload
def perturbed_uniform_sampling(  # type: ignore[overload-overlap]
    feature_array: np.ndarray, *, num_snippets: int
) -> np.ndarray:
    ...


@overload
def perturbed_uniform_sampling(
    feature_array: np.ndarray, *labels: np.ndarray, num_snippets: int
) -> tuple[np.ndarray, ...]:
    ...


def perturbed_uniform_sampling(
    feature_array: np.ndarray, *labels: np.ndarray, num_snippets: int
) -> np.ndarray | tuple[np.ndarray, ...]:
    input_length = feature_array.shape[0]
    assert input_length > 0 and num_snippets > 0

    if input_length < num_snippets:
        indices = np.sort(np.random.choice(input_length, num_snippets, replace=True))
        if not labels:
            return feature_array[indices]
        return (feature_array[indices],) + tuple(label[indices] for label in labels)
    milestones = np.rint(
        np.arange(num_snippets + 1) * (input_length - 1) / num_snippets
    ).astype(int)
    indices = np.zeros(num_snippets, dtype=int)
    for i, (start, end) in enumerate(zip(milestones[:-1], milestones[1:])):
        indices[i] = np.random.choice(np.arange(start, end + 1))
    if not labels:
        return feature_array[indices]  # type: ignore[no-any-return]
    return (feature_array[indices],) + tuple(label[indices] for label in labels)


@overload
def perturbed_uniform_sampling_acmtype(  # type: ignore[overload-overlap]
    feature_array: np.ndarray, *, num_snippets: int
) -> np.ndarray:
    ...


@overload
def perturbed_uniform_sampling_acmtype(
    feature_array: np.ndarray, *labels: np.ndarray, num_snippets: int
) -> tuple[np.ndarray, ...]:
    ...


def perturbed_uniform_sampling_acmtype(
    feature_array: np.ndarray, *labels: np.ndarray, num_snippets: int
) -> np.ndarray | tuple[np.ndarray, ...]:
    input_length = feature_array.shape[0]
    assert input_length > 0 and num_snippets > 0

    if input_length < num_snippets:
        indices = np.sort(np.random.choice(input_length, num_snippets, replace=True))
        if not labels:
            return feature_array[indices]
        return (feature_array[indices],) + tuple(label[indices] for label in labels)
    milestones_fp = np.arange(num_snippets) * input_length / num_snippets
    milestones = np.floor(milestones_fp).astype(int)
    indices = np.zeros(num_snippets, dtype=int)
    for i, (start, end) in enumerate(zip(milestones[:-1], milestones[1:])):
        indices[i] = np.random.choice(np.arange(start, end + 1))
    indices[-1] = np.random.choice(np.arange(milestones[-2], input_length))
    if not labels:
        return feature_array[indices]  # type: ignore[no-any-return]
    return (feature_array[indices],) + tuple(label[indices] for label in labels)


@overload
def temporal_jittering(  # type: ignore[overload-overlap]
    feature_array: np.ndarray, *, half_width: float, max_length: int = -1
) -> np.ndarray:
    ...


@overload
def temporal_jittering(
    feature_array: np.ndarray,
    *labels: np.ndarray,
    half_width: float,
    max_length: int = -1,
) -> tuple[np.ndarray, ...]:
    ...


def temporal_jittering(
    feature_array: np.ndarray,
    *labels: np.ndarray,
    half_width: float,
    max_length: int = -1,
) -> np.ndarray | tuple[np.ndarray, ...]:
    input_length = feature_array.shape[0]
    if max_length > 0 and input_length > max_length:
        result_length = max_length
        start_frame = np.random.randint(input_length - max_length + 1)
    else:
        result_length = input_length
        start_frame = 0
    steps = np.random.random(result_length) * 2 * half_width + 1 - half_width
    indices = np.cumsum(steps) - 1 + start_frame
    indices = np.clip(indices, a_min=0, a_max=input_length - 1)

    f = interpolate.interp1d(
        np.arange(input_length),
        feature_array,
        axis=0,
        kind="linear",
        fill_value="extrapolate",
    )
    if not labels:
        return f(indices)  # type: ignore[no-any-return]
    return (f(indices),) + tuple(
        label[np.rint(indices).astype(int)] for label in labels
    )
