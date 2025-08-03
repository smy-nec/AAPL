from __future__ import annotations

from typing import Any, Callable, Generic, Literal, NamedTuple, Type, TypeVar, overload

import numpy as np
import torch.nn as nn
from torch import Tensor

from aapl.config import Config
from aapl.datasets.structures import VideoInstance, WeakSupBatch

T = TypeVar("T")


class Predictions(NamedTuple):
    labels: np.ndarray
    segments: np.ndarray
    scores: np.ndarray


class EmbedderBase(nn.Module):
    def forward(self, input_features: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, input_features: Tensor) -> Tensor:
        return super().__call__(input_features)  # type: ignore[no-any-return]

    @classmethod
    def from_cfg(cls: Type[EmbedderBase], cfg: Config) -> EmbedderBase:
        raise NotImplementedError


class HeadBase(nn.Module, Generic[T]):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, features: Tensor) -> T:
        return self.forward_head(features)

    def forward_head(self, features: Tensor) -> T:
        raise NotImplementedError

    def calculate_losses(
        self, head_out: T, instance: WeakSupBatch
    ) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplementedError

    def __call__(self, features: Tensor) -> T:
        return super().__call__(features)  # type: ignore[no-any-return]

    @classmethod
    def from_cfg(cls: Type[HeadBase], cfg: Config) -> HeadBase:
        raise NotImplementedError


class PredictorBase:
    @classmethod
    def from_cfg(cls: Type[PredictorBase], cfg: Config) -> PredictorBase:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError


class AAPLModel(nn.Module, Generic[T]):
    def __init__(
        self,
        embedder: EmbedderBase,
        head: HeadBase[T],
        predictor: Callable[..., Predictions],
        head_out_postprocess: Callable,
    ) -> None:
        super().__init__()
        self.embedder = embedder
        self.head = head
        self.predictor = predictor
        self.head_out_postprocess = head_out_postprocess

    def forward(
        self,
        instance: WeakSupBatch,
        predict_segments: bool = False,
    ) -> T | tuple[Predictions, T]:
        assert not predict_segments or instance["batch_size"] == 1
        embedded_features = self.embedder(instance["input"])
        head_out = self.head(embedded_features)
        if predict_segments:
            t_stride = (instance["stride"][0] * instance["orig_length"][0]) / (
                instance["frame_rate"][0] * instance["input"].size(1)
            )
            return (
                self.predictor(*self.head_out_postprocess(head_out), t_stride),
                head_out,
            )
        return head_out

    @overload
    def __call__(
        self, instance: WeakSupBatch, predict_segments: Literal[True]
    ) -> tuple[Predictions, T]:
        ...

    @overload
    def __call__(self, instance: WeakSupBatch, predict_segments: Literal[False]) -> T:
        ...

    def __call__(
        self, instance: WeakSupBatch, predict_segments: bool = False
    ) -> T | tuple[Predictions, T]:
        return super().__call__(instance, predict_segments)  # type: ignore[no-any-return]

    def calculate_loss(
        self, head_out: T, instance: WeakSupBatch
    ) -> tuple[Tensor, dict[str, Tensor]]:
        return self.head.calculate_losses(head_out, instance)

    def predict(self, instance: VideoInstance) -> Predictions:
        t_stride = (instance["stride"] * instance["orig_length"]) / (
            instance["frame_rate"] * instance["input"].size(0)
        )
        head_out = self.head(self.embedder(instance["input"].unsqueeze(0)))
        return self.predictor(*self.head_out_postprocess(head_out), t_stride)
