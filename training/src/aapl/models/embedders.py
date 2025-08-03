from __future__ import annotations

from typing import Callable, Type

import torch.nn as nn
from torch import Tensor

from aapl.config import Config

from .aapl_model import EmbedderBase


class Permute(nn.Module):
    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(*self.dims)


class OneConvLayerEmbedder(EmbedderBase):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 1,
        groups: int = 1,
        activation_fn: Callable[[], Callable[[Tensor], Tensor]] = nn.ReLU,
        dropout_rate: float | None = None,
        prenorm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if dropout_rate:
            self.dropout: nn.Dropout | None = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.conv = nn.Conv1d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups,
        )
        self.act = activation_fn()
        self.prenorm = prenorm

    def forward(self, input_features: Tensor) -> Tensor:
        # input_features: [B, T, D]
        if self.prenorm:
            input_features = self.prenorm(input_features)
        if self.dropout:
            input_features = self.dropout(input_features)
        input_features = input_features.permute((0, 2, 1))  # [B, D, T]
        out = self.act(self.conv(input_features).permute((0, 2, 1)))  # [B, T, D]
        return out

    @classmethod
    def from_cfg(cls: Type[OneConvLayerEmbedder], cfg: Config) -> OneConvLayerEmbedder:
        embedder_cfg = cfg.model.embedder

        assert embedder_cfg.kernel_size is not None
        return cls(
            cfg.model.dim_input_features,
            cfg.model.dim_embedded_features,
            embedder_cfg.kernel_size,
            groups=embedder_cfg.groups,
            dropout_rate=cfg.model.dropout_rate if embedder_cfg.dropout_input else None,
        )
