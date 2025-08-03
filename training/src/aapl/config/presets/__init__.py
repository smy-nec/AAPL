from typing import Protocol as _Protocol

from omegaconf import DictConfig as _DictConfig


class Preset(_Protocol):
    @classmethod
    def load(cls, cfg: _DictConfig) -> _DictConfig:
        ...
