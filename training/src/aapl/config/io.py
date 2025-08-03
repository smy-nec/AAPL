import argparse
from collections.abc import Iterable
from typing import Any, overload

from omegaconf import DictConfig, ListConfig, OmegaConf

from .autofill import autofill
from .presets.aapl import AAPLPreset
from .schema import Config


@overload
def load_cfg(description: str = "", parent_parsers: None = None) -> Config:
    ...


@overload
def load_cfg(
    *,
    parent_parsers: list[argparse.ArgumentParser],
) -> tuple[Config, argparse.Namespace]:
    ...


@overload
def load_cfg(
    description: str, parent_parsers: list[argparse.ArgumentParser]
) -> tuple[Config, argparse.Namespace]:
    ...


def load_cfg(
    description: str = "", parent_parsers: list[argparse.ArgumentParser] | None = None
) -> Config | tuple[Config, argparse.Namespace]:
    parent_parsers = [] if parent_parsers is None else parent_parsers
    parser = argparse.ArgumentParser(description=description, parents=parent_parsers)
    parser.add_argument(
        "--cfg",
        type=str,
        metavar="path/to/config.yaml",
        help="Path to configuration YAML type file.",
        required=True,
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()
    path, overrides = args.cfg, args.overrides

    cfg = OmegaConf.load(path)
    cfg = AAPLPreset.load(cfg)  # type: ignore
    cfg = OmegaConf.merge(Config, cfg)

    if overrides:
        cfg_overrides = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, cfg_overrides)

    autofill(cfg)  # type: ignore

    if parent_parsers:
        return cfg, args  # type: ignore
    return cfg  # type: ignore


def to_flat_dict(
    cfg: DictConfig | ListConfig, prefix: str | None = None
) -> dict[str, Any]:
    if prefix is None:
        prefix = ""
    if isinstance(cfg, DictConfig):
        cfg_items: Iterable = cfg.items()
    elif isinstance(cfg, ListConfig):
        cfg_items = enumerate(cfg)
    else:
        raise TypeError(f"Unsupported type: {type(cfg)}")
    flat_dict = {}
    for key, value in cfg_items:
        key = str(key)
        if isinstance(value, (DictConfig, ListConfig)):
            flat_dict.update(to_flat_dict(value, prefix=f"{prefix}{key}."))
        else:
            flat_dict[f"{prefix}{key}"] = value
    return flat_dict
