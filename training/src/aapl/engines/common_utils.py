from collections import defaultdict
from collections.abc import Mapping
from typing import TypeVar

import numpy as np
import torch
from ignite.engine import Engine
from torch import Tensor


def calculate_average_partial_losses(engine: Engine) -> None:
    partial_loss_lists = defaultdict(list)
    for loss_dict in engine.state.loss_dict_store:  # type: ignore
        for key, value in loss_dict.items():
            partial_loss_lists[key].append(value.item())
    for key, loss_list in partial_loss_lists.items():
        engine.state.metrics[key] = np.array(loss_list).mean()


T_dict = TypeVar("T_dict", bound=Mapping)


def to_device(
    instance: T_dict, device: torch.device | str = "cuda", non_blocking: bool = True
) -> T_dict:
    return {  # type: ignore
        key: (
            value.to(device=device, non_blocking=non_blocking)
            if isinstance(value, Tensor)
            else (
                [elem.to(device=device, non_blocking=non_blocking) for elem in value]
                if isinstance(value, list) and isinstance(value[0], Tensor)
                else value
            )
        )
        for key, value in instance.items()
    }


def record_stream(instance: T_dict, stream: torch.cuda.Stream) -> None:
    for value in instance.values():
        if isinstance(value, Tensor):
            value.record_stream(stream)  # type: ignore
        elif isinstance(value, list) and isinstance(value[0], Tensor):
            for elem in value:
                elem.record_stream(stream)
