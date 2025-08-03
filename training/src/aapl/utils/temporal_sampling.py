from __future__ import annotations

import numpy as np


def linear_upsampling(
    array: np.ndarray, scale_factor: float, dim: int = 0, align: bool = False
) -> np.ndarray:
    if align:
        ticks = np.arange(0.5 / scale_factor, array.shape[dim], 1 / scale_factor) - 0.5
        start_padding = 2 * array.take([0], axis=dim) - array.take([1], axis=dim)
        end_padding = 2 * array.take([-1], axis=dim) - array.take([-2], axis=dim)
        array = np.concatenate([start_padding, array, end_padding], axis=dim)
    else:
        ticks = np.arange(0, array.shape[dim], 1 / scale_factor)
        end_padding = 2 * array.take([-1], axis=dim) - array.take([-2], axis=dim)
        array = np.concatenate([array, end_padding], axis=dim)
    idx_low = np.floor(ticks).astype(int)
    idx_high = idx_low + 1

    values_low = array.take(idx_low, axis=dim)
    values_high = array.take(idx_high, axis=dim)

    shape = (1,) * dim + (len(idx_high),) + (1,) * (array.ndim - dim - 1)
    ticks = ticks.reshape(shape)
    idx_high = idx_high.reshape(shape)
    idx_low = idx_low.reshape(shape)
    out: np.ndarray = values_high * (ticks - idx_low) + values_low * (idx_high - ticks)

    return out
