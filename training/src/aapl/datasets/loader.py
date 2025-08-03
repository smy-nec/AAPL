from __future__ import annotations

import multiprocessing
from collections.abc import Iterable, Iterator, Sequence
from contextlib import suppress
from functools import partial
from typing import Generator, Optional, TypeVar, Union

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.collate import collate_tensor_fn
from torch.utils.data.dataloader import _collate_fn_t, _worker_init_fn_t

from aapl.engines.common_utils import record_stream, to_device

from .structures import WeakSupBatch, WeakSupInstance

T_co = TypeVar("T_co", covariant=True)
_valid_context_t = Union[str, multiprocessing.context.BaseContext]


def collate_instances(batch: list[WeakSupInstance]) -> WeakSupBatch:
    elem = batch[0]
    instance_keys = elem.keys()
    input_tn = collate_tensor_fn([inst["input"] for inst in batch])
    video_labels_tn = collate_tensor_fn([inst["video_labels"] for inst in batch])

    collated: WeakSupBatch = {
        key: [inst[key] for inst in batch]  # type: ignore
        for key in instance_keys
        if key not in ("input", "video_labels")
    }
    collated["batch_size"] = len(batch)
    collated["input"] = input_tn
    collated["video_labels"] = video_labels_tn

    return collated


class MultiEpochsDataLoader(DataLoader[T_co]):
    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        sampler: Union[Sampler, Iterable, None] = None,
        batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context: Optional[_valid_context_t] = None,
        generator: Optional[torch.Generator] = None,
        *,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
        pin_memory_device: str = "",
    ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device,
        )
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self) -> int:
        return (
            len(self.sampler.sampler)  # type: ignore
            if self.batch_sampler is None
            else len(self.batch_sampler.sampler)  # type: ignore
        )

    def __iter__(self) -> Generator:  # type: ignore[override]
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler: Iterable):
        self.sampler = sampler

    def __iter__(self) -> Iterator:
        while True:
            yield from iter(self.sampler)


class PrefetchLoader:
    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

    def __iter__(self) -> Iterator:
        first = True
        if torch.cuda.is_available():
            stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
            stream_context = partial(torch.cuda.stream, stream=stream)
        else:
            stream = None
            stream_context = suppress  # type: ignore[assignment]

        for next_batch in self.loader:
            with stream_context():
                next_batch = to_device(next_batch)

            if not first:
                yield batch  # type: ignore[has-type] # noqa
                record_stream(batch, torch.cuda.default_stream())  # type: ignore[has-type] # noqa
            else:
                first = False

            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)  # type: ignore[no-untyped-call]

            batch = next_batch

        yield batch

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def sampler(self) -> Iterable:
        return self.loader.sampler

    @property
    def dataset(self) -> Dataset:
        return self.loader.dataset
