from torch.utils.data import DataLoader

from aapl.config.schema import Config

from .loader import MultiEpochsDataLoader, PrefetchLoader, collate_instances
from .video_dataset import VideoDataset


def get_data_loaders(
    cfg: Config, with_train: bool = True, with_val: bool = True
) -> dict[str, PrefetchLoader | DataLoader]:
    data_loaders: dict[str, DataLoader] = {}
    loader_type = MultiEpochsDataLoader if cfg.dataset.persistent_loader else DataLoader
    if with_train:
        train_data = VideoDataset.from_cfg(cfg, "training")
        base_train_loader = loader_type(
            train_data,
            batch_size=cfg.optim.batch_size,
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            prefetch_factor=2,
            pin_memory=cfg.dataset.pin_memory,
            collate_fn=collate_instances,
        )
        data_loaders["training"] = base_train_loader
    if with_val:
        val_data = VideoDataset.from_cfg(cfg, "validation")
        base_val_loader = loader_type(
            val_data,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
            prefetch_factor=2,
            pin_memory=cfg.dataset.pin_memory,
            collate_fn=collate_instances,
        )
        data_loaders["validation"] = base_val_loader

    if cfg.dataset.prefetch_loader:
        return _wrap_loaders_with_prefetch_loader(data_loaders)  # type: ignore

    return data_loaders  # type: ignore


def _wrap_loaders_with_prefetch_loader(
    data_loaders: dict[str, DataLoader]
) -> dict[str, PrefetchLoader]:
    ret = {}
    for key in data_loaders:
        base_loader = data_loaders[key]
        assert isinstance(base_loader, DataLoader)
        ret[key] = PrefetchLoader(base_loader)
    return ret
