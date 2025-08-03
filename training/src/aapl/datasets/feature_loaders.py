import os
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from aapl.config.schema import Config


def npy_loader(path: str) -> Tensor:
    np_feature = np.load(path)
    return torch.from_numpy(np_feature).float()


class PreloadedWTALCStyleFeatureLoader:
    def __init__(self, dataset_name: str, data_root: str) -> None:
        video_names = np.load(
            os.path.join(data_root, "videoname.npy"), allow_pickle=True
        )
        self.video_name_to_idx = {
            name.decode(): idx for idx, name in enumerate(video_names)
        }
        self.joint_features = np.load(
            os.path.join(data_root, f"{dataset_name}-I3D-JOINTFeatures.npy"),
            allow_pickle=True,
        )

    def __call__(self, video_name: str) -> Tensor:
        return torch.from_numpy(
            self.joint_features[self.video_name_to_idx[video_name]]
        ).float()


def npy_two_stream_loader(path: str) -> Tensor:
    path_comp = path.split("/")

    rgb_path = os.path.join("/".join(path_comp[:-1]), "rgb", path_comp[-1])
    flow_path = os.path.join("/".join(path_comp[:-1]), "flow", path_comp[-1])

    rgb_feats = np.load(rgb_path)
    flow_feats = np.load(flow_path)
    np_feature = np.concatenate((rgb_feats, flow_feats), axis=1)

    return torch.from_numpy(np_feature).float()


def npz_loader(path: str) -> Tensor:
    *file_path_comp, npz_key = path.split("/")
    file_path = "/".join(file_path_comp)
    np_feature = np.load(file_path)[npz_key]
    return torch.from_numpy(np_feature).float()


def get_feature_loader(feature_type: str, cfg: Config) -> Callable[[str], Tensor]:
    if feature_type == "npy":
        feature_loader: Callable[[str], Tensor] = npy_loader
    elif feature_type == "npy_two_stream":
        feature_loader = npy_two_stream_loader
    elif feature_type == "npz":
        feature_loader = npz_loader
    elif feature_type == "wtalc_style":
        feature_loader = PreloadedWTALCStyleFeatureLoader(
            cfg.dataset.dataset_name, cfg.dataset.dataset_root
        )
    else:
        raise ValueError(f"unknown feature type: {feature_type}")
    return feature_loader
