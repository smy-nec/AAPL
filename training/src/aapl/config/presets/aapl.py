from copy import deepcopy

from omegaconf import DictConfig, OmegaConf

from . import Preset
from .datasets import DATASET_CFG_PRESETS


class AAPLPreset(Preset):
    @classmethod
    def load(cls, cfg: DictConfig) -> DictConfig:
        dataset_name = cfg.dataset.dataset_name

        preset_dict = deepcopy(
            cls.AAPL_DEFAULT.get(cfg.dataset.dataset_name, cls.AAPL_DEFAULT["THUMOS14"])
        )
        preset_dict["dataset"] = deepcopy(DATASET_CFG_PRESETS[dataset_name])
        preset_dict["dataset"].update(cls.AAPL_DATASET_DEFAULT[dataset_name])  # type: ignore
        merged = OmegaConf.merge(preset_dict, cfg)

        return merged  # type: ignore

    AAPL_DEFAULT = {
        "THUMOS14": {
            "model": {
                "dim_input_features": 2048,
                "dim_embedded_features": "${.dim_input_features}",
                "dropout_rate": 0.7,
                "embedder": {
                    "embedder_type": "OneConvLayerEmbedder",
                    "kernel_size": 3,
                    "groups": 1,
                },
                "head": {
                    "head_type": "AAPLHead",
                    "topk_denoms": [8, 3],
                    "fg_threshold": 0.9,
                    "bg_threshold": 0.5,
                    "video_loss_weight": 1.0,
                },
                "predictor": {
                    "predictor_type": "OICPredictor",
                    "proposal_threshold_range": (0.10, 1.0, 0.1),
                    "video_cls_threshold": 0.25,
                    "zero_fill_below_thresh": False,
                    "upsampling_rate": 24,
                    "aligned_upsampling": False,
                    "oic_outer_margin": 0.25,
                    "oic_offset": 0.15,
                    "fuser_type": "soft_nms",
                    "fuser_iou_threshold": 0.4,
                    "fuser_sigma": 0.35,
                },
            },
            "inference": {
                "tiou_threshold_range": (0.1, 1, 0.1),
                "average_mAP_over": [
                    [0, 1, 2, 3, 4, 5, 6],
                    [0, 1, 2, 3, 4],
                    [2, 3, 4, 5, 6],
                ],
            },
            "optim": {
                "optim_type": "Adam",
                "learning_rate": 1e-4,
                "weight_decay": 5e-5,
                "adam_betas": [0.9, 0.999],
                "adam_eps": 1e-8,
                "batch_size": 16,
            },
            "num_epochs": 300,
        },
    }
    AAPL_DATASET_DEFAULT = {
        "THUMOS14": {
            "temporal_sampler_train": "perturbed_uniform_acmtype",
            "temporal_sampler_eval": "uniform",
            "temporal_sampler_target_length": 750,
        },
    }
