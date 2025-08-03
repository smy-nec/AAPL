from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DatasetCfg:
    dataset_name: str = MISSING
    dataset_root: str = MISSING

    with_video_labels: bool = True
    with_sparse_labels: bool = False

    ground_truth_file: str = MISSING
    action_class_names: list[str] = MISSING
    frame_rate: float | None = None
    snippet_stride: int = 16
    num_frames_per_snippet: int = 16
    excluded_video_list: str | None = None

    feature_path_format: str = "{root}/features/{video_id}.npy"
    feature_type: str = "npy"

    training_subset: str | None = None
    validation_subset: str | None = None

    # Available samplers are: "uniform", "perturbed_uniform", "jittering"
    temporal_sampler_train: str | None = None
    temporal_sampler_eval: str | None = None
    temporal_sampler_target_length: int | None = None
    temporal_jittering_half_width: float | None = None

    sparse_label_file: str | None = None

    # Loader settings
    num_workers: int = 4
    pin_memory: bool = True
    persistent_loader: bool = True
    prefetch_loader: bool = True
