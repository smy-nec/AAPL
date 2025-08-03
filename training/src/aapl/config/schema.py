from dataclasses import dataclass

from omegaconf import MISSING

from ._dataset import DatasetCfg
from ._inference import InferenceCfg
from ._logging import LoggingCfg
from ._model import ModelCfg
from ._optim import OptimCfg


@dataclass
class Config:
    experiment_name: str | None = "experiment"

    model: ModelCfg = ModelCfg()
    dataset: DatasetCfg = DatasetCfg()
    inference: InferenceCfg = InferenceCfg()
    optim: OptimCfg = OptimCfg()
    logging: LoggingCfg = LoggingCfg()

    output_dir: str = MISSING
    num_epochs: int = MISSING
    evaluate_every: int | None = None
    checkpoint_path: str | None = None

    rng_seed: int = 0
