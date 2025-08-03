from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class OptimCfg:
    optim_type: str | None = None
    learning_rate: float = MISSING
    weight_decay: float = 0.0

    adam_betas: tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8

    batch_size: int = MISSING
