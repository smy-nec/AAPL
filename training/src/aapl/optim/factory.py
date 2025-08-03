from typing import Any

from torch.optim import Adam, Optimizer

from aapl.config.schema import Config


def get_optimizer(cfg: Config, params: Any) -> Optimizer:
    optim_cfg = cfg.optim
    return Adam(
        params,
        optim_cfg.learning_rate,
        betas=optim_cfg.adam_betas,
        eps=optim_cfg.adam_eps,
        weight_decay=optim_cfg.weight_decay,
    )
