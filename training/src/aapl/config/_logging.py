from dataclasses import dataclass


@dataclass
class LoggingCfg:
    enable_checkpoints: bool = True
