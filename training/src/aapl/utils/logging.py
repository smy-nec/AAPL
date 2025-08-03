import logging


def setup_logging(
    name: str | None = None,
    level: int = logging.INFO,
    with_stream_handler: bool = False,
    format: str = "%(asctime)s %(name)s %(levelname)s: %(message)s",
    filepath: str | None = None,
    reset: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(name)

    if reset:
        if logger.hasHandlers():
            for h in list(logger.handlers):
                logger.removeHandler(h)

    logger.setLevel(level)
    formatter = logging.Formatter(format)
    if with_stream_handler:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if filepath:
        fh = logging.FileHandler(filepath)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
