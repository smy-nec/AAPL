import os

from aapl.config.schema import Config


def prepare_directory(
    cfg: Config,
    with_count_if_exists: bool = True,
    raise_if_exists: bool = False,
) -> str:
    assert not (with_count_if_exists and raise_if_exists)

    try:
        os.makedirs(cfg.output_dir)
    except FileExistsError:
        if raise_if_exists:
            raise
        if with_count_if_exists:
            output_dir = cfg.output_dir.rstrip("/")
            count = 0
            while os.path.exists(f"{output_dir}_{count:03d}"):
                count += 1
                if count >= 1000:
                    raise RuntimeError
            cfg.output_dir = f"{output_dir}_{count:03d}"
            os.makedirs(cfg.output_dir)
    return cfg.output_dir
