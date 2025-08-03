from __future__ import annotations

import os
import re
from typing import Any

import mlflow as _mlflow
from flask import Config
from mlflow.entities import Experiment, Run

from aapl.config.io import to_flat_dict

_INVALID_METRIC_NAME_PATTERN = re.compile(r"[^/\w.\- ]+")


def set_experiment(
    experiment_name: str | None = None, experiment_id: str | None = None
) -> Experiment:
    """
    Set the given experiment as the active experiment in distributed settings. The experiment must
    either be specified by name via `experiment_name` or by ID via `experiment_id`. The experiment
    name and ID cannot both be specified. If the experiment name is specified, the experiment will
    be created by the 0th-rank process if it does not already exist. If the experiment ID is
    specified, the experiment must already exist.

    Args:
        experiment_name: Name of the experiment.
        experiment_id: ID of the experiment.
    Returns:
        The experiment object.
    """
    # Check that only one of experiment_name and experiment_id is specified.
    if experiment_name is not None and experiment_id is not None:
        raise ValueError(
            "Only one of experiment_name and experiment_id can be specified."
        )

    if experiment_name is not None:
        experiment = _mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = _mlflow.create_experiment(experiment_name)
            experiment = _mlflow.get_experiment(experiment_id)
    elif experiment_id is not None:
        experiment = _mlflow.get_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment {experiment_id} does not exist.")
    else:
        raise ValueError("Either experiment_name or experiment_id must be specified.")

    _mlflow.set_experiment(experiment.name)
    return experiment


def start_run(
    run_id: str | None = None,
    experiment_id: str | None = None,
    run_name: str | None = None,
    nested: bool = False,
    tags: dict[str, Any] | None = None,
    description: str | None = None,
    cfg: Config | None = None,
) -> _mlflow.ActiveRun:
    """Start a new MLflow run possibly in distributed settigns.

    Args:
        run_id: ID of the run. If provided, the specified run will be resumed. Otherwise, a new run
            will be created in the 0th-rank process, and the run ID will be broadcasted to all
            other processes.
        experiment_id: ID of the experiment under which to create the run. If not provided, the
            active experiment will be used.
        run_name: Name of the run. Used only if no `run_id` is specified. If not provided, a unique
            name will be generated.
        nested: Whether to start a nested run.
        tags: A dictionary of string tags to log for the run.
        description: A string description of the run.
        distributed: Whether to start a distributed run.
    """
    tags = tags or {}

    if cfg:
        flat_cfg = to_flat_dict(cfg, prefix="cfg.")  # type: ignore
        tags.update({k: str(v) for k, v in flat_cfg.items()})

    tags["env.hostname"] = os.uname()[1]
    tags["env.cwd"] = os.getcwd()

    return _mlflow.start_run(
        run_id=run_id,
        experiment_id=experiment_id,
        run_name=run_name,
        nested=nested,
        tags=tags,
        description=description,
    )


def active_run() -> _mlflow.ActiveRun:
    """Get the active MLflow run."""
    return _mlflow.active_run()


def last_active_run() -> Run | None:
    """Get the last active MLflow run."""
    return _mlflow.last_active_run()


def end_run() -> None:
    """End the current MLflow run possibly in distributed settings."""
    _mlflow.end_run()


def log_dict(dictionary: dict[str, Any], artifact_file: str) -> None:
    """
    Log a JSON/YAML-serializable object (e.g. `dict`) as an artifact. The serialization
    format (JSON or YAML) is automatically inferred from the extension of `artifact_file`.
    If the file extension doesn't exist or match any of [".json", ".yml", ".yaml"],
    JSON format is used. If the active run is in distributed settings, the artifact is
    saved only by the 0th-rank process.
    """
    _mlflow.log_dict(dictionary, artifact_file)


def log_metric(key: str, value: float, step: int | None = None) -> None:
    """
    Log a metric as a metric of the active run. If the active run is in distributed settings, the
    metric is logged only by the 0th-rank process.
    """
    _mlflow.log_metric(key, value, step)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """
    Log a dictionary of metrics as metrics of the active run. If the active run is in
    distributed settings, the metrics are logged only by the 0th-rank process.
    """
    metrics = {_INVALID_METRIC_NAME_PATTERN.sub("_", k): v for k, v in metrics.items()}
    _mlflow.log_metrics(metrics, step)


def log_params(params: dict[str, Any]) -> None:
    """
    Log a dictionary of parameters as params of the active run. If the active run is in
    distributed settings, the params are logged only by the 0th-rank process.
    """
    _mlflow.log_params(params)


def log_table(data: dict[str, Any], artifact_file: str) -> None:
    """
    Log a table as an artifact. If the active run is in distributed settings, the artifact is
    saved only by the 0th-rank process.
    """
    _mlflow.log_table(data, artifact_file)
