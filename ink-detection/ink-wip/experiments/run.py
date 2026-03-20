from __future__ import annotations

import importlib
import logging
import sys

from ink.core import run_experiment


def _configure_progress_logger():
    logger = logging.getLogger("ink.progress")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)
    return logger


def _progress_log(message: str) -> None:
    logger = _configure_progress_logger()
    logger.info(str(message))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("usage: python -m ink.experiments.run [experiment_module_name]")

    experiment_name = str(args[0]).strip()
    try:
        module = importlib.import_module(f"ink.experiments.{experiment_name}")
    except ModuleNotFoundError as exc:
        if exc.name == f"ink.experiments.{experiment_name}":
            raise SystemExit(f"unknown experiment module {experiment_name!r}") from exc
        raise
    experiment = getattr(module, "EXPERIMENT", None)
    if experiment is None:
        raise SystemExit(f"experiment module {experiment_name!r} must define EXPERIMENT")
    _configure_progress_logger()
    return run_experiment(experiment, logger=_progress_log)


if __name__ == "__main__":
    main()
