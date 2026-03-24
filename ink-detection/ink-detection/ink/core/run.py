from __future__ import annotations

from ink.core.assemble import assemble_experiment, build_experiment_data
from ink.core.experiment import Experiment

def run_experiment(experiment: Experiment, *, logger=None, **trainer_kwargs):
    """Build the bound experiment graph and delegate execution to the trainer."""
    bundle = build_experiment_data(
        experiment.data,
        runtime=experiment.runtime,
        augment=experiment.augment,
    )
    bound_experiment = assemble_experiment(experiment, bundle, logger=logger)
    assert bound_experiment.trainer is not None
    return bound_experiment.trainer.run(**trainer_kwargs)


__all__ = ["run_experiment"]
