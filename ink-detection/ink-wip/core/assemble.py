from __future__ import annotations

from dataclasses import replace

from ink.core.experiment import Experiment
from ink.core.types import DataBundle


def build_experiment_data(data, *, runtime=None, augment=None) -> DataBundle:
    """Build the shared data bundle before binding the rest of the experiment."""
    bundle = data.build(runtime=runtime, augment=augment)
    assert isinstance(bundle, DataBundle)
    return bundle


def assemble_experiment(experiment: Experiment, bundle: DataBundle, *, logger=None) -> Experiment:
    """Bind runtime, model, evaluator, and trainer against one prepared data bundle."""
    bound_augment = bundle.augment
    bound_runtime = experiment.runtime.build(data=bundle, augment=bound_augment)
    bound_model = experiment.model.build(data=bundle, runtime=bound_runtime, augment=bound_augment)
    bound_objective = None if experiment.objective is None else experiment.objective.build(bundle)
    bound_evaluator = None if experiment.evaluator is None else experiment.evaluator.build(
        data=bundle,
        runtime=bound_runtime,
        logger=logger,
    )

    bound_experiment = replace(
        experiment,
        runtime=bound_runtime,
        augment=bound_augment,
        model=bound_model,
        objective=bound_objective,
        evaluator=bound_evaluator,
    )

    bound_trainer = None
    if experiment.trainer is not None:
        bound_trainer = experiment.trainer.build(
            experiment=bound_experiment,
            data=bundle,
            logger=logger,
        )
    return replace(bound_experiment, trainer=bound_trainer)


__all__ = [
    "assemble_experiment",
    "build_experiment_data",
]
