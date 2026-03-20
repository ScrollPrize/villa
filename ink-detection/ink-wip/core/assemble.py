from __future__ import annotations

from dataclasses import replace

from ink.core.experiment import Experiment
from ink.core.types import DataBundle


def build_experiment_data(data, *, runtime=None, augment=None) -> DataBundle:
    if isinstance(data, DataBundle):
        return data
    bundle = data.build(runtime=runtime, augment=augment)
    assert isinstance(bundle, DataBundle)
    return bundle


def assemble_experiment(experiment: Experiment, bundle: DataBundle, *, logger=None) -> Experiment:
    assert "augment" in bundle.extras
    bound_augment = bundle.extras["augment"]
    bound_runtime = experiment.runtime.build(data=bundle, augment=bound_augment)
    bound_model = experiment.model.build(data=bundle, runtime=bound_runtime, augment=bound_augment)
    bound_objective = experiment.objective.build(bundle)

    bound_stitch = None
    if experiment.stitch is not None:
        bound_stitch = experiment.stitch.build(bundle, logger=logger, patch_loss=experiment.loss)

    bound_evaluator = None
    if experiment.evaluator is not None:
        bound_evaluator = experiment.evaluator.build(
            data=bundle,
            runtime=bound_runtime,
            stitch=bound_stitch,
            logger=logger,
            patch_loss=experiment.loss,
        )

    bound_experiment = replace(
        experiment,
        runtime=bound_runtime,
        augment=bound_augment,
        model=bound_model,
        objective=bound_objective,
        stitch=bound_stitch,
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
