from __future__ import annotations

from dataclasses import replace

from ink.experiments.erm import EXPERIMENT as ERM_EXPERIMENT, _STITCH_RUNTIME
from ink.recipes.stitch import StitchInferenceRecipe
from ink.recipes.trainers import StitchInferenceTrainer


EXPERIMENT = replace(
    ERM_EXPERIMENT,
    name="erm_stitch_inference",
    loss=None,
    objective=None,
    trainer=StitchInferenceTrainer(
        stitch_inference=StitchInferenceRecipe(
            stitch_runtime=_STITCH_RUNTIME,
        ),
    ),
    evaluator=None,
)
