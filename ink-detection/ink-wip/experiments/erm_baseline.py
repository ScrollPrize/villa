from __future__ import annotations

from ink.core import Experiment
from ink.recipes.augment import TrainAugment
from ink.recipes.data.normalization import ClipMaxDiv255Normalization
from ink.recipes.data.zarr_data import ZarrPatchDataRecipe
from ink.recipes.eval import (
    PatchEval,
    ValidationEvaluator,
)
from ink.recipes.metrics import (
    BalancedAccuracy,
    Dice,
)
from ink.recipes.losses.dice_bce import DiceBCEBatch
from ink.recipes.models import ResNet3D
from ink.recipes.objectives import ERMBatch
from ink.recipes.runtime import TrainRuntime
from ink.recipes.trainers import PatchTrainer


EXPERIMENT = Experiment(
    name="erm_baseline",
    data=ZarrPatchDataRecipe(
        dataset_root=".",
        segments={},
        train_segment_ids=(),
        val_segment_ids=(),
        in_channels=62,
        patch_size=256,
        train_batch_size=1,
        valid_batch_size=1,
        shuffle=False,
        normalization=ClipMaxDiv255Normalization(),
    ),
    model=ResNet3D(),
    loss=DiceBCEBatch(),
    objective=ERMBatch(),
    runtime=TrainRuntime(),
    augment=TrainAugment(),
    stitch=None,
    trainer=PatchTrainer(),
    evaluator=ValidationEvaluator(
        patch=PatchEval(
            metrics=(
                Dice(),
                BalancedAccuracy(),
                Dice(threshold=96.0 / 255.0),
                BalancedAccuracy(threshold=96.0 / 255.0),
            ),
        )
    ),
)
