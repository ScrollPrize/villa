from __future__ import annotations

from ink.core import Experiment
from ink.recipes.augment import TrainAugment
from ink.recipes.data.normalization import ClipMaxDiv255Normalization
from ink.recipes.data.zarr_data import ZarrPatchDataRecipe
from ink.recipes.eval import PatchEval, StitchEval, ValidationEvaluator
from ink.recipes.losses.dice_bce import DiceBCEBatch
from ink.recipes.metrics import BalancedAccuracy, Dice
from ink.recipes.models import ResNet3D
from ink.recipes.objectives import ERMBatch
from ink.recipes.runtime import TrainRuntime
from ink.recipes.runtime.optimizers import AdamWOptimizer
from ink.recipes.runtime.schedulers import OneCycleScheduler
from ink.recipes.stitch import (
    StitchInferenceRecipe,
    StitchRuntimeRecipe,
)
from ink.recipes.trainers import PatchTrainer
from ink.recipes.trainers.support import WandbLogger


_TRAIN_SEGMENT_IDS = (
    "-1",
    "814_46527_2um_try2",
    "auto_grown_20260220144552896",
    "auto_grown_20260220174252405",
    "w00",
    "w013_20240304141531_2um",
    "w016_2025010815",
    "w017_2025010815",
    "w018_20240304144031_2um",
    "w023_20240304161941_2um",
    "w028_20251208130119156_2um",
    "w028_20260115221",
    "w029_20251212185248662_2um",
    "w029_202601261946",
    "w030_202601301912",
    "w031_2025122323_2um",
)
_VAL_SEGMENT_IDS = (
    "auto_grown_20250919055754487_inp_hr_2um",
)

_SEGMENTS = {
    str(segment_id): {}
    for segment_id in (*_TRAIN_SEGMENT_IDS, *_VAL_SEGMENT_IDS)
}
_STITCH_RUNTIME = StitchRuntimeRecipe(
    config={
        "downsample": 1,
        "train": {
            "viz": {
                "enabled": True,
                "every_n_epochs": 10,
            },
        },
    }
)


EXPERIMENT = Experiment(
    name="erm",
    seed=130697,
    data=ZarrPatchDataRecipe(
        dataset_root="/pscratch/cpa232_scrollprize/data/philodemos/2um_dataset/villa/ink-detection/ink-0309",
        segments=_SEGMENTS,
        train_segment_ids=_TRAIN_SEGMENT_IDS,
        val_segment_ids=_VAL_SEGMENT_IDS,
        in_channels=62,
        patch_size=256,
        tile_size=256,
        stride=128,
        label_suffix="",
        mask_suffix="",
        train_batch_size=24,
        valid_batch_size=32,
        num_workers=12,
        shuffle=True,
        patch_index_cache_dir=(
            "/pscratch/cpa232_scrollprize/data/philodemos/2um_dataset/villa/ink-detection/ink-0309"
            "/.patch_index_cache/erm"
        ),
        cache_train_patches_in_memory=True,
        include_train_valid_mask=False,
        normalization=ClipMaxDiv255Normalization(),
    ),
    model=ResNet3D(
        depth=50,
        norm="batch",
        pretrained=True,
        backbone_pretrained_path=(
            "/pscratch/cpa232_scrollprize/data/philodemos/2um_dataset/villa/ink-detection/r3d50_KM_200ep.pth"
        ),
    ),
    loss=DiceBCEBatch(),
    objective=ERMBatch(),
    runtime=TrainRuntime(
        use_amp=True,
        grad_clip_norm=100.0,
        optimizer=AdamWOptimizer(
            lr=2.20e-04,
            weight_decay=1.79e-07,
            exclude_weight_decay_bias_norm=True,
        ),
        scheduler=OneCycleScheduler(
            pct_start=0.15,
            div_factor=25.0,
            final_div_factor=8.8,
        ),
        wandb=WandbLogger(
            enabled=True,
            project="ink-2um-experiments",
            entity="vesuvius-challenge",
            group="ink-2um",
            tags=("ink-2um",),
            media_downsample=1,
            log_train_every_n_steps=10,
        ),
    ),
    augment=TrainAugment(),
    trainer=PatchTrainer(
        stitch_runtime=_STITCH_RUNTIME,
        epochs=30,
        log_every_n_steps=100,
        save_every_n_epochs=1,
    ),
    evaluator=ValidationEvaluator(
        patch=PatchEval(
            metrics=(
                Dice(),
                BalancedAccuracy(),
                Dice(threshold=96.0 / 255.0),
                BalancedAccuracy(threshold=96.0 / 255.0),
            )
        ),
        stitch_inference=StitchInferenceRecipe(
            stitch_runtime=_STITCH_RUNTIME,
        ),
        stitch=StitchEval(
            metrics=(
                Dice(),
                BalancedAccuracy(),
                Dice(threshold=96.0 / 255.0),
                BalancedAccuracy(threshold=96.0 / 255.0),
            )
        ),
    ),
)
