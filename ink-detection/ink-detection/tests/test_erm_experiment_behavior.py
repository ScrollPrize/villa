from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader

from ink.core import Batch, BatchMeta, assemble_experiment, build_experiment_data
from ink.experiments import ERM_EXPERIMENT, ERM_PATCH_BUNDLE_EXPERIMENT, ERM_STITCH_INFERENCE_EXPERIMENT
from ink.experiments import run as experiment_run
from ink.recipes.augment import TrainAugment
from ink.recipes.data.normalization import ClipMaxDiv255Normalization
from ink.recipes.data.patch_bundle import GeneratedPatchBundleDataRecipe
from ink.recipes.data.zarr_data import ZarrPatchDataRecipe
from ink.recipes.eval import PatchEval, StitchEval, ValidationEvaluator
from ink.recipes.metrics import BalancedAccuracy, Dice
from ink.recipes.losses.dice_bce import DiceBCEBatch
from ink.recipes.models import ResNet3D, ResNet3DSegmentationModel
from ink.recipes.objectives import ERMObjective
from ink.recipes.runtime import (
    AdamWOptimizer,
    CosineScheduler,
    OneCycleScheduler,
    SGDOptimizer,
    TrainRuntime,
)
from ink.recipes.stitch import StitchInferenceRecipe, StitchRuntimeRecipe
from ink.recipes.trainers import PatchTrainer, StitchInferenceTrainer
from tests.support.build_recipes import optional_build_recipe, required_build_recipe
from tests.support.in_memory_data import InMemoryPatchDataRecipe, InMemoryPatchSamples


def _image(value: int, *, size: int = 8, in_channels: int = 62):
    return np.full((size, size, in_channels), value, dtype=np.uint8)


def _label(value: int, *, size: int = 8):
    return np.full((size, size, 1), value, dtype=np.uint8)


def _patch_samples(*, train_count: int, val_count: int, in_channels: int = 62):
    train = InMemoryPatchSamples(
        images=tuple(_image(16 * (idx + 1), in_channels=in_channels) for idx in range(train_count)),
        labels=tuple(_label(255 if idx % 2 == 0 else 0) for idx in range(train_count)),
        groups=tuple(idx % 2 for idx in range(train_count)),
    )
    val = InMemoryPatchSamples(
        images=tuple(_image(24 * (idx + 1), in_channels=in_channels) for idx in range(val_count)),
        labels=tuple(_label(255 if idx % 2 == 0 else 0) for idx in range(val_count)),
        xyxys=tuple((idx * 8, idx * 8, idx * 8 + 8, idx * 8 + 8) for idx in range(val_count)),
        groups=tuple(idx % 2 for idx in range(val_count)),
    )
    return train, val


def _patch_data_from_template(
    *,
    train: InMemoryPatchSamples,
    val: InMemoryPatchSamples,
    valid_batch_size: int | None,
    extras: dict[str, object] | None = None,
):
    return InMemoryPatchDataRecipe(
        train=train,
        val=val,
        in_channels=int(ERM_EXPERIMENT.data.in_channels),
        patch_size=int(ERM_EXPERIMENT.data.patch_size),
        train_batch_size=1,
        valid_batch_size=valid_batch_size,
        normalization=ERM_EXPERIMENT.data.normalization,
        extras={} if extras is None else dict(extras),
    )


def _write_zarr_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(path), mode="w", shape=tuple(array.shape), dtype=array.dtype)
    store[:] = array


class _FixedLogitModel:
    def __init__(self, logits: torch.Tensor):
        self._logits = logits

    def __call__(self, _x):
        return self._logits.clone()


def _assemble_patch_training(experiment, bundle=None):
    if experiment.trainer is None:
        experiment = replace(experiment, trainer=PatchTrainer())
    experiment = replace(
        experiment,
        model=required_build_recipe(experiment.model),
        objective=required_build_recipe(experiment.objective),
        runtime=required_build_recipe(experiment.runtime),
        augment=required_build_recipe(experiment.augment),
        evaluator=optional_build_recipe(experiment.evaluator),
        trainer=optional_build_recipe(experiment.trainer),
    )
    if bundle is None:
        bundle = build_experiment_data(
            experiment.data,
            runtime=experiment.runtime,
            augment=experiment.augment,
        )
    return assemble_experiment(experiment, bundle).trainer


def _with_patch_only_eval(experiment):
    return replace(
        experiment,
        evaluator=replace(ERM_EXPERIMENT.evaluator, stitch_inference=None, stitch=None),
    )


class ERMExperimentAssemblyTests(unittest.TestCase):
    def test_run_module_main_requires_explicit_experiment_name(self):
        with self.assertRaisesRegex(SystemExit, "usage: python -m ink.experiments.run"):
            experiment_run.main([])

    def test_run_module_main_delegates_to_named_erm_experiment(self):
        run_fs = object()
        with (
            mock.patch("ink.experiments.run.build_run_dir", return_value=Path("runs/erm_test")) as build_run_dir_mock,
            mock.patch("ink.experiments.run.RunFS", return_value=run_fs) as run_fs_mock,
            mock.patch("ink.experiments.run.run_experiment") as run_experiment_mock,
        ):
            experiment_run.main(["erm"])

        args, kwargs = run_experiment_mock.call_args
        self.assertEqual(args, (ERM_EXPERIMENT,))
        self.assertTrue(callable(kwargs["logger"]))
        self.assertIs(kwargs["run_fs"], run_fs)
        build_run_dir_mock.assert_called_once_with(Path("runs"), ERM_EXPERIMENT.name)
        run_fs_mock.assert_called_once_with(Path("runs/erm_test"), ERM_EXPERIMENT)

    def test_run_module_main_accepts_explicit_experiment_name(self):
        run_fs = object()
        with (
            mock.patch(
                "ink.experiments.run.build_run_dir",
                return_value=Path("runs/erm_stitch_inference_test"),
            ) as build_run_dir_mock,
            mock.patch("ink.experiments.run.RunFS", return_value=run_fs) as run_fs_mock,
            mock.patch("ink.experiments.run.run_experiment") as run_experiment_mock,
        ):
            experiment_run.main(["erm_stitch_inference"])

        args, kwargs = run_experiment_mock.call_args
        self.assertEqual(args, (ERM_STITCH_INFERENCE_EXPERIMENT,))
        self.assertTrue(callable(kwargs["logger"]))
        self.assertIs(kwargs["run_fs"], run_fs)
        build_run_dir_mock.assert_called_once_with(Path("runs"), ERM_STITCH_INFERENCE_EXPERIMENT.name)
        run_fs_mock.assert_called_once_with(
            Path("runs/erm_stitch_inference_test"),
            ERM_STITCH_INFERENCE_EXPERIMENT,
        )

    def test_run_module_main_accepts_patch_bundle_experiment_name(self):
        run_fs = object()
        with (
            mock.patch(
                "ink.experiments.run.build_run_dir",
                return_value=Path("runs/erm_patch_bundle_test"),
            ) as build_run_dir_mock,
            mock.patch("ink.experiments.run.RunFS", return_value=run_fs) as run_fs_mock,
            mock.patch("ink.experiments.run.run_experiment") as run_experiment_mock,
        ):
            experiment_run.main(["erm_patch_bundle"])

        args, kwargs = run_experiment_mock.call_args
        self.assertEqual(args, (ERM_PATCH_BUNDLE_EXPERIMENT,))
        self.assertTrue(callable(kwargs["logger"]))
        self.assertIs(kwargs["run_fs"], run_fs)
        build_run_dir_mock.assert_called_once_with(Path("runs"), ERM_PATCH_BUNDLE_EXPERIMENT.name)
        run_fs_mock.assert_called_once_with(Path("runs/erm_patch_bundle_test"), ERM_PATCH_BUNDLE_EXPERIMENT)

    def test_run_module_main_rejects_unknown_experiment_name(self):
        with self.assertRaisesRegex(SystemExit, "unknown experiment"):
            experiment_run.main(["erm_stitch_only"])

    def test_erm_experiment_template_uses_declarative_slots(self):
        self.assertEqual(ERM_EXPERIMENT.name, "erm")
        self.assertIsInstance(ERM_EXPERIMENT.data, ZarrPatchDataRecipe)
        self.assertEqual(
            ERM_EXPERIMENT.data.dataset_root,
            "/pscratch/cpa232_scrollprize/data/philodemos/2um_dataset/villa/ink-detection/ink-0309",
        )
        self.assertEqual(ERM_EXPERIMENT.data.patch_size, 256)
        self.assertEqual(ERM_EXPERIMENT.data.tile_size, 256)
        self.assertEqual(ERM_EXPERIMENT.data.stride, 128)
        self.assertEqual(ERM_EXPERIMENT.data.num_workers, 12)
        self.assertTrue(ERM_EXPERIMENT.data.shuffle)
        self.assertEqual(
            ERM_EXPERIMENT.data.patch_index_cache_dir,
            "/pscratch/cpa232_scrollprize/data/philodemos/2um_dataset/villa/ink-detection/ink-0309/.patch_index_cache/erm",
        )
        self.assertTrue(ERM_EXPERIMENT.data.cache_train_patches_in_memory)
        self.assertFalse(ERM_EXPERIMENT.data.include_train_valid_mask)
        self.assertEqual(len(ERM_EXPERIMENT.data.train_segment_ids), 16)
        self.assertEqual(
            ERM_EXPERIMENT.data.val_segment_ids,
            ("auto_grown_20250919055754487_inp_hr_2um",),
        )
        self.assertIsInstance(ERM_EXPERIMENT.data.normalization, ClipMaxDiv255Normalization)
        self.assertIsInstance(ERM_EXPERIMENT.model, ResNet3D)
        self.assertIsInstance(ERM_EXPERIMENT.runtime, TrainRuntime)
        self.assertIsInstance(ERM_EXPERIMENT.runtime.optimizer, AdamWOptimizer)
        self.assertIsInstance(ERM_EXPERIMENT.runtime.scheduler, OneCycleScheduler)
        self.assertTrue(ERM_EXPERIMENT.runtime.wandb.enabled)
        self.assertEqual(ERM_EXPERIMENT.runtime.wandb.project, "ink-2um-experiments")
        self.assertIsInstance(ERM_EXPERIMENT.augment, TrainAugment)
        self.assertIsInstance(ERM_EXPERIMENT.objective, ERMObjective)
        self.assertIsInstance(ERM_EXPERIMENT.evaluator, ValidationEvaluator)
        self.assertIsInstance(ERM_EXPERIMENT.evaluator.patch, PatchEval)
        self.assertIsInstance(ERM_EXPERIMENT.evaluator.stitch, StitchEval)
        self.assertIsInstance(ERM_EXPERIMENT.evaluator.stitch_inference, StitchInferenceRecipe)
        self.assertNotIn("stitch", ERM_EXPERIMENT.data.extras)
        self.assertEqual(ERM_EXPERIMENT.trainer.log_every_n_steps, 100)
        self.assertEqual(ERM_EXPERIMENT.trainer.save_every_n_epochs, 1)
        self.assertIs(ERM_EXPERIMENT.trainer.stitch_runtime, ERM_EXPERIMENT.evaluator.stitch_inference.stitch_runtime)
        self.assertIsInstance(ERM_EXPERIMENT.evaluator.stitch_inference.stitch_runtime, StitchRuntimeRecipe)
        stitch_cfg = ERM_EXPERIMENT.evaluator.stitch_inference.stitch_runtime.config
        self.assertTrue(stitch_cfg["train"]["viz"]["enabled"])
        self.assertEqual(stitch_cfg["train"]["viz"]["every_n_epochs"], 10)
        self.assertNotIn("segment_ids", stitch_cfg["train"]["viz"])

    def test_erm_stitch_inference_experiment_template_uses_stitch_inference_trainer(self):
        self.assertEqual(ERM_STITCH_INFERENCE_EXPERIMENT.name, "erm_stitch_inference")
        self.assertIs(ERM_STITCH_INFERENCE_EXPERIMENT.data, ERM_EXPERIMENT.data)
        self.assertIs(ERM_STITCH_INFERENCE_EXPERIMENT.model, ERM_EXPERIMENT.model)
        self.assertIs(ERM_STITCH_INFERENCE_EXPERIMENT.runtime, ERM_EXPERIMENT.runtime)
        self.assertIs(ERM_STITCH_INFERENCE_EXPERIMENT.augment, ERM_EXPERIMENT.augment)
        self.assertIsNone(ERM_STITCH_INFERENCE_EXPERIMENT.loss)
        self.assertIsNone(ERM_STITCH_INFERENCE_EXPERIMENT.objective)
        self.assertIsNone(ERM_STITCH_INFERENCE_EXPERIMENT.evaluator)
        self.assertIsInstance(ERM_STITCH_INFERENCE_EXPERIMENT.trainer, StitchInferenceTrainer)
        self.assertIsInstance(ERM_STITCH_INFERENCE_EXPERIMENT.trainer.stitch_inference, StitchInferenceRecipe)

    def test_erm_patch_bundle_experiment_template_reuses_erm_except_for_data(self):
        self.assertEqual(ERM_PATCH_BUNDLE_EXPERIMENT.name, "erm_patch_bundle")
        self.assertIsInstance(ERM_PATCH_BUNDLE_EXPERIMENT.data, GeneratedPatchBundleDataRecipe)
        self.assertEqual(ERM_PATCH_BUNDLE_EXPERIMENT.data.bundle_root, ".tmp/patch_bundles/erm")
        self.assertIs(ERM_PATCH_BUNDLE_EXPERIMENT.data.source, ERM_EXPERIMENT.data)
        self.assertIs(ERM_PATCH_BUNDLE_EXPERIMENT.model, ERM_EXPERIMENT.model)
        self.assertIs(ERM_PATCH_BUNDLE_EXPERIMENT.runtime, ERM_EXPERIMENT.runtime)
        self.assertIs(ERM_PATCH_BUNDLE_EXPERIMENT.augment, ERM_EXPERIMENT.augment)
        self.assertIs(ERM_PATCH_BUNDLE_EXPERIMENT.trainer, ERM_EXPERIMENT.trainer)
        self.assertIs(ERM_PATCH_BUNDLE_EXPERIMENT.evaluator, ERM_EXPERIMENT.evaluator)

    def test_erm_experiment_can_be_replaced_with_real_model_recipe(self):
        train, val = _patch_samples(train_count=4, val_count=2)
        experiment = replace(
            ERM_EXPERIMENT,
            name="erm_bound",
            data=InMemoryPatchDataRecipe(
                train=train,
                val=val,
                in_channels=62,
                patch_size=256,
                train_batch_size=1,
                valid_batch_size=2,
                normalization=ERM_EXPERIMENT.data.normalization,
                extras={"group_counts": [3, 4]},
            ),
            model=ResNet3D(pretrained=False),
        )

        training = _assemble_patch_training(_with_patch_only_eval(experiment))

        self.assertEqual(training.experiment.name, "erm_bound")
        self.assertIsInstance(training.experiment.runtime, TrainRuntime)
        self.assertEqual(training.experiment.data.patch_size, 256)
        self.assertIsInstance(training.experiment.data.normalization, ClipMaxDiv255Normalization)
        self.assertEqual(training.experiment.runtime.steps_per_epoch, 4)
        self.assertEqual(training.experiment.runtime.precision, "16-mixed")
        self.assertIsInstance(training.experiment.augment, TrainAugment)
        self.assertIsInstance(experiment.model, ResNet3D)
        self.assertIsInstance(training.experiment.model, ResNet3DSegmentationModel)
        self.assertIsInstance(training.experiment.evaluator, ValidationEvaluator)
        self.assertIsInstance(training.experiment.evaluator.patch, PatchEval)
        self.assertEqual(len(training.experiment.evaluator.patch.metrics), 4)
        metric_names = [getattr(metric, "name", None) for metric in training.experiment.evaluator.patch.metrics]
        self.assertIn("Dice", metric_names)
        self.assertIn("BalancedAccuracy", metric_names)
        self.assertIn("Dice_thr_96_255", metric_names)
        self.assertIn("BalancedAccuracy_thr_96_255", metric_names)
        self.assertIsInstance(training.experiment.evaluator.patch.metrics[0], Dice)
        self.assertIsInstance(training.experiment.evaluator.patch.metrics[1], BalancedAccuracy)
        self.assertEqual(training.experiment.evaluator.patch.n_groups, 2)
        self.assertIsInstance(training.train_loader, DataLoader)
        self.assertIsInstance(training.eval_loader, DataLoader)

    def test_erm_experiment_accepts_zarr_backed_data_recipe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(
                segment_dir / "segA.zarr",
                np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
            )
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", np.full((8, 8), 255, dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((8, 8), 255, dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_validation_mask.zarr", np.full((8, 8), 255, dtype=np.uint8))

            data = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={
                    "segA": {},
                },
                train_segment_ids=("segA",),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
            )

            experiment = replace(
                ERM_EXPERIMENT,
                name="erm_zarr",
                data=data,
                model=ResNet3D(pretrained=False),
            )
            training = _assemble_patch_training(experiment)

            self.assertIs(training.experiment.data, data)
            self.assertIsInstance(training.train_loader, DataLoader)
            self.assertIsInstance(training.eval_loader, DataLoader)
            self.assertIsNone(training.experiment.evaluator.patch.n_groups)
            self.assertIsNotNone(training.experiment.evaluator.stitch_inference)
            self.assertIsNotNone(training.experiment.evaluator.stitch)
            self.assertEqual(len(training.train_loader), 1)
            self.assertEqual(len(training.eval_loader), 1)

    def test_assemble_training_runtime_accepts_runtime_override(self):
        train, val = _patch_samples(train_count=5, val_count=2)
        experiment = replace(
            ERM_EXPERIMENT,
            name="erm_runtime_override",
            data=_patch_data_from_template(
                train=train,
                val=val,
                valid_batch_size=2,
                extras={},
            ),
            model=ResNet3D(pretrained=False),
        )
        experiment = replace(experiment, data=replace(experiment.data, extras={"group_counts": [3, 4]}))
        training = _assemble_patch_training(
            _with_patch_only_eval(
                replace(
                    experiment,
                    runtime=TrainRuntime(
                        use_amp=False,
                        grad_accum=2,
                        grad_clip_norm=5.0,
                        optimizer=SGDOptimizer(lr=1e-3, momentum=0.85, nesterov=True),
                        scheduler=CosineScheduler(warmup_pct=0.25, min_lr=1e-5, warmup_factor=5.0),
                    ),
                )
            )
        )

        self.assertIsInstance(training.experiment.runtime, TrainRuntime)
        self.assertEqual(training.experiment.runtime.precision, "32-true")
        self.assertEqual(training.experiment.runtime.steps_per_epoch, 3)
        self.assertEqual(training.experiment.runtime.grad_accum, 2)
        self.assertEqual(training.experiment.runtime.grad_clip_norm, 5.0)
        self.assertIsInstance(training.experiment.runtime.optimizer, SGDOptimizer)
        self.assertIsInstance(training.experiment.runtime.scheduler, CosineScheduler)

    def test_assemble_training_runtime_accepts_preconfigured_experiment(self):
        train, val = _patch_samples(train_count=4, val_count=2)
        experiment = replace(
            ERM_EXPERIMENT,
            name="erm_preconfigured",
            data=_patch_data_from_template(
                train=train,
                val=val,
                valid_batch_size=2,
                extras={},
            ),
            model=ResNet3D(pretrained=False),
        )
        experiment = replace(experiment, data=replace(experiment.data, extras={"group_counts": [3, 4]}))

        training = _assemble_patch_training(_with_patch_only_eval(experiment))

        self.assertEqual(training.experiment.name, "erm_preconfigured")
        self.assertEqual(training.experiment.data.patch_size, 256)
        self.assertIsInstance(training.experiment.data.normalization, ClipMaxDiv255Normalization)
        self.assertIsInstance(training.experiment.model, ResNet3DSegmentationModel)
        self.assertIsInstance(training.experiment.loss, DiceBCEBatch)
        self.assertIsInstance(training.experiment.objective, ERMObjective)
        self.assertEqual(training.experiment.evaluator.patch.n_groups, 2)
        self.assertEqual(len(training.train_loader), 4)
        self.assertEqual(len(training.eval_loader), 1)

    def test_assemble_training_runtime_accepts_recipe_override(self):
        train, val = _patch_samples(train_count=4, val_count=2)
        experiment = replace(
            ERM_EXPERIMENT,
            name="erm_template",
            data=_patch_data_from_template(
                train=train,
                val=val,
                valid_batch_size=2,
                extras={},
            ),
        )
        experiment = replace(experiment, data=replace(experiment.data, extras={"group_counts": [3, 4]}))
        training = _assemble_patch_training(
            _with_patch_only_eval(
                replace(
                    experiment,
                    name="erm",
                    model=ResNet3D(pretrained=False),
                )
            )
        )

        self.assertEqual(training.experiment.name, "erm")
        self.assertIsInstance(training.train_loader, DataLoader)
        self.assertIsInstance(training.eval_loader, DataLoader)
        self.assertIsInstance(training.experiment.data, InMemoryPatchDataRecipe)
        self.assertIsInstance(training.experiment.loss, DiceBCEBatch)
        self.assertIsInstance(training.experiment.objective, ERMObjective)
        self.assertIsInstance(training.experiment.runtime, TrainRuntime)
        self.assertIsInstance(training.experiment.augment, TrainAugment)
        self.assertIsInstance(training.experiment.model, ResNet3DSegmentationModel)
        self.assertEqual(training.experiment.evaluator.patch.n_groups, 2)
        self.assertEqual(len(training.train_loader), 4)
        self.assertEqual(len(training.eval_loader), 1)

    def test_assemble_training_runtime_accepts_zarr_data_recipe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(
                segment_dir / "segA.zarr",
                np.stack([np.full((8, 8), value, dtype=np.uint8) for value in (10, 20, 30, 40)], axis=0),
            )
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", np.full((8, 8), 255, dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((8, 8), 255, dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_validation_mask.zarr", np.full((8, 8), 255, dtype=np.uint8))

            data = ZarrPatchDataRecipe(
                dataset_root=str(root),
                segments={
                    "segA": {},
                },
                train_segment_ids=("segA",),
                val_segment_ids=("segA",),
                in_channels=4,
                patch_size=8,
                train_batch_size=1,
                valid_batch_size=1,
                shuffle=False,
            )

            experiment = replace(
                ERM_EXPERIMENT,
                name="erm_zarr_runtime",
                model=ResNet3D(pretrained=False),
                data=data,
            )
            training = _assemble_patch_training(experiment)

            self.assertIs(training.experiment.data, data)
            self.assertIsNone(training.experiment.evaluator.patch.n_groups)
            self.assertIsNotNone(training.experiment.evaluator.stitch_inference)
            self.assertIsNotNone(training.experiment.evaluator.stitch)
            self.assertEqual(len(training.train_loader), 1)
            self.assertEqual(len(training.eval_loader), 1)


class ERMTrainingStepContractTests(unittest.TestCase):
    def test_run_training_step_uses_training_runtime_contract(self):
        logits = torch.tensor(
            [
                [[[0.2, -0.4], [1.1, 0.0]]],
                [[[-1.0, 0.8], [0.3, -0.7]]],
                [[[0.5, -0.2], [-0.1, 0.4]]],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [1.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        group_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        batch = Batch(
            x=torch.zeros_like(logits),
            y=targets,
            meta=BatchMeta(segment_ids=["segA", "segB", "segC"], group_idx=group_idx),
        )
        train, val = _patch_samples(train_count=1, val_count=1, in_channels=62)
        experiment = replace(
            ERM_EXPERIMENT,
            name="erm_batch",
            data=_patch_data_from_template(
                train=train,
                val=val,
                valid_batch_size=1,
                extras={"group_counts": [3, 4, 5]},
            ),
            model=_FixedLogitModel(logits),
        )
        training = _assemble_patch_training(_with_patch_only_eval(experiment))

        step = training.run_step(batch)

        torch.testing.assert_close(step.loss, step.primary_loss)
        torch.testing.assert_close(step.components["train/loss"], step.loss)
        self.assertIn("train/DiceBCEBatch", step.components)
        self.assertIn("train/dice_loss", step.components)
        self.assertIn("train/bce_loss", step.components)


if __name__ == "__main__":
    unittest.main()
