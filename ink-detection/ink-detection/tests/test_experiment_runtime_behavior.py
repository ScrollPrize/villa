from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import zarr

from ink.core import Batch, BatchMeta, DataBundle, Experiment, assemble_experiment, build_experiment_data, run_experiment
from ink.recipes.data.layout import NestedZarrLayout
from ink.recipes.eval import PatchEval, StitchEval, ValidationEvaluator
from ink.recipes.metrics import Dice
from ink.recipes.objectives import ERMGroupTopK, ERMObjective, GroupDROComputer, GroupDROObjective
from ink.recipes.stitch import (
    StitchInference,
    StitchInferenceRecipe,
    StitchRuntime,
    StitchRuntimeRecipe,
    ZarrStitchStore,
)
from ink.recipes.trainers import PatchTrainer, StitchInferenceTrainer
from tests.support.build_recipes import StaticBuildRecipe


def _bundle(*, train_loader="train", eval_loader="val", augment="bound_augment", group_counts=None, extras=None) -> DataBundle:
    return DataBundle(
        train_loader=train_loader,
        eval_loader=eval_loader,
        in_channels=62,
        augment=augment,
        group_counts=group_counts,
        extras={} if extras is None else dict(extras),
    )


class _StaticLoader(list):
    def __init__(self, batches, *, dataset):
        super().__init__(batches)
        self.dataset = dataset


def _write_zarr_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(path), mode="w", shape=tuple(array.shape), dtype=array.dtype)
    store[:] = array


class ExperimentRuntimeBindingTests(unittest.TestCase):
    def test_assemble_experiment_uses_data_bound_augment_for_runtime_and_model(self):
        calls = []

        class RuntimeRecipe:
            def build(self, *, data, augment):
                calls.append(("runtime", data, augment))
                return SimpleNamespace(kind="runtime", data=data, augment=augment)

        class ModelRecipe:
            def build(self, *, data, runtime, augment):
                calls.append(("model", data, runtime, augment))
                return SimpleNamespace(kind="model", data=data, runtime=runtime, augment=augment)

        bound_augment = SimpleNamespace(kind="augment")
        experiment = Experiment(
            name="runtime_model_augment",
            data="data",
            model=ModelRecipe(),
            loss="loss",
            objective=ERMObjective(),
            runtime=RuntimeRecipe(),
            augment=object(),
        )

        bound = assemble_experiment(experiment, _bundle(augment=bound_augment))

        self.assertEqual([entry[0] for entry in calls], ["runtime", "model"])
        self.assertEqual(bound.runtime.kind, "runtime")
        self.assertIs(bound.augment, bound_augment)
        self.assertEqual(bound.model.kind, "model")
        self.assertIs(bound.runtime.augment, bound_augment)
        self.assertIs(bound.model.runtime, bound.runtime)
        self.assertIs(bound.model.augment, bound_augment)

    def test_assemble_experiment_allows_explicit_none_augment_in_data_bundle(self):
        experiment = Experiment(
            name="missing_bound_augment",
            data="data",
            model=StaticBuildRecipe(SimpleNamespace(kind="model")),
            loss="loss",
            objective=ERMObjective(),
            runtime=StaticBuildRecipe(SimpleNamespace(kind="runtime")),
            augment=object(),
        )
        bundle = DataBundle(
            train_loader="train",
            eval_loader="val",
            in_channels=62,
            augment=None,
            extras={},
        )

        bound = assemble_experiment(experiment, bundle)

        self.assertIsNone(bound.augment)
        self.assertEqual(bound.runtime.kind, "runtime")
        self.assertEqual(bound.model.kind, "model")

    def test_assemble_experiment_requires_buildable_runtime_and_model(self):
        experiment = Experiment(
            name="missing_build_contract",
            data="data",
            model=object(),
            loss="loss",
            objective=ERMObjective(),
            runtime=object(),
            augment=object(),
        )

        with self.assertRaises(AttributeError):
            assemble_experiment(experiment, _bundle())

    def test_assemble_experiment_builds_group_dro_objective(self):
        experiment = Experiment(
            name="groupdro",
            data="data",
            model=StaticBuildRecipe("model"),
            loss="loss",
            objective=GroupDROObjective(step_size=0.2, gamma=0.3, btl=True, alpha=0.5),
            runtime=StaticBuildRecipe("runtime"),
            augment=StaticBuildRecipe("augment"),
        )

        bound = assemble_experiment(experiment, _bundle(group_counts=[3, 7]))

        self.assertIsInstance(bound.objective, GroupDROComputer)
        self.assertAlmostEqual(bound.objective.step_size, 0.2)
        self.assertAlmostEqual(bound.objective.gamma, 0.3)
        self.assertTrue(bound.objective.btl)
        self.assertIsInstance(experiment.objective, GroupDROObjective)

    def test_assemble_experiment_builds_objective_from_generic_recipe_contract(self):
        class ObjectiveRecipe:
            def build(self, bundle):
                return SimpleNamespace(kind="objective", bundle=bundle)

        experiment = Experiment(
            name="generic_objective",
            data="data",
            model=StaticBuildRecipe("model"),
            loss="loss",
            objective=ObjectiveRecipe(),
            runtime=StaticBuildRecipe("runtime"),
            augment=StaticBuildRecipe("augment"),
        )

        bundle = _bundle()
        bound = assemble_experiment(experiment, bundle)

        self.assertEqual(bound.objective.kind, "objective")
        self.assertIs(bound.objective.bundle, bundle)
        self.assertIsInstance(experiment.objective, ObjectiveRecipe)

    def test_assemble_experiment_builds_evaluator_from_generic_recipe_contract(self):
        class EvaluatorRecipe:
            def build(self, *, data, runtime=None, logger=None):
                return SimpleNamespace(
                    kind="evaluator",
                    bundle=data,
                    runtime=runtime,
                    logger=logger,
                )

        logger = object()
        experiment = Experiment(
            name="generic_evaluator",
            data="data",
            model=StaticBuildRecipe("model"),
            loss="loss",
            objective=ERMObjective(),
            runtime=StaticBuildRecipe("runtime"),
            augment=StaticBuildRecipe("augment"),
            evaluator=EvaluatorRecipe(),
        )

        bundle = _bundle()
        bound = assemble_experiment(experiment, bundle, logger=logger)

        self.assertEqual(bound.evaluator.kind, "evaluator")
        self.assertIs(bound.evaluator.bundle, bundle)
        self.assertEqual(bound.evaluator.runtime, "runtime")
        self.assertIs(bound.evaluator.logger, logger)

    def test_assemble_experiment_binds_validation_patch_stage_metrics(self):
        experiment = Experiment(
            name="patch_evaluator",
            data="data",
            model=StaticBuildRecipe("model"),
            loss="loss",
            objective=ERMObjective(),
            runtime=StaticBuildRecipe("runtime"),
            augment=StaticBuildRecipe("augment"),
            evaluator=ValidationEvaluator(
                patch=PatchEval(
                    metrics=(
                        Dice(),
                    ),
                )
            ),
        )

        bound = assemble_experiment(experiment, _bundle(group_counts=[3, 7]))

        self.assertIsInstance(bound.evaluator, ValidationEvaluator)
        self.assertIsInstance(bound.evaluator.patch, PatchEval)
        self.assertEqual(len(bound.evaluator.patch.metrics), 1)
        self.assertIsInstance(bound.evaluator.patch.metrics[0], Dice)
        self.assertEqual(bound.evaluator.patch.n_groups, 2)

    def test_assemble_experiment_builds_validation_patch_metric_from_generic_recipe_contract(self):
        class MetricRecipe:
            def build(self, *, data, runtime=None, logger=None):
                built = MetricRecipe()
                built.bundle = data
                built.runtime = runtime
                built.logger = logger
                return built

        metric = MetricRecipe()
        experiment = Experiment(
            name="patch_metric_build_contract",
            data="data",
            model=StaticBuildRecipe("model"),
            loss="loss_recipe",
            objective=ERMObjective(),
            runtime=StaticBuildRecipe("runtime"),
            augment=StaticBuildRecipe("augment"),
            evaluator=ValidationEvaluator(
                patch=PatchEval(
                    metrics=(metric,),
                )
            ),
        )

        bundle = _bundle()
        bound = assemble_experiment(experiment, bundle)

        self.assertIs(bound.evaluator.patch.metrics[0].bundle, bundle)
        self.assertEqual(bound.evaluator.patch.metrics[0].runtime, "runtime")

    def test_validation_evaluator_builds_stitch_runtime_once_from_owned_recipe(self):
        class CountingStitchRecipe(StitchRuntimeRecipe):
            def __init__(self, *, config=None):
                super().__init__(config=config)
                object.__setattr__(self, "calls", [])

            def build(self, bundle, *, runtime=None, logger=None, patch_loss=None):
                self.calls.append(
                    {
                        "bundle": bundle,
                        "runtime": runtime,
                        "logger": logger,
                        "patch_loss": patch_loss,
                    }
                )
                return super().build(bundle, runtime=runtime, logger=logger, patch_loss=patch_loss)

        recipe = CountingStitchRecipe(
            config={
                "eval": {
                    "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                },
            }
        )
        logger = object()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", np.zeros((4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((4, 4), 255, dtype=np.uint8))
            bundle = DataBundle(
                train_loader=[],
                eval_loader=_StaticLoader([], dataset=SimpleNamespace(layout=NestedZarrLayout(root))),
                in_channels=1,
                augment=None,
            )

            evaluator = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=recipe,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(metrics=(Dice(),)),
            ).build(data=bundle, runtime="runtime", logger=logger)

        self.assertEqual(len(recipe.calls), 1)
        self.assertIs(recipe.calls[0]["bundle"], bundle)
        self.assertEqual(recipe.calls[0]["runtime"], "runtime")
        self.assertIs(recipe.calls[0]["logger"], logger)
        self.assertIsNone(recipe.calls[0]["patch_loss"])
        self.assertIsInstance(evaluator.stitch_inference.stitch_runtime, StitchRuntime)
        self.assertIs(evaluator.stitch._inference, evaluator.stitch_inference)

    def test_stitch_trainer_builds_stitch_runtime_once_from_owned_recipe(self):
        class CountingStitchRecipe(StitchRuntimeRecipe):
            def __init__(self, *, config=None):
                super().__init__(config=config)
                object.__setattr__(self, "calls", [])

            def build(self, bundle, *, runtime=None, logger=None, patch_loss=None):
                self.calls.append(
                    {
                        "bundle": bundle,
                        "runtime": runtime,
                        "logger": logger,
                        "patch_loss": patch_loss,
                    }
                )
                return super().build(bundle, runtime=runtime, logger=logger, patch_loss=patch_loss)

        recipe = CountingStitchRecipe(
            config={
                "eval": {
                    "segments": [{"segment_id": "segA", "shape": (8, 12)}],
                },
            }
        )
        logger = object()
        experiment = Experiment(
            name="stitch_trainer",
            data="data",
            model=StaticBuildRecipe("model"),
            loss="loss",
            objective=ERMObjective(),
            runtime=StaticBuildRecipe("runtime"),
            augment=StaticBuildRecipe("augment"),
            trainer=StitchInferenceTrainer(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=recipe,
                    store=ZarrStitchStore(root_dir=Path(".tmp") / "stitch_eval"),
                ),
            ),
        )
        bundle = _bundle(
            eval_loader=[],
            augment="bound_augment",
        )
        experiment = replace(
            experiment,
            trainer=StitchInferenceTrainer(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=recipe,
                    store=ZarrStitchStore(root_dir=Path(".tmp") / "stitch_eval"),
                ),
            ),
        )

        bound = assemble_experiment(experiment, bundle, logger=logger)

        self.assertEqual(len(recipe.calls), 1)
        self.assertIs(recipe.calls[0]["bundle"], bundle)
        self.assertEqual(recipe.calls[0]["runtime"], "runtime")
        self.assertIs(recipe.calls[0]["logger"], logger)
        self.assertEqual(recipe.calls[0]["patch_loss"], "loss")
        self.assertIsInstance(bound.trainer.stitch_inference.stitch_runtime, StitchRuntime)
        self.assertEqual(bound.trainer.inference_loader, bundle.eval_loader)

    def test_patch_eval_requires_metrics(self):
        with self.assertRaisesRegex(ValueError, "at least one metric"):
            PatchEval().build(data=_bundle())

    def test_group_dro_recipe_requires_group_counts_from_data_bundle(self):
        objective = GroupDROObjective(step_size=0.2)

        with self.assertRaisesRegex(ValueError, "group_counts"):
            objective.build(_bundle(augment="bound_augment"))

    def test_patch_trainer_builds_trainer_owned_stitch_runtime_once_from_recipe(self):
        class CountingStitchRecipe(StitchRuntimeRecipe):
            def __init__(self, *, config=None):
                super().__init__(config=config)
                object.__setattr__(self, "calls", [])

            def build(self, bundle, *, runtime=None, logger=None, patch_loss=None):
                self.calls.append(
                    {
                        "bundle": bundle,
                        "runtime": runtime,
                        "logger": logger,
                        "patch_loss": patch_loss,
                    }
                )
                return super().build(bundle, runtime=runtime, logger=logger, patch_loss=patch_loss)

        recipe = CountingStitchRecipe(
            config={
                "eval": {
                    "segments": [{"segment_id": "segA", "shape": (8, 12)}],
                },
                "train": {
                    "segments": [{"segment_id": "segA", "shape": (8, 12)}],
                    "viz": {"enabled": True},
                },
            }
        )
        logger = object()
        experiment = Experiment(
            name="patch_trainer_stitch_support",
            data="data",
            model=StaticBuildRecipe("model"),
            loss="loss",
            objective=ERMObjective(),
            runtime=StaticBuildRecipe("runtime"),
            augment=StaticBuildRecipe("augment"),
            trainer=PatchTrainer(
                stitch_runtime=recipe,
            ),
        )

        bound = assemble_experiment(experiment, _bundle(train_loader=[], eval_loader=[]), logger=logger)

        self.assertEqual(len(recipe.calls), 1)
        self.assertEqual(recipe.calls[0]["runtime"], "runtime")
        self.assertIs(recipe.calls[0]["logger"], logger)
        self.assertEqual(recipe.calls[0]["patch_loss"], "loss")
        self.assertIsInstance(bound.trainer.stitch_runtime, StitchRuntime)

    def test_erm_objective_build_returns_same_recipe_object(self):
        objective = ERMGroupTopK(group_topk=2)
        bundle = _bundle()

        self.assertIs(objective.build(bundle), objective)


class ExperimentDataContractTests(unittest.TestCase):
    def test_build_experiment_data_requires_explicit_data_contract(self):
        with self.assertRaises(AttributeError):
            build_experiment_data(object(), runtime="runtime", augment="augment")

    def test_build_experiment_data_requires_recipe_to_return_data_bundle(self):
        class BadRecipe:
            def build(self, *, runtime=None, augment=None):
                return {"runtime": runtime, "augment": augment}

        with self.assertRaises(AssertionError):
            build_experiment_data(BadRecipe(), runtime="runtime", augment="augment")

    def test_run_experiment_builds_data_assembles_and_runs_trainer(self):
        calls = []

        class DataRecipe:
            def build(self, *, runtime=None, augment=None):
                del runtime, augment
                return _bundle()

        class RuntimeRecipe:
            def build(self, *, data, augment):
                calls.append(("runtime", data, augment))
                return SimpleNamespace(kind="runtime", augment=augment)

        class ModelRecipe:
            def build(self, *, data, runtime, augment):
                calls.append(("model", data, runtime, augment))
                return SimpleNamespace(kind="model", runtime=runtime, augment=augment)

        class TrainerRecipe:
            def build(self, *, experiment, data, logger=None):
                del logger
                calls.append(("trainer", experiment, data))
                return SimpleNamespace(run=lambda **kwargs: ("ran", kwargs, experiment, data))

        experiment = Experiment(
            name="runner",
            data=DataRecipe(),
            model=ModelRecipe(),
            loss="loss",
            objective=ERMObjective(),
            runtime=RuntimeRecipe(),
            augment=StaticBuildRecipe("augment_recipe"),
            trainer=TrainerRecipe(),
        )

        status, kwargs, bound_experiment, bundle = run_experiment(experiment, device="cpu")

        self.assertEqual(status, "ran")
        self.assertEqual(kwargs, {"device": "cpu"})
        self.assertEqual([entry[0] for entry in calls], ["runtime", "model", "trainer"])
        self.assertEqual(bound_experiment.augment, "bound_augment")
        self.assertEqual(bound_experiment.runtime.augment, "bound_augment")
        self.assertEqual(bound_experiment.model.augment, "bound_augment")
        self.assertEqual(bundle.train_loader, "train")

    def test_run_experiment_passes_logger_to_trainer_build(self):
        messages = []

        class DataRecipe:
            def build(self, *, runtime=None, augment=None):
                del runtime, augment
                return _bundle()

        class TrainerRecipe:
            def build(self, *, experiment, data, logger=None):
                del experiment, data
                return SimpleNamespace(run=lambda **kwargs: ("ran", kwargs, logger))

        experiment = Experiment(
            name="runner_with_logs",
            data=DataRecipe(),
            model=StaticBuildRecipe(SimpleNamespace(kind="model")),
            loss="loss",
            objective=ERMObjective(),
            runtime=StaticBuildRecipe(SimpleNamespace(kind="runtime")),
            augment=StaticBuildRecipe("augment_recipe"),
            trainer=TrainerRecipe(),
        )

        status, kwargs, passed_logger = run_experiment(experiment, logger=messages.append, device="cpu")

        self.assertEqual(status, "ran")
        self.assertEqual(kwargs, {"device": "cpu"})
        self.assertTrue(callable(passed_logger))

    def test_run_experiment_supports_stitch_inference_only_workflow(self):
        logits = torch.full((1, 1, 4, 4), 10.0, dtype=torch.float32)
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 4, 4]], dtype=torch.long),
            ),
        )

        class _FixedLogitModel:
            def __call__(self, _x):
                return logits.clone()

        class _Runtime:
            def __init__(self):
                self.init_ckpt_path = None
                self.resume_ckpt_path = None
                self.wandb = None
                self.precision_context_calls = []

            def precision_context(self, *, device=None):
                self.precision_context_calls.append(device)
                return nullcontext()

        runtime_obj = _Runtime()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", np.zeros((4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((4, 4), 255, dtype=np.uint8))

            class DataRecipe:
                def build(self, *, runtime=None, augment=None):
                    del runtime, augment
                    layout = NestedZarrLayout(root)
                    return DataBundle(
                        train_loader=[],
                        eval_loader=_StaticLoader([batch], dataset=SimpleNamespace(layout=layout)),
                        in_channels=1,
                        augment=None,
                        extras={},
                    )

            run = run_experiment(
                Experiment(
                    name="stitch_inference_only",
                    data=DataRecipe(),
                    model=StaticBuildRecipe(_FixedLogitModel()),
                    loss=None,
                    objective=None,
                    runtime=StaticBuildRecipe(runtime_obj),
                    augment=StaticBuildRecipe(None),
                    trainer=StitchInferenceTrainer(
                        stitch_inference=StitchInferenceRecipe(
                            stitch_runtime=StitchRuntimeRecipe(
                                config={
                                    "downsample": 1,
                                    "eval": {
                                        "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                                    },
                                }
                            ),
                            store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                        ),
                    ),
                    evaluator=None,
                ),
            )

        self.assertEqual(run.batches, 1)
        self.assertEqual(len(runtime_obj.precision_context_calls), 1)
        self.assertTrue(str(run.store_root_dir).endswith("stitch_eval"))
        self.assertEqual(set((run.segment_prob_paths or {}).keys()), {"segA"})
        self.assertTrue(str((run.segment_prob_paths or {})["segA"]).endswith("segA__prob.zarr"))
        self.assertEqual(set((run.segment_preview_paths or {}).keys()), {"segA"})
        self.assertTrue(str((run.segment_preview_paths or {})["segA"]).endswith("segA__prob.png"))


if __name__ == "__main__":
    unittest.main()
