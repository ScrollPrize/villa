from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import zarr

from ink.core import Batch, BatchMeta, DataBundle, EvalReport, ModelOutputBatch
from ink.recipes.data.layout import NestedZarrLayout
from ink.recipes.eval import PatchEval, StitchEval, ValidationEvaluator
from ink.recipes.eval.stitch_prepared import skeletonize_binary as prepared_skeletonize_binary
from ink.recipes.eval import stitch_regions as stitch_regions_module
from ink.recipes.metrics import (
    BalancedAccuracy,
    Dice,
    DRD,
    MetricReport,
    PFMWeighted,
    StitchMetricBatch,
    flatten_eval_report,
)
from ink.recipes.stitch import StitchInference, StitchInferenceRecipe, StitchRuntime, StitchRuntimeRecipe, ZarrStitchStore
from ink.recipes.stitch.store import downsample_preview_for_media
from ink.recipes.trainers import StitchInferenceRun


class _FixedLogitModel:
    def __init__(self, logits: torch.Tensor):
        self._logits = logits

    def __call__(self, _x):
        return self._logits.clone()


class _CountingLogitModel(_FixedLogitModel):
    def __init__(self, logits: torch.Tensor):
        super().__init__(logits)
        self.calls = 0

    def __call__(self, x):
        del x
        self.calls += 1
        return self._logits.clone()


class _StaticLoader(list):
    def __init__(self, batches, *, dataset):
        super().__init__(batches)
        self.dataset = dataset


class _ComponentCountMetric:
    def empty_state(self, *, n_groups=None):
        del n_groups
        return 0

    def update(self, state, batch, *, shared=None):
        del batch, shared
        return int(state) + 1

    def finalize(self, state):
        return MetricReport(summary={"component_count": float(state)})


def _write_zarr_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(path), mode="w", shape=tuple(array.shape), dtype=array.dtype)
    store[:] = array


def _bound_stitch_inference(stitch, *, root_dir=None) -> StitchInference:
    layout = stitch.eval_segment_layout()
    store = ZarrStitchStore(root_dir=root_dir).build(
        segment_shapes=layout.segment_shapes,
        downsample=layout.downsample,
        segment_rois=layout.segment_rois,
    )
    return StitchInference(
        store=store,
        segment_shapes=layout.segment_shapes,
        stitch_runtime=stitch,
    )


class ConfusionMetricRecipeTests(unittest.TestCase):
    def test_thresholded_metric_names_are_derived_automatically(self):
        dice = Dice(threshold=96.0 / 255.0)
        ba = BalancedAccuracy(threshold=96.0 / 255.0)

        self.assertEqual(dice.name, "Dice_thr_96_255")
        self.assertEqual(ba.name, "BalancedAccuracy_thr_96_255")

    def test_dice_and_balanced_accuracy_compute_expected_scores(self):
        batch = ModelOutputBatch(
            logits=torch.tensor([[[[2.0, -2.0], [-0.3, 0.4]]]], dtype=torch.float32),
            targets=torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
        )
        dice = Dice()
        ba = BalancedAccuracy()

        dice_state = dice.update(dice.empty_state(), batch)
        ba_state = ba.update(ba.empty_state(), batch)

        self.assertAlmostEqual(dice.finalize(dice_state).summary["Dice"], 0.5)
        self.assertAlmostEqual(ba.finalize(ba_state).summary["BalancedAccuracy"], 0.5)

    def test_confusion_metrics_apply_valid_mask(self):
        batch = ModelOutputBatch(
            logits=torch.tensor([[[[2.0, -2.0], [-0.3, 0.4]]]], dtype=torch.float32),
            targets=torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32),
            valid_mask=torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]], dtype=torch.float32),
        )
        dice = Dice()
        ba = BalancedAccuracy()

        dice_state = dice.update(dice.empty_state(), batch)
        ba_state = ba.update(ba.empty_state(), batch)

        self.assertAlmostEqual(dice.finalize(dice_state).summary["Dice"], 2.0 / 3.0)
        self.assertAlmostEqual(ba.finalize(ba_state).summary["BalancedAccuracy"], 0.75)


class ValidationEvaluatorTests(unittest.TestCase):
    def test_validation_evaluator_owns_the_only_inference_loop(self):
        logits = torch.ones((1, 1, 2, 2), dtype=torch.float32)
        batch = Batch(
            x=torch.zeros_like(logits),
            y=torch.ones_like(logits),
            meta=BatchMeta(
                segment_ids=["segA"],
                valid_mask=torch.ones_like(logits),
                patch_xyxy=torch.tensor([[0, 0, 2, 2]], dtype=torch.long),
                group_idx=torch.tensor([0], dtype=torch.long),
            ),
        )

        class _PatchStage:
            def __init__(self):
                self.begin_calls = 0
                self.observe_calls = []
                self.finalize_calls = 0

            def begin_epoch(self):
                self.begin_calls += 1

            def observe_batch(self, eval_batch):
                self.observe_calls.append(eval_batch)

            def finalize_epoch(self):
                self.finalize_calls += 1
                return EvalReport(summary={"patch_metric": 1.0})

            def evaluate(self, *_args, **_kwargs):
                raise AssertionError("ValidationEvaluator must not delegate inference back to PatchEval.evaluate")

        class _StitchInferenceStage:
            def __init__(self):
                self.begin_calls = 0
                self.observe_calls = []

            def begin_epoch(self):
                self.begin_calls += 1

            def observe_batch(self, eval_batch):
                self.observe_calls.append(eval_batch)

            def evaluate(self, *_args, **_kwargs):
                raise AssertionError("ValidationEvaluator must not delegate inference back to StitchInference.evaluate")

        class _StitchStage:
            def __init__(self):
                self.finalize_calls = 0

            def finalize_epoch(self):
                self.finalize_calls += 1
                return EvalReport(summary={"stitch_metric": 2.0})

            def evaluate(self, *_args, **_kwargs):
                raise AssertionError("ValidationEvaluator must not delegate inference back to StitchEval.evaluate")

        patch = _PatchStage()
        stitch_inference = _StitchInferenceStage()
        stitch = _StitchStage()
        model = _CountingLogitModel(logits)

        report = ValidationEvaluator(
            patch=patch,
            stitch_inference=stitch_inference,
            stitch=stitch,
        ).evaluate(model, [batch])

        self.assertEqual(model.calls, 1)
        self.assertEqual(patch.begin_calls, 1)
        self.assertEqual(stitch_inference.begin_calls, 1)
        self.assertEqual(patch.finalize_calls, 1)
        self.assertEqual(stitch.finalize_calls, 1)
        self.assertEqual(len(patch.observe_calls), 1)
        self.assertEqual(len(stitch_inference.observe_calls), 1)
        self.assertIs(patch.observe_calls[0], stitch_inference.observe_calls[0])
        self.assertIsInstance(patch.observe_calls[0], ModelOutputBatch)
        self.assertEqual(report.stages["patch"].summary["patch_metric"], 1.0)
        self.assertEqual(report.stages["stitch"].summary["stitch_metric"], 2.0)
    def test_patch_eval_runs_metrics_once_and_validation_evaluator_returns_staged_patch_report(self):
        logits = torch.tensor(
            [
                [[[10.0, -10.0], [10.0, -10.0]]],
                [[[-10.0, 10.0], [-10.0, 10.0]]],
            ],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [0.0, 1.0]]],
            ],
            dtype=torch.float32,
        )
        batch = Batch(
            x=torch.zeros_like(logits),
            y=targets,
            meta=BatchMeta(
                segment_ids=["segA", "segB"],
                valid_mask=torch.ones_like(targets),
                group_idx=torch.tensor([0, 1], dtype=torch.long),
            ),
        )
        evaluator = ValidationEvaluator(
            patch=PatchEval(
                metrics=(
                    Dice(),
                    BalancedAccuracy(),
                    Dice(threshold=96.0 / 255.0),
                    BalancedAccuracy(threshold=96.0 / 255.0),
                )
            )
        ).build(data=DataBundle(train_loader=[], eval_loader=[], in_channels=1, group_counts=[1, 1]))

        report = evaluator.evaluate(_FixedLogitModel(logits), [batch])
        flattened = flatten_eval_report(report, stage_prefix=evaluator.stage_prefix)

        self.assertEqual(report.summary, {})
        self.assertEqual(set(report.stages.keys()), {"patch"})
        self.assertAlmostEqual(report.stages["patch"].summary["Dice"], 1.0)
        self.assertAlmostEqual(report.stages["patch"].summary["BalancedAccuracy"], 1.0)
        self.assertAlmostEqual(flattened.summary["val/patch/Dice"], 1.0)
        self.assertAlmostEqual(flattened.summary["val/patch/BalancedAccuracy"], 1.0)
        self.assertAlmostEqual(flattened.summary["val/patch/Dice_thr_96_255"], 1.0)
        self.assertAlmostEqual(flattened.summary["val/patch/BalancedAccuracy_thr_96_255"], 1.0)

    def test_patch_eval_requires_metrics(self):
        with self.assertRaisesRegex(ValueError, "at least one metric"):
            PatchEval().build(data=DataBundle(train_loader=[], eval_loader=[], in_channels=1, extras={}))

    def test_stitch_eval_requires_metrics(self):
        stitch = StitchRuntime._from_config(
            {
                "downsample": 1,
                "eval": {
                    "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                },
            }
        )
        stitch_inference = _bound_stitch_inference(stitch)
        dataset = SimpleNamespace(layout=NestedZarrLayout("."))
        bundle = DataBundle(
            train_loader=[],
            eval_loader=_StaticLoader([], dataset=dataset),
            in_channels=1,
            extras={},
        )

        with self.assertRaisesRegex(ValueError, "at least one metric"):
            StitchEval().build(data=bundle, inference=stitch_inference)

    def test_validation_evaluator_requires_explicit_stitch_inference_for_stitch_eval(self):
        stitch = StitchRuntime._from_config(
            {
                "downsample": 1,
                "eval": {
                    "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "requires stitch_inference"):
            ValidationEvaluator(
                stitch=StitchEval(metrics=(Dice(),)),
            ).build(
                data=DataBundle(train_loader=[], eval_loader=[], in_channels=1, extras={}),
            )

    def test_validation_evaluator_rejects_stitch_recipe_without_eval_segments(self):
        with self.assertRaisesRegex(ValueError, "stitch\\.eval\\.segments"):
            ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=StitchRuntimeRecipe(),
                ),
                stitch=StitchEval(metrics=(Dice(),)),
            ).build(
                data=DataBundle(train_loader=[], eval_loader=[], in_channels=1, extras={}),
            )

    def test_stitch_inference_requires_explicit_store_root_dir(self):
        stitch = StitchRuntime._from_config(
            {
                "downsample": 1,
                "eval": {
                    "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "root_dir"):
            _bound_stitch_inference(stitch).begin_epoch()

    def test_patch_and_stitch_eval_share_single_inference_loop_with_zarr_store(self):
        labels_hw = np.array(
            [
                [255, 0, 255, 0],
                [0, 255, 0, 255],
                [255, 0, 255, 0],
                [0, 255, 0, 255],
            ],
            dtype=np.uint8,
        )
        logits = torch.where(
            torch.as_tensor(labels_hw, dtype=torch.float32).unsqueeze(0).unsqueeze(0) > 0.5,
            torch.full((1, 1, 4, 4), 10.0, dtype=torch.float32),
            torch.full((1, 1, 4, 4), -10.0, dtype=torch.float32),
        )
        targets = (torch.as_tensor(labels_hw, dtype=torch.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        batch = Batch(
            x=torch.zeros_like(logits),
            y=targets,
            meta=BatchMeta(
                segment_ids=["segA"],
                valid_mask=torch.ones_like(targets),
                patch_xyxy=torch.tensor([[0, 0, 4, 4]], dtype=torch.long),
                group_idx=torch.tensor([0], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((4, 4), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                    },
                }
            )
            model = _CountingLogitModel(logits)
            evaluator = ValidationEvaluator(
                patch=PatchEval(metrics=(Dice(),)),
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(
                    metrics=(Dice(),),
                    components=(
                        {"component_key": ("segA", 0), "shape": (4, 4), "bbox": (0, 4, 0, 4)},
                    ),
                ),
            ).build(
                data=DataBundle(
                    train_loader=[],
                    eval_loader=eval_loader,
                    in_channels=1,
                    group_counts=[1],
                ),
            )

            report = flatten_eval_report(
                evaluator.evaluate(model, eval_loader),
                stage_prefix=evaluator.stage_prefix,
            )

        self.assertEqual(model.calls, 1)
        self.assertAlmostEqual(report.summary["val/patch/Dice"], 1.0)
        self.assertAlmostEqual(report.summary["val/stitch/Dice"], 1.0)

    def test_zarr_stitch_store_blends_logits_before_sigmoid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval").build(
                segment_shapes={"segA": (1, 1)},
                downsample=1,
            )
            logits = torch.tensor([[[[4.0]]], [[[-1.0]]]], dtype=torch.float32)

            store.add_batch(
                logits=logits,
                xyxys=[(0, 0, 1, 1), (0, 0, 1, 1)],
                segment_ids=("segA", "segA"),
            )
            probs, coverage = store.read_region_probs_and_coverage(segment_id="segA")
            recovered_logits, _ = store.read_region_logits_and_coverage(segment_id="segA")

        expected_logit = float((4.0 + (-1.0)) / 2.0)
        expected = float(torch.sigmoid(torch.tensor(expected_logit, dtype=torch.float32)).item())
        self.assertTrue(bool(coverage[0, 0]))
        self.assertAlmostEqual(float(probs[0, 0]), expected, places=6)
        self.assertAlmostEqual(float(recovered_logits[0, 0]), expected_logit, places=6)

    def test_stitch_eval_uses_eval_roi_as_default_stitch_region(self):
        labels_hw = np.zeros((4, 4), dtype=np.uint8)
        labels_hw[1:3, 1:3] = 255
        logits = torch.full((1, 1, 4, 4), 10.0, dtype=torch.float32)
        batch = Batch(
            x=torch.zeros_like(logits),
            y=(torch.as_tensor(labels_hw, dtype=torch.float32) / 255.0).unsqueeze(0).unsqueeze(0),
            meta=BatchMeta(
                segment_ids=["segA"],
                valid_mask=torch.ones((1, 1, 4, 4), dtype=torch.float32),
                patch_xyxy=torch.tensor([[0, 0, 4, 4]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((4, 4), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "use_roi": True,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (4, 4), "bbox": (1, 3, 1, 3)}],
                    },
                }
            )
            evaluator = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(metrics=(Dice(),)),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

            report = flatten_eval_report(
                evaluator.evaluate(_FixedLogitModel(logits), eval_loader),
                stage_prefix=evaluator.stage_prefix,
            )
            probs_path = root / ".tmp" / "stitch_eval" / "segA__prob.zarr"
            probs = np.asarray(zarr.open(str(probs_path), mode="r"))

        expected_prob = float(torch.sigmoid(torch.tensor(10.0, dtype=torch.float32)).item())
        self.assertAlmostEqual(report.summary["val/stitch/Dice"], 1.0)
        self.assertEqual(tuple(int(v) for v in probs.shape), (4, 4))
        self.assertAlmostEqual(float(probs[1, 1]), expected_prob, places=6)
        self.assertAlmostEqual(float(probs[2, 2]), expected_prob, places=6)
        self.assertAlmostEqual(float(probs[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(probs[3, 3]), 0.0, places=6)

    def test_zarr_stitch_store_uses_full_segment_arrays_and_only_updates_roi_area(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval").build(
                segment_shapes={"segA": (4, 4)},
                downsample=1,
                segment_rois={"segA": ((1, 3, 1, 3),)},
            )
            logits = torch.full((1, 1, 4, 4), 10.0, dtype=torch.float32)

            store.add_batch(
                logits=logits,
                xyxys=[(0, 0, 4, 4)],
                segment_ids=("segA",),
            )
            probs, coverage = store.read_region_probs_and_coverage(segment_id="segA")

        expected_prob = float(torch.sigmoid(torch.tensor(10.0, dtype=torch.float32)).item())
        self.assertEqual(tuple(int(v) for v in probs.shape), (4, 4))
        self.assertTrue(bool(coverage[1, 1]))
        self.assertTrue(bool(coverage[2, 2]))
        self.assertFalse(bool(coverage[0, 0]))
        self.assertFalse(bool(coverage[3, 3]))
        self.assertAlmostEqual(float(probs[1, 1]), expected_prob, places=6)
        self.assertAlmostEqual(float(probs[2, 2]), expected_prob, places=6)
        self.assertAlmostEqual(float(probs[0, 0]), 0.0, places=6)

    def test_zarr_stitch_store_supports_disjoint_segment_rois_without_filling_gap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval").build(
                segment_shapes={"segA": (6, 6)},
                downsample=1,
                segment_rois={"segA": ((1, 2, 1, 2), (4, 5, 4, 5))},
            )
            logits = torch.full((1, 1, 6, 6), 10.0, dtype=torch.float32)

            store.add_batch(
                logits=logits,
                xyxys=[(0, 0, 6, 6)],
                segment_ids=("segA",),
            )
            probs, coverage = store.read_region_probs_and_coverage(segment_id="segA")

        expected_prob = float(torch.sigmoid(torch.tensor(10.0, dtype=torch.float32)).item())
        self.assertTrue(bool(coverage[1, 1]))
        self.assertTrue(bool(coverage[4, 4]))
        self.assertFalse(bool(coverage[2, 2]))
        self.assertFalse(bool(coverage[3, 3]))
        self.assertAlmostEqual(float(probs[1, 1]), expected_prob, places=6)
        self.assertAlmostEqual(float(probs[4, 4]), expected_prob, places=6)

    def test_stitched_preview_downsample_matches_legacy_uint8_rounding(self):
        image = np.array(
            [
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [8, 9, 10, 11],
                [12, 13, 14, 15],
            ],
            dtype=np.uint8,
        )

        reduced = downsample_preview_for_media(
            image,
            source_downsample=1,
            media_downsample=2,
        )

        expected = np.array(
            [
                [2, 4],
                [10, 12],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(reduced, expected)

    def test_zarr_stitch_store_writes_preview_png_next_to_prob_zarr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval").build(
                segment_shapes={"segA": (4, 4)},
                downsample=1,
                segment_rois={"segA": ((1, 3, 1, 3),)},
            )
            logits = torch.full((1, 1, 4, 4), 10.0, dtype=torch.float32)

            store.add_batch(
                logits=logits,
                xyxys=[(0, 0, 4, 4)],
                segment_ids=("segA",),
            )
            preview_path = store.write_full_segment_preview_png(segment_id="segA")
            self.assertTrue(Path(preview_path).is_file())

        self.assertTrue(str(preview_path).endswith("segA__prob.png"))

    def test_zarr_stitch_store_indexes_roi_slices_relative_to_cropped_patch_window(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval").build(
                segment_shapes={"segA": (4, 4)},
                downsample=1,
                segment_rois={"segA": ((1, 3, 1, 3),)},
            )
            logits = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

            store.add_batch(
                logits=logits,
                xyxys=[(-1, -1, 3, 3)],
                segment_ids=("segA",),
            )
            recovered_logits, coverage = store.read_region_logits_and_coverage(segment_id="segA")

        expected = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 10.0, 11.0, 0.0],
                [0.0, 14.0, 15.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.array_equal(coverage, expected != 0.0))
        self.assertTrue(np.allclose(recovered_logits, expected))

    def test_stitch_eval_reports_component_metrics(self):
        labels_hw = np.array(
            [
                [255, 255, 0, 0],
                [255, 255, 0, 0],
                [0, 0, 255, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        logits = torch.full((1, 1, 4, 4), -10.0, dtype=torch.float32)
        logits[..., :2, :2] = 10.0
        targets = (torch.as_tensor(labels_hw, dtype=torch.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        batch = Batch(
            x=torch.zeros_like(logits),
            y=targets,
            meta=BatchMeta(
                segment_ids=["segA"],
                valid_mask=torch.ones_like(targets),
                patch_xyxy=torch.tensor([[0, 0, 4, 4]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((4, 4), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                    },
                }
            )

            component_specs = (
                {"component_key": ("segA", 0), "shape": (4, 4), "bbox": (0, 2, 0, 2)},
                {"component_key": ("segA", 1), "shape": (4, 4), "bbox": (2, 3, 2, 3)},
            )
            component_mean = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_component_mean"),
                ),
                stitch=StitchEval(
                    metrics=(Dice(),),
                    components=component_specs,
                ),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

            component_mean_report = flatten_eval_report(
                component_mean.evaluate(_FixedLogitModel(logits), eval_loader),
                stage_prefix=component_mean.stage_prefix,
            )

        self.assertAlmostEqual(component_mean_report.summary["val/stitch/Dice"], 8.0 / 9.0)
        self.assertEqual(set(component_mean_report.by_segment.keys()), {"segA#component_0", "segA#component_1"})
        self.assertAlmostEqual(component_mean_report.by_segment["segA#component_0"]["val/stitch/Dice"], 1.0)
        self.assertAlmostEqual(component_mean_report.by_segment["segA#component_1"]["val/stitch/Dice"], 0.0)

    def test_stitch_eval_summary_uses_metric_state_across_components(self):
        labels_hw = np.array(
            [
                [255, 255, 0, 0],
                [255, 255, 0, 0],
                [0, 0, 255, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        logits = torch.full((1, 1, 4, 4), -10.0, dtype=torch.float32)
        logits[..., :2, :2] = 10.0
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 4, 4]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((4, 4), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                    },
                }
            )
            report = flatten_eval_report(
                ValidationEvaluator(
                    stitch_inference=StitchInferenceRecipe(
                        stitch_runtime=stitch,
                        store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_component_count"),
                    ),
                    stitch=StitchEval(metrics=(_ComponentCountMetric(),), component_connectivity=1),
                ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={})).evaluate(
                    _FixedLogitModel(logits),
                    eval_loader,
                ),
                stage_prefix="val",
            )

        self.assertEqual(report.summary["val/stitch/component_count"], 2.0)
        self.assertEqual(set(report.by_segment.keys()), {"segA#component_0", "segA#component_1"})
        self.assertEqual(report.by_segment["segA#component_0"]["val/stitch/component_count"], 1.0)
        self.assertEqual(report.by_segment["segA#component_1"]["val/stitch/component_count"], 1.0)

    def test_stitch_eval_component_pad_affects_confusion_style_metrics_too(self):
        labels_hw = np.zeros((3, 3), dtype=np.uint8)
        labels_hw[1, 1] = 255
        logits = torch.full((1, 1, 3, 3), 10.0, dtype=torch.float32)
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 3, 3]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 3, 3), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((3, 3), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (3, 3)}],
                    },
                }
            )
            component_spec = (
                {"component_key": ("segA", 0), "shape": (3, 3), "bbox": (1, 2, 1, 2)},
            )
            report_pad0 = flatten_eval_report(
                ValidationEvaluator(
                    stitch_inference=StitchInferenceRecipe(
                        stitch_runtime=stitch,
                        store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_pad0"),
                    ),
                    stitch=StitchEval(metrics=(Dice(),), components=component_spec, component_pad=0),
                ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={})).evaluate(
                    _FixedLogitModel(logits),
                    eval_loader,
                ),
                stage_prefix="val",
            )
            report_pad1 = flatten_eval_report(
                ValidationEvaluator(
                    stitch_inference=StitchInferenceRecipe(
                        stitch_runtime=stitch,
                        store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_pad1"),
                    ),
                    stitch=StitchEval(metrics=(Dice(),), components=component_spec, component_pad=1),
                ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={})).evaluate(
                    _FixedLogitModel(logits),
                    eval_loader,
                ),
                stage_prefix="val",
            )

        self.assertAlmostEqual(report_pad0.summary["val/stitch/Dice"], 1.0, places=6)
        self.assertAlmostEqual(report_pad1.summary["val/stitch/Dice"], 0.2, places=6)

    def test_stitch_eval_rejects_explicit_components_without_bbox(self):
        stitch = StitchRuntime._from_config(
            {
                "downsample": 1,
                "eval": {
                    "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                },
            }
        )

        with self.assertRaisesRegex(ValueError, "require bbox for every component spec"):
            StitchEval(
                metrics=(Dice(),),
                components=(
                    {"component_key": ("segA", 0), "shape": (4, 4)},
                ),
            ).build(
                data=DataBundle(train_loader=[], eval_loader=[], in_channels=1, extras={}),
                inference=_bound_stitch_inference(stitch),
            )

    def test_stitch_eval_rejects_explicit_component_bbox_with_multiple_components(self):
        labels_hw = np.zeros((4, 4), dtype=np.uint8)
        labels_hw[0, 0] = 255
        labels_hw[3, 3] = 255
        logits = torch.full((1, 1, 4, 4), -10.0, dtype=torch.float32)
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 4, 4]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((4, 4), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                    },
                }
            )
            evaluator = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(
                    metrics=(Dice(),),
                    components=(
                        {"component_key": ("segA", 0), "shape": (4, 4), "bbox": (0, 4, 0, 4)},
                    ),
                ),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

            with self.assertRaisesRegex(ValueError, "must isolate exactly one supervised GT component"):
                evaluator.evaluate(_FixedLogitModel(logits), eval_loader)

    def test_stitch_eval_defaults_to_inklabel_components_inside_supervision_mask(self):
        labels_hw = np.zeros((6, 6), dtype=np.uint8)
        labels_hw[1:2, 1:2] = 255
        labels_hw[4:5, 4:5] = 255
        supervision_hw = np.zeros((6, 6), dtype=np.uint8)
        supervision_hw[1:2, 1:2] = 255
        supervision_hw[4:5, 4:5] = 255

        logits = torch.full((1, 1, 6, 6), -10.0, dtype=torch.float32)
        logits[..., 1:2, 1:2] = 10.0
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 6, 6]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 6, 6), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", supervision_hw)

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "use_roi": True,
                    "eval": {
                        "segments": [
                            {
                                "segment_id": "segA",
                                "shape": (6, 6),
                                "bbox": [(1, 2, 1, 2), (4, 5, 4, 5)],
                            }
                        ],
                    },
                }
            )
            evaluator = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(metrics=(Dice(),)),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

            report = flatten_eval_report(
                evaluator.evaluate(_FixedLogitModel(logits), eval_loader),
                stage_prefix=evaluator.stage_prefix,
            )

        self.assertAlmostEqual(report.summary["val/stitch/Dice"], 2.0 / 3.0)
        self.assertEqual(set(report.by_segment.keys()), {"segA#component_0", "segA#component_1"})
        self.assertAlmostEqual(report.by_segment["segA#component_0"]["val/stitch/Dice"], 1.0)
        self.assertAlmostEqual(report.by_segment["segA#component_1"]["val/stitch/Dice"], 0.0)

    def test_stitch_eval_detected_components_read_full_supervision_then_roi_labels(self):
        labels_hw = np.zeros((6, 6), dtype=np.uint8)
        labels_hw[1:2, 1:2] = 255
        labels_hw[4:5, 4:5] = 255
        supervision_hw = np.zeros((6, 6), dtype=np.uint8)
        supervision_hw[1:2, 1:2] = 255
        supervision_hw[4:5, 4:5] = 255

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 6, 6), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", supervision_hw)

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (6, 6)}],
                    },
                }
            )
            evaluator = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(metrics=(Dice(),)),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))
            region_reader = evaluator.stitch._region_reader
            store = evaluator.stitch._inference.store

            with (
                mock.patch(
                    "ink.recipes.eval.stitch_regions.read_supervision_mask_region",
                    wraps=stitch_regions_module.read_supervision_mask_region,
                ) as read_supervision,
                mock.patch(
                    "ink.recipes.eval.stitch_regions.read_label_region",
                    wraps=stitch_regions_module.read_label_region,
                ) as read_label,
                mock.patch(
                    "ink.recipes.eval.stitch_regions.read_label_and_supervision_mask_region",
                    wraps=stitch_regions_module.read_label_and_supervision_mask_region,
                ) as read_pair,
            ):
                components = region_reader.detected_segment_components(
                    segment_id="segA",
                    store=store,
                    connectivity=2,
                )

        self.assertEqual(len(components), 2)
        self.assertEqual(read_supervision.call_count, 1)
        self.assertEqual(read_label.call_count, 2)
        self.assertEqual(read_pair.call_count, 0)

    def test_stitch_eval_reuses_component_disk_cache_without_rereading_masks(self):
        labels_hw = np.zeros((6, 6), dtype=np.uint8)
        labels_hw[1:2, 1:2] = 255
        labels_hw[4:5, 4:5] = 255
        supervision_hw = np.zeros((6, 6), dtype=np.uint8)
        supervision_hw[1:2, 1:2] = 255
        supervision_hw[4:5, 4:5] = 255

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 6, 6), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", supervision_hw)

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (6, 6)}],
                    },
                }
            )
            prepared_cache_root = root / ".tmp" / "prepared_artifacts"
            evaluator_a = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(metrics=(Dice(),), prepared_cache_root=prepared_cache_root),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))
            evaluator_a.stitch._region_reader.detected_segment_components(
                segment_id="segA",
                store=evaluator_a.stitch._inference.store,
                connectivity=2,
            )

            evaluator_b = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(metrics=(Dice(),), prepared_cache_root=prepared_cache_root),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))
            region_reader = evaluator_b.stitch._region_reader
            store = evaluator_b.stitch._inference.store

            with (
                mock.patch(
                    "ink.recipes.eval.stitch_regions.read_supervision_mask_region",
                    wraps=stitch_regions_module.read_supervision_mask_region,
                ) as read_supervision,
                mock.patch(
                    "ink.recipes.eval.stitch_regions.read_label_region",
                    wraps=stitch_regions_module.read_label_region,
                ) as read_label,
            ):
                components = region_reader.detected_segment_components(
                    segment_id="segA",
                    store=store,
                    connectivity=2,
                )
            cache_files = sorted(
                path.relative_to(prepared_cache_root).as_posix()
                for path in prepared_cache_root.rglob("*")
                if path.is_file()
            )

        self.assertEqual(len(components), 2)
        self.assertEqual(read_supervision.call_count, 0)
        self.assertEqual(read_label.call_count, 0)
        self.assertTrue(any(path.startswith("components/segA/components_") and path.endswith("_c2.npz") for path in cache_files))

    def test_stitch_eval_keeps_one_component_across_adjacent_configured_rois(self):
        labels_hw = np.zeros((3, 6), dtype=np.uint8)
        labels_hw[1, :] = 255
        logits = torch.full((1, 1, 3, 6), -10.0, dtype=torch.float32)
        logits[..., 1, :] = 10.0
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 6, 3]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 3, 6), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((3, 6), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "use_roi": True,
                    "eval": {
                        "segments": [
                            {
                                "segment_id": "segA",
                                "shape": (3, 6),
                                "bbox": [(0, 3, 0, 3), (0, 3, 3, 6)],
                            }
                        ],
                    },
                }
            )
            evaluator = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(metrics=(Dice(),)),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

            report = flatten_eval_report(
                evaluator.evaluate(_FixedLogitModel(logits), eval_loader),
                stage_prefix=evaluator.stage_prefix,
            )

        self.assertAlmostEqual(report.summary["val/stitch/Dice"], 1.0)
        self.assertEqual(set(report.by_segment.keys()), {"segA#component_0"})
        self.assertAlmostEqual(report.by_segment["segA#component_0"]["val/stitch/Dice"], 1.0)

    def test_stitch_eval_excludes_other_component_foreground_inside_component_bbox(self):
        labels_hw = np.zeros((5, 5), dtype=np.uint8)
        labels_hw[0, :] = 255
        labels_hw[4, :] = 255
        labels_hw[:, 0] = 255
        labels_hw[:, 4] = 255
        labels_hw[2, 2] = 255
        logits = torch.full((1, 1, 5, 5), -10.0, dtype=torch.float32)
        logits[..., 0, :] = 10.0
        logits[..., 4, :] = 10.0
        logits[..., :, 0] = 10.0
        logits[..., :, 4] = 10.0
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 5, 5]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 5, 5), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((5, 5), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (5, 5)}],
                    },
                }
            )
            evaluator = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(metrics=(Dice(),)),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

            report = flatten_eval_report(
                evaluator.evaluate(_FixedLogitModel(logits), eval_loader),
                stage_prefix=evaluator.stage_prefix,
            )

        self.assertEqual(set(report.by_segment.keys()), {"segA#component_0", "segA#component_1"})
        self.assertAlmostEqual(report.summary["val/stitch/Dice"], 32.0 / 33.0, places=6)
        component_scores = sorted(float(metrics["val/stitch/Dice"]) for metrics in report.by_segment.values())
        self.assertAlmostEqual(component_scores[0], 0.0, places=6)
        self.assertAlmostEqual(component_scores[1], 1.0, places=6)

    def test_stitch_eval_matches_legacy_component_connectivity(self):
        labels_hw = np.zeros((3, 3), dtype=np.uint8)
        labels_hw[0, 0] = 255
        labels_hw[1, 1] = 255
        logits = torch.full((1, 1, 3, 3), -10.0, dtype=torch.float32)
        logits[..., 0, 0] = 10.0
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 3, 3]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 3, 3), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((3, 3), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (3, 3)}],
                    },
                }
            )
            evaluator_default = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval_default"),
                ),
                stitch=StitchEval(metrics=(Dice(),)),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))
            evaluator_conn4 = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval_conn4"),
                ),
                stitch=StitchEval(metrics=(Dice(),), component_connectivity=1),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

            report_default = flatten_eval_report(
                evaluator_default.evaluate(_FixedLogitModel(logits), eval_loader),
                stage_prefix=evaluator_default.stage_prefix,
            )
            report_conn4 = flatten_eval_report(
                evaluator_conn4.evaluate(_FixedLogitModel(logits), eval_loader),
                stage_prefix=evaluator_conn4.stage_prefix,
            )

        self.assertEqual(set(report_default.by_segment.keys()), {"segA#component_0"})
        self.assertAlmostEqual(report_default.summary["val/stitch/Dice"], 2.0 / 3.0, places=6)
        self.assertEqual(set(report_conn4.by_segment.keys()), {"segA#component_0", "segA#component_1"})
        self.assertAlmostEqual(report_conn4.summary["val/stitch/Dice"], 2.0 / 3.0, places=6)

    def test_stitch_eval_keeps_component_bbox_in_full_resolution_when_layout_downsample_is_greater_than_one(self):
        labels_hw = np.array(
            [
                [255, 255, 0, 0],
                [255, 255, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        logits = torch.full((1, 1, 4, 4), -10.0, dtype=torch.float32)
        logits[..., :2, :2] = 10.0
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 4, 4]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((4, 4), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 2,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                    },
                }
            )
            evaluator = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(
                    metrics=(Dice(),),
                    components=(
                        {"component_key": ("segA", 0), "shape": (4, 4), "bbox": (0, 2, 0, 2)},
                    ),
                ),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

            report = flatten_eval_report(
                evaluator.evaluate(_FixedLogitModel(logits), eval_loader),
                stage_prefix=evaluator.stage_prefix,
            )

        self.assertEqual(evaluator.stitch._inference.store.downsample, 1)
        self.assertAlmostEqual(report.summary["val/stitch/Dice"], 1.0)

    def test_pfm_weighted_metric_matches_perfect_and_empty_component_cases(self):
        targets = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        targets[..., 1:4, 1:4] = 1.0
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        metric = PFMWeighted()

        perfect_batch = ModelOutputBatch(
            logits=torch.where(targets > 0.5, torch.full_like(targets, 10.0), torch.full_like(targets, -10.0)),
            targets=targets,
            valid_mask=valid_mask,
        )
        empty_pred_batch = ModelOutputBatch(
            logits=torch.full_like(targets, -10.0),
            targets=targets,
            valid_mask=valid_mask,
        )

        perfect_state = metric.update(
            metric.empty_state(),
            StitchMetricBatch(
                logits=perfect_batch.logits,
                targets=perfect_batch.targets,
                valid_mask=perfect_batch.valid_mask,
                connectivity=2,
            ),
        )
        empty_state = metric.update(
            metric.empty_state(),
            StitchMetricBatch(
                logits=empty_pred_batch.logits,
                targets=empty_pred_batch.targets,
                valid_mask=empty_pred_batch.valid_mask,
                connectivity=2,
            ),
        )

        self.assertAlmostEqual(metric.finalize(perfect_state).summary["PFMWeighted"], 1.0, places=6)
        self.assertAlmostEqual(metric.finalize(empty_state).summary["PFMWeighted"], 0.0, places=6)

    def test_pfm_weighted_requires_binary_targets(self):
        targets = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        targets[..., 1:4, 1:4] = 1.0
        targets[..., 2, 2] = 0.25
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        metric = PFMWeighted()
        batch = ModelOutputBatch(
            logits=torch.full_like(targets, -10.0),
            targets=targets,
            valid_mask=valid_mask,
        )

        with self.assertRaisesRegex(ValueError, "binary targets"):
            metric.update(
                metric.empty_state(),
                StitchMetricBatch(
                    logits=batch.logits,
                    targets=batch.targets,
                    valid_mask=batch.valid_mask,
                    connectivity=2,
                ),
            )

    def test_pfm_weighted_supports_legacy_skeleton_methods(self):
        target = np.zeros((5, 5), dtype=bool)
        target[1:4, 1:4] = True

        for method in ("guo_hall", "zhang_suen"):
            skeleton = prepared_skeletonize_binary(target, method=method)
            self.assertEqual(tuple(int(v) for v in skeleton.shape), (5, 5))
            self.assertFalse(bool(np.logical_and(skeleton, ~target).any()))

    def test_pfm_weighted_threads_configured_skeleton_method_without_prepared_artifacts(self):
        targets = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        targets[..., 1:4, 1:4] = 1.0
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        metric = PFMWeighted(skeleton_method="zhang_suen")
        batch = ModelOutputBatch(
            logits=torch.full_like(targets, -10.0),
            targets=targets,
            valid_mask=valid_mask,
        )

        seen_methods = []

        def _fake_skeletonize(mask, *, method="guo_hall"):
            seen_methods.append(str(method))
            return np.asarray(mask, dtype=bool)

        with mock.patch(
            "ink.recipes.metrics.pfm_weighted.skeletonize_binary",
            side_effect=_fake_skeletonize,
        ):
            metric.update(
                metric.empty_state(),
                StitchMetricBatch(
                    logits=batch.logits,
                    targets=batch.targets,
                    valid_mask=batch.valid_mask,
                    connectivity=2,
                ),
            )

        self.assertEqual(seen_methods, ["zhang_suen"])

    def test_drd_metric_matches_perfect_and_mismatch_component_cases(self):
        targets = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
        targets[..., 1:4, 1:4] = 1.0
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
        metric = DRD()

        perfect_batch = ModelOutputBatch(
            logits=torch.where(targets > 0.5, torch.full_like(targets, 10.0), torch.full_like(targets, -10.0)),
            targets=targets,
            valid_mask=valid_mask,
        )
        mismatch_logits = torch.where(targets > 0.5, torch.full_like(targets, 10.0), torch.full_like(targets, -10.0))
        mismatch_logits[..., 2, 2] = -10.0
        mismatch_batch = ModelOutputBatch(
            logits=mismatch_logits,
            targets=targets,
            valid_mask=valid_mask,
        )

        perfect_state = metric.update(
            metric.empty_state(),
            StitchMetricBatch(
                logits=perfect_batch.logits,
                targets=perfect_batch.targets,
                valid_mask=perfect_batch.valid_mask,
                connectivity=2,
            ),
        )
        mismatch_state = metric.update(
            metric.empty_state(),
            StitchMetricBatch(
                logits=mismatch_batch.logits,
                targets=mismatch_batch.targets,
                valid_mask=mismatch_batch.valid_mask,
                connectivity=2,
            ),
        )

        self.assertAlmostEqual(metric.finalize(perfect_state).summary["drd"], 0.0, places=6)
        self.assertGreater(metric.finalize(mismatch_state).summary["drd"], 0.0)

    def test_stitch_eval_reports_pfm_weighted_for_component_metrics(self):
        labels_hw = np.zeros((5, 5), dtype=np.uint8)
        labels_hw[1:4, 1:4] = 255
        logits = torch.full((1, 1, 5, 5), -10.0, dtype=torch.float32)
        logits[..., 1:4, 1:4] = 10.0
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 5, 5]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 5, 5), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((5, 5), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (5, 5)}],
                    },
                }
            )
            evaluator = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(metrics=(PFMWeighted(),)),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

            report = flatten_eval_report(
                evaluator.evaluate(_FixedLogitModel(logits), eval_loader),
                stage_prefix=evaluator.stage_prefix,
            )

        self.assertAlmostEqual(report.summary["val/stitch/PFMWeighted"], 1.0, places=6)
        self.assertEqual(set(report.by_segment.keys()), {"segA#component_0"})
        self.assertAlmostEqual(report.by_segment["segA#component_0"]["val/stitch/PFMWeighted"], 1.0, places=6)

    def test_stitch_eval_reuses_prepared_pfm_artifacts_across_metrics(self):
        labels_hw = np.zeros((5, 5), dtype=np.uint8)
        labels_hw[1:4, 1:4] = 255
        logits = torch.full((1, 1, 5, 5), -10.0, dtype=torch.float32)
        logits[..., 1:4, 1:4] = 10.0
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 5, 5]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 5, 5), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((5, 5), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (5, 5)}],
                    },
                }
            )
            with mock.patch(
                "ink.recipes.eval.stitch_prepared.skeletonize_binary",
                wraps=prepared_skeletonize_binary,
            ) as skeletonize:
                evaluator = ValidationEvaluator(
                    stitch_inference=StitchInferenceRecipe(
                        stitch_runtime=stitch,
                        store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                    ),
                    stitch=StitchEval(
                        metrics=(
                            PFMWeighted(name="PFMWeighted_a"),
                            PFMWeighted(name="PFMWeighted_b"),
                        ),
                    ),
                ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

                report = flatten_eval_report(
                    evaluator.evaluate(_FixedLogitModel(logits), eval_loader),
                    stage_prefix=evaluator.stage_prefix,
                )

        self.assertAlmostEqual(report.summary["val/stitch/PFMWeighted_a"], 1.0, places=6)
        self.assertAlmostEqual(report.summary["val/stitch/PFMWeighted_b"], 1.0, places=6)
        self.assertEqual(skeletonize.call_count, 1)

    def test_stitch_eval_reuses_prepared_pfm_artifacts_across_runs_with_disk_store(self):
        labels_hw = np.zeros((5, 5), dtype=np.uint8)
        labels_hw[1:4, 1:4] = 255
        logits = torch.full((1, 1, 5, 5), -10.0, dtype=torch.float32)
        logits[..., 1:4, 1:4] = 10.0
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 5, 5]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 5, 5), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((5, 5), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (5, 5)}],
                    },
                }
            )
            prepared_cache_root = root / ".tmp" / "prepared_artifacts"
            with mock.patch(
                "ink.recipes.eval.stitch_prepared.skeletonize_binary",
                wraps=prepared_skeletonize_binary,
            ) as skeletonize:
                evaluator_a = ValidationEvaluator(
                    stitch_inference=StitchInferenceRecipe(
                        stitch_runtime=stitch,
                        store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                    ),
                    stitch=StitchEval(
                        metrics=(PFMWeighted(),),
                        prepared_cache_root=prepared_cache_root,
                    ),
                ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))
                report_a = flatten_eval_report(
                    evaluator_a.evaluate(_FixedLogitModel(logits), eval_loader),
                    stage_prefix=evaluator_a.stage_prefix,
                )

                evaluator_b = ValidationEvaluator(
                    stitch_inference=StitchInferenceRecipe(
                        stitch_runtime=stitch,
                        store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                    ),
                    stitch=StitchEval(
                        metrics=(PFMWeighted(),),
                        prepared_cache_root=prepared_cache_root,
                    ),
                ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))
                report_b = flatten_eval_report(
                    evaluator_b.evaluate(_FixedLogitModel(logits), eval_loader),
                    stage_prefix=evaluator_b.stage_prefix,
                )
                cache_files = sorted(
                    path.relative_to(prepared_cache_root).as_posix()
                    for path in prepared_cache_root.rglob("*")
                    if path.is_file()
                )
                cache_artifact_names = sorted(Path(path).name for path in cache_files)

        self.assertAlmostEqual(report_a.summary["val/stitch/PFMWeighted"], 1.0, places=6)
        self.assertAlmostEqual(report_b.summary["val/stitch/PFMWeighted"], 1.0, places=6)
        self.assertEqual(skeletonize.call_count, 1)
        self.assertTrue(any(name.startswith("components_") and name.endswith("_c2.npz") for name in cache_artifact_names))
        self.assertTrue(
            {"pfm_weights_guo_hall.npz", "selected_component_gt.npy", "skeleton_guo_hall.npy"}.issubset(
                cache_artifact_names
            )
        )

    def test_stitch_eval_reports_drd_for_component_metrics(self):
        labels_hw = np.zeros((5, 5), dtype=np.uint8)
        labels_hw[1:4, 1:4] = 255
        logits = torch.full((1, 1, 5, 5), -10.0, dtype=torch.float32)
        logits[..., 1:4, 1:4] = 10.0
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 5, 5]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 5, 5), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((5, 5), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (5, 5)}],
                    },
                }
            )
            evaluator = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(metrics=(DRD(),)),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

            report = flatten_eval_report(
                evaluator.evaluate(_FixedLogitModel(logits), eval_loader),
                stage_prefix=evaluator.stage_prefix,
            )

        self.assertAlmostEqual(report.summary["val/stitch/drd"], 0.0, places=6)
        self.assertEqual(set(report.by_segment.keys()), {"segA#component_0"})
        self.assertAlmostEqual(report.by_segment["segA#component_0"]["val/stitch/drd"], 0.0, places=6)

    def test_stitch_eval_pfm_weighted_uses_component_crop_not_full_roi(self):
        labels_hw = np.zeros((5, 20), dtype=np.uint8)
        labels_hw[1:4, 1:4] = 255
        labels_hw[1:4, 15:18] = 255
        logits = torch.full((1, 1, 5, 20), -10.0, dtype=torch.float32)
        logits[..., 1:4, 1:4] = 10.0
        logits[..., 1:4, 15:18] = 10.0
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 20, 5]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 5, 20), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", labels_hw)
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((5, 20), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (5, 20)}],
                    },
                }
            )
            evaluator = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
                stitch=StitchEval(metrics=(PFMWeighted(),)),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

            report = flatten_eval_report(
                evaluator.evaluate(_FixedLogitModel(logits), eval_loader),
                stage_prefix=evaluator.stage_prefix,
            )

        self.assertAlmostEqual(report.summary["val/stitch/PFMWeighted"], 1.0, places=6)
        self.assertEqual(set(report.by_segment.keys()), {"segA#component_0", "segA#component_1"})
        self.assertAlmostEqual(report.by_segment["segA#component_0"]["val/stitch/PFMWeighted"], 1.0, places=6)
        self.assertAlmostEqual(report.by_segment["segA#component_1"]["val/stitch/PFMWeighted"], 1.0, places=6)

    def test_stitch_eval_allows_empty_metrics_for_inference_only_runs(self):
        logits = torch.full((1, 1, 4, 4), 10.0, dtype=torch.float32)
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 4, 4]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", np.zeros((4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((4, 4), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                    },
                }
            )
            evaluator = ValidationEvaluator(
                stitch_inference=StitchInferenceRecipe(
                    stitch_runtime=stitch,
                    store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval"),
                ),
            ).build(data=DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={}))

            report = evaluator.evaluate(_FixedLogitModel(logits), eval_loader)

        self.assertEqual(report.summary, {})

    def test_stitch_training_runs_stitch_only_inference(self):
        logits = torch.full((1, 1, 4, 4), 10.0, dtype=torch.float32)
        batch = Batch(
            x=torch.zeros_like(logits),
            y=None,
            meta=BatchMeta(
                segment_ids=["segA"],
                patch_xyxy=torch.tensor([[0, 0, 4, 4]], dtype=torch.long),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            segment_dir = root / "group_a" / "segA"
            _write_zarr_array(segment_dir / "segA.zarr", np.zeros((2, 4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_inklabels.zarr", np.zeros((4, 4), dtype=np.uint8))
            _write_zarr_array(segment_dir / "segA_supervision_mask.zarr", np.full((4, 4), 255, dtype=np.uint8))

            layout = NestedZarrLayout(root)
            eval_loader = _StaticLoader([batch], dataset=SimpleNamespace(layout=layout))
            stitch = StitchRuntime._from_config(
                {
                    "downsample": 1,
                    "eval": {
                        "segments": [{"segment_id": "segA", "shape": (4, 4)}],
                    },
                }
            )
            bundle = DataBundle(train_loader=[], eval_loader=eval_loader, in_channels=1, extras={})
            layout_info = stitch.eval_segment_layout()
            stitch_inference = StitchInference(
                store=ZarrStitchStore(root_dir=root / ".tmp" / "stitch_eval").build(
                    segment_shapes=layout_info.segment_shapes,
                    downsample=layout_info.downsample,
                    segment_rois=layout_info.segment_rois,
                ),
                segment_shapes=layout_info.segment_shapes,
                stitch_runtime=stitch,
            )

            run = StitchInferenceRun(
                experiment=SimpleNamespace(
                    model=_FixedLogitModel(logits),
                    runtime=SimpleNamespace(init_ckpt_path=None, resume_ckpt_path=None, wandb=None),
                ),
                inference_loader=eval_loader,
                stitch_inference=stitch_inference,
            ).run()

        self.assertEqual(run.batches, 1)
        self.assertTrue(str(run.store_root_dir).endswith("stitch_eval"))
        self.assertEqual(set((run.segment_prob_paths or {}).keys()), {"segA"})
        self.assertTrue(str((run.segment_prob_paths or {})["segA"]).endswith("segA__prob.zarr"))
        self.assertEqual(set((run.segment_preview_paths or {}).keys()), {"segA"})
        self.assertTrue(str((run.segment_preview_paths or {})["segA"]).endswith("segA__prob.png"))


if __name__ == "__main__":
    unittest.main()
