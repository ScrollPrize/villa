from __future__ import annotations

from contextlib import nullcontext
import json
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from ink.core import (
    Batch,
    BatchMeta,
    DataBundle,
    EvalReport,
    Experiment,
    ModelOutputBatch,
    RunFS,
    assemble_experiment,
    build_experiment_data,
)
from ink.recipes.eval import PatchEval, ValidationEvaluator
from ink.recipes.metrics import BalancedAccuracy, Dice, MetricReport, flatten_eval_report
from ink.recipes.losses.dice_bce import DiceBCEBatch, DiceBCEPerSample
from ink.recipes.objectives import ERMBatch, ERMGroupTopK, GroupDROComputer, GroupDROObjective, reduce_group_topk_loss
from ink.recipes.augment import TrainAugment
from ink.recipes.data.normalization import ClipMaxDiv255Normalization
from ink.recipes.data.samplers import GroupStratifiedSampler
from ink.recipes.trainers import PatchTrainer
from ink.recipes.trainers.support import WandbLogger
from tests.support.build_recipes import optional_build_recipe, required_build_recipe
from tests.support.in_memory_data import GroupedInMemoryPatchDataRecipe, InMemoryPatchDataRecipe, InMemoryPatchSamples


def _neutral_augment(size: int = 8) -> TrainAugment:
    return TrainAugment(
        size=size,
        horizontal_flip_p=0.0,
        vertical_flip_p=0.0,
        brightness_contrast_p=0.0,
        random_gamma_p=0.0,
        multiplicative_noise_p=0.0,
        shift_scale_rotate_p=0.0,
        blur_p=0.0,
        coarse_dropout_p=0.0,
        fourth_augment_p=0.0,
        invert_p=0.0,
    )


def _image(value: int, *, size: int = 8, in_channels: int = 1):
    import numpy as np

    return np.full((size, size, in_channels), value, dtype=np.uint8)


def _label(value: int, *, size: int = 8):
    import numpy as np

    return np.full((size, size, 1), value, dtype=np.uint8)


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


class TrainingRuntimeObjectiveDispatchTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor(
            [
                [[[0.2, -0.4], [1.1, 0.0]]],
                [[[-1.0, 0.8], [0.3, -0.7]]],
                [[[0.5, -0.2], [-0.1, 0.4]]],
            ],
            dtype=torch.float32,
        )
        self.targets = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 1.0], [1.0, 0.0]]],
                [[[1.0, 1.0], [0.0, 0.0]]],
            ],
            dtype=torch.float32,
        )
        self.group_idx = torch.tensor([0, 1, 2], dtype=torch.long)
        self.batch_loss = DiceBCEBatch()
        self.per_sample_loss = DiceBCEPerSample()

    def _base_experiment(self, *, objective, loss):
        return Experiment(
            name="objective_dispatch",
            data="data",
            model=_FixedLogitModel(self.logits),
            loss=loss,
            objective=objective,
            runtime=_LoopRuntime(None),
            augment=_neutral_augment(),
        )

    def _batch(self):
        return Batch(
            x=torch.zeros_like(self.logits),
            y=self.targets,
            meta=BatchMeta(
                segment_ids=[""] * int(self.logits.shape[0]),
                group_idx=self.group_idx,
            ),
        )

    def test_erm_batch_runtime_uses_primary_batch_loss(self):
        bundle = DataBundle(
            train_loader="train",
            eval_loader="val",
            in_channels=62,
            group_counts=[3, 4, 5],
        )
        runtime = _assemble_patch_training(self._base_experiment(objective=ERMBatch(), loss=self.batch_loss), bundle)

        out = runtime.run_step(self._batch())

        torch.testing.assert_close(out.loss, out.primary_loss)

    def test_erm_group_topk_runtime_uses_explicit_reduction_contract(self):
        bundle = DataBundle(
            train_loader="train",
            eval_loader="val",
            in_channels=62,
            group_counts=[3, 4, 5],
        )
        runtime = _assemble_patch_training(self._base_experiment(objective=ERMGroupTopK(group_topk=2), loss=self.per_sample_loss), bundle)

        out = runtime.run_step(self._batch())
        expected = reduce_group_topk_loss(
            out.primary_loss,
            group_idx=self.group_idx,
            n_groups=3,
            group_topk=2,
        )

        torch.testing.assert_close(out.loss, expected)

    def test_group_dro_runtime_matches_group_dro_computer_contract_and_state_updates(self):
        objective_kwargs = {
            "step_size": 0.3,
            "gamma": 0.2,
            "normalize_loss": True,
        }
        group_counts = [8, 4, 2]
        bundle = DataBundle(
            train_loader="train",
            eval_loader="val",
            in_channels=62,
            group_counts=group_counts,
        )
        runtime = _assemble_patch_training(
            self._base_experiment(objective=GroupDROObjective(**objective_kwargs), loss=self.per_sample_loss),
            bundle,
        )

        out = runtime.run_step(self._batch())

        reference = GroupDROComputer(
            n_groups=len(group_counts),
            group_counts=group_counts,
            **objective_kwargs,
        )
        expected_loss, _group_loss, _group_count, _weights = reference.loss(out.primary_loss, self.group_idx)

        torch.testing.assert_close(out.loss, expected_loss)
        torch.testing.assert_close(runtime.experiment.objective.adv_probs, reference.adv_probs)
        torch.testing.assert_close(runtime.experiment.objective.exp_avg_loss, reference.exp_avg_loss)
        torch.testing.assert_close(runtime.experiment.objective.exp_avg_initialized, reference.exp_avg_initialized)


class _TrainableBiasModel(torch.nn.Module):
    def __init__(self, *, init_bias: float = 1.0):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32))

    def forward(self, x):
        return x + self.bias


class _TrainableDownsampleBiasModel(torch.nn.Module):
    def __init__(self, *, init_bias: float = 1.0):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32))

    def forward(self, x):
        import torch.nn.functional as F

        return F.avg_pool3d(x, kernel_size=(1, 4, 4), stride=(1, 4, 4)) + self.bias


class _DeviceTrackingBiasModel(_TrainableBiasModel):
    def __init__(self, *, init_bias: float = 1.0):
        super().__init__(init_bias=init_bias)
        self.to_calls = []

    def to(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and args:
            device = args[0]
        self.to_calls.append(device)
        if device is not None and str(device).startswith("cuda"):
            return self
        return super().to(*args, **kwargs)


class _FixedLogitModel:
    def __init__(self, logits: torch.Tensor):
        self._logits = logits

    def __call__(self, _x):
        return self._logits.clone()


class _ModeTrackingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eval_calls = 0
        self.train_calls = 0

    def forward(self, x):
        return torch.zeros_like(x)

    def eval(self):
        self.eval_calls += 1
        return super().eval()

    def train(self, mode: bool = True):
        if mode:
            self.train_calls += 1
        return super().train(mode)


class _IdentityModel(torch.nn.Module):
    def forward(self, x):
        return x


class _QuadraticRegionLoss:
    def __call__(self, logits, targets, *, valid_mask=None):
        residual = logits - targets
        if valid_mask is not None:
            residual = residual * valid_mask
        per_sample = residual.square().reshape(residual.shape[0], -1).mean(dim=1)
        return per_sample

    def loss_values(self, logits, targets, *, valid_mask=None):
        return self(logits, targets, valid_mask=valid_mask)


class _RecordingObjective:
    def __init__(self):
        self.calls = []

    def __call__(self, loss, *, meta=None, group_idx=None, n_groups=None):
        del n_groups
        if meta is not None:
            group_idx = getattr(meta, "group_idx", group_idx)
        self.calls.append(
            {
                "loss": loss.detach().clone(),
                "group_idx": None if group_idx is None else group_idx.detach().clone(),
            }
        )
        return loss.mean() if loss.ndim > 0 else loss


class _RecordingScheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1


class _RecordingMetricRecipe:
    def __init__(self, outputs, final_report: MetricReport):
        self._outputs = list(outputs)
        self.final_report = final_report
        self.calls = []
        self.states = []

    def build(self, _ctx):
        return self

    def empty_state(self, *, n_groups=None):
        state = {"n_groups": n_groups, "outputs": []}
        self.states.append(state)
        return state

    def update(self, state, batch: ModelOutputBatch, *, shared=None):
        del shared
        self.calls.append(
            {
                "logits_shape": tuple(batch.logits.shape),
                "targets_shape": tuple(batch.targets.shape),
                "valid_mask": None if batch.valid_mask is None else batch.valid_mask.detach().clone(),
                "group_idx": None if batch.group_idx is None else batch.group_idx.detach().clone(),
                "segment_ids": tuple(batch.segment_ids),
            }
        )
        return {
            "n_groups": state["n_groups"],
            "outputs": [*state["outputs"], self._outputs[len(self.calls) - 1]],
        }

    def finalize(self, state):
        self.states.append(state)
        return self.final_report


class _SimpleBatchMetric:
    key = "metrics/val/simple_batch_metric"

    def build(self, _ctx):
        return self

    def empty_state(self, *, n_groups=None):
        del n_groups
        return {"total": 0.0, "count": 0.0}

    def update(self, state, batch: ModelOutputBatch, *, shared=None):
        del shared
        return {
            "total": float(state["total"] + float(batch.logits.detach().mean().item())),
            "count": float(state["count"] + 1.0),
        }

    def finalize(self, state):
        count = float(state["count"])
        return MetricReport(
            summary={
                self.key: 0.0 if count <= 0.0 else float(state["total"] / count),
            }
        )


class _LoopRuntime:
    def __init__(
        self,
        optimizer_setup,
        *,
        grad_accum=1,
        grad_clip_norm=None,
        wandb=None,
    ):
        self._optimizer_setup = optimizer_setup
        self.grad_accum = int(grad_accum)
        self.grad_clip_norm = grad_clip_norm
        self.wandb = wandb
        self.precision_context_calls = []

    def build(self, *, data, augment=None):
        del data, augment
        return self

    def build_optimizer_setup(self, model, *, epochs):
        del model
        self.last_build_epochs = int(epochs)
        return self._optimizer_setup

    def precision_context(self, *, device=None):
        self.precision_context_calls.append(device)
        return nullcontext()


class _FakeWandbRun:
    def __init__(self):
        self.summary = {}
        self.logged = []
        self.finished = False

    def log(self, payload, *, step=None):
        self.logged.append({"payload": dict(payload), "step": step})

    def finish(self):
        self.finished = True


class _FakeWandbImage:
    def __init__(self, image, caption=None):
        self.image = image
        self.caption = caption


class TrainingLoopOrchestrationTests(unittest.TestCase):
    def _make_runtime(self, *, grad_accum=1, epochs=1, eval_every=1, save_every_n_epochs=None):
        objective = _RecordingObjective()
        model = _TrainableBiasModel(init_bias=1.0)
        train_loader = [
            Batch(
                x=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                y=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                meta=BatchMeta(
                    segment_ids=[],
                    valid_mask=torch.ones((1, 1, 1, 1), dtype=torch.float32),
                    group_idx=torch.tensor([0], dtype=torch.long),
                ),
            ),
            Batch(
                x=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                y=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                meta=BatchMeta(
                    segment_ids=[],
                    valid_mask=torch.ones((1, 1, 1, 1), dtype=torch.float32),
                    group_idx=torch.tensor([1], dtype=torch.long),
                ),
            ),
            Batch(
                x=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                y=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                meta=BatchMeta(
                    segment_ids=[],
                    valid_mask=torch.ones((1, 1, 1, 1), dtype=torch.float32),
                    group_idx=torch.tensor([0], dtype=torch.long),
                ),
            ),
        ]
        experiment = Experiment(
            name="loop_runtime",
            data="data",
            model=model,
            loss=_QuadraticRegionLoss(),
            objective=objective,
            runtime=_LoopRuntime(
                optimizer_setup=None,
                grad_accum=grad_accum,
            ),
            augment=None,
            trainer=PatchTrainer(
                epochs=epochs,
                eval_every=eval_every,
                save_every_n_epochs=save_every_n_epochs,
            ),
        )
        runtime = _assemble_patch_training(
            experiment,
            DataBundle(
                train_loader=train_loader,
                eval_loader=[],
                in_channels=1,
                group_counts=[2, 1],
            ),
        )
        return runtime, model, objective

    def _optimizer_setup(self, model, *, interval):
        scheduler = _RecordingScheduler()
        return (
            SimpleNamespace(
                optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                scheduler=scheduler,
                scheduler_interval=interval,
            ),
            scheduler,
        )

    def test_run_training_epoch_steps_optimizer_and_step_scheduler_with_grad_accum(self):
        runtime, model, objective = self._make_runtime(grad_accum=2)
        optimizer_setup, scheduler = self._optimizer_setup(model, interval="step")

        result = runtime.run_epoch(optimizer_setup=optimizer_setup)

        self.assertEqual(result.batches, 3)
        self.assertEqual(result.optimizer_steps, 2)
        self.assertEqual(result.scheduler_steps, 2)
        self.assertEqual(scheduler.step_count, 2)
        self.assertEqual(len(objective.calls), 3)
        self.assertEqual([call["group_idx"].item() for call in objective.calls], [0, 1, 0])
        self.assertEqual(len(runtime.experiment.runtime.precision_context_calls), 3)
        self.assertLess(float(model.bias.detach().item()), 1.0)

    def test_run_training_epoch_steps_epoch_scheduler_once_per_epoch(self):
        runtime, model, _objective = self._make_runtime(grad_accum=2)
        optimizer_setup, scheduler = self._optimizer_setup(model, interval="epoch")

        result = runtime.run_epoch(optimizer_setup=optimizer_setup)

        self.assertEqual(result.optimizer_steps, 2)
        self.assertEqual(result.scheduler_steps, 1)
        self.assertEqual(scheduler.step_count, 1)

    def test_run_training_respects_eval_cadence(self):
        runtime, model, _objective = self._make_runtime(grad_accum=1, epochs=5, eval_every=2)
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")
        calls = {"count": 0}

        class _Evaluator:
            def evaluate(self, bound_model, eval_loader, *, device=None):
                del bound_model, eval_loader, device
                calls["count"] += 1
                return EvalReport(summary={"val/dice": float(calls["count"])})

        runtime.experiment = replace(runtime.experiment, evaluator=_Evaluator())
        result = runtime.run(optimizer_setup=optimizer_setup)

        self.assertEqual(len(result.train_epochs), 5)
        self.assertEqual([entry.epoch for entry in result.eval_epochs], [1, 3])
        self.assertEqual([report.report.summary["val/dice"] for report in result.eval_epochs], [1.0, 2.0])
        self.assertEqual(calls["count"], 2)

    def test_run_training_defaults_to_cuda_when_device_not_provided(self):
        model = _DeviceTrackingBiasModel(init_bias=1.0)
        objective = _RecordingObjective()
        experiment = Experiment(
            name="loop_runtime_device_default",
            data="data",
            model=model,
            loss=_QuadraticRegionLoss(),
            objective=objective,
            runtime=_LoopRuntime(optimizer_setup=None, grad_accum=1),
            augment=None,
            trainer=PatchTrainer(epochs=1, eval_every=1),
        )
        runtime = _assemble_patch_training(
            experiment,
            DataBundle(
                train_loader=[],
                eval_loader=[],
                in_channels=1,
                group_counts=[1],
            ),
        )
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")

        with patch("ink.core.device.torch.cuda.is_available", return_value=True):
            runtime.run(optimizer_setup=optimizer_setup)

        self.assertEqual(model.to_calls, ["cuda"])

    def test_run_training_reuses_bound_stitch_runtime_for_train_viz_and_eval_store_root(self):
        runtime, model, _objective = self._make_runtime(grad_accum=1, epochs=3, eval_every=1)
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")
        train_viz_calls = []
        eval_store_roots = []
        eval_preview_paths = []

        class _TrainStitch:
            def __init__(self):
                self.data = SimpleNamespace(
                    train=SimpleNamespace(
                        viz=SimpleNamespace(enabled=True),
                        segments=[],
                    )
                )

            def set_loaders(self, loaders):
                self.loaders = list(loaders)

            def run_viz_pass(self, bound_model, *, epoch):
                del bound_model
                epoch = int(epoch)
                train_viz_calls.append(epoch)
                if ((epoch + 1) % 2) != 0:
                    return None
                return {
                    "segA": {
                        "img_u8": [[epoch]],
                        "has": [[True]],
                        "meta": {"epoch": epoch},
                        }
                }

        class _Store:
            def __init__(self):
                self.root_dir = Path("initial")
                self.downsample = 1

            def segment_ids(self):
                return ("segA",)

            def full_segment_prob_preview_u8(self, *, segment_id, media_downsample=1):
                del segment_id, media_downsample
                return np.array([[127]], dtype=np.uint8)

            def write_full_segment_preview_png(self, *, segment_id, media_downsample=1, image_u8=None):
                del media_downsample, image_u8
                path = Path(self.root_dir) / f"{str(segment_id).replace('/', '__')}__prob.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"png")
                eval_preview_paths.append(path)
                return str(path)

        store = _Store()
        stitch_runtime = SimpleNamespace(
            data=SimpleNamespace(
                layout=SimpleNamespace(downsample=1),
                train=SimpleNamespace(
                    viz=SimpleNamespace(enabled=True),
                    segments=[],
                )
            ),
            train=_TrainStitch(),
        )

        class _Evaluator:
            stage_prefix = "val"

            def __init__(self):
                self.stitch_inference = SimpleNamespace(store=store, stitch_runtime=stitch_runtime)

            def prepare_run_artifacts(self, *, run_fs=None, epoch=None):
                if run_fs is not None:
                    root_dir = run_fs.artifacts_dir / "stitch_eval"
                    if epoch is not None:
                        root_dir = root_dir / f"epoch_{int(epoch):04d}"
                    self.stitch_inference.store.root_dir = root_dir

            def export_logged_images(self, *, media_downsample):
                del media_downsample
                preview_path = self.stitch_inference.store.write_full_segment_preview_png(
                    segment_id="segA",
                    media_downsample=1,
                    image_u8=np.array([[127]], dtype=np.uint8),
                )
                self.preview_path = Path(preview_path)
                return {}

            def evaluate(self, bound_model, eval_loader, *, device=None):
                del bound_model, eval_loader, device
                eval_store_roots.append(Path(self.stitch_inference.store.root_dir))
                return EvalReport(summary={"val/dice": 0.5})

        runtime.experiment = replace(runtime.experiment, evaluator=_Evaluator())
        runtime.stitch_runtime = stitch_runtime

        with tempfile.TemporaryDirectory() as tmpdir:
            run_fs = RunFS(
                Path(tmpdir) / "run",
                Experiment(
                    name=runtime.experiment.name,
                    data="data",
                    model="model",
                    loss="loss",
                    objective="objective",
                    runtime="runtime",
                    augment=None,
                ),
            )

            runtime.run(
                optimizer_setup=optimizer_setup,
                run_fs=run_fs,
            )

            self.assertEqual(train_viz_calls, [0, 1, 2])
            self.assertEqual(
                eval_store_roots,
                [
                    run_fs.artifacts_dir / "stitch_eval" / "epoch_0000",
                    run_fs.artifacts_dir / "stitch_eval" / "epoch_0001",
                    run_fs.artifacts_dir / "stitch_eval" / "epoch_0002",
                ],
            )
            self.assertTrue((run_fs.artifacts_dir / "stitch_train" / "epoch_0001" / "segA.png").is_file())
            self.assertTrue((run_fs.artifacts_dir / "stitch_train" / "epoch_0001" / "segA.json").is_file())
            self.assertEqual(
                eval_preview_paths,
                [
                    run_fs.artifacts_dir / "stitch_eval" / "epoch_0000" / "segA__prob.png",
                    run_fs.artifacts_dir / "stitch_eval" / "epoch_0001" / "segA__prob.png",
                    run_fs.artifacts_dir / "stitch_eval" / "epoch_0002" / "segA__prob.png",
                ],
            )
            self.assertTrue((run_fs.artifacts_dir / "stitch_eval" / "epoch_0000" / "segA__prob.png").is_file())
            self.assertTrue((run_fs.artifacts_dir / "stitch_eval" / "epoch_0001" / "segA__prob.png").is_file())
            self.assertTrue((run_fs.artifacts_dir / "stitch_eval" / "epoch_0002" / "segA__prob.png").is_file())

    def test_group_stratified_sampler_runtime_keeps_batches_group_mixed(self):
        train = InMemoryPatchSamples(
            images=(
                _image(16),
                _image(24),
                _image(32),
                _image(40),
                _image(48),
                _image(56),
            ),
            labels=(
                _label(255),
                _label(255),
                _label(255),
                _label(0),
                _label(0),
                _label(0),
            ),
            groups=(0, 0, 0, 1, 1, 1),
        )
        val = InMemoryPatchSamples(
            images=(_image(64),),
            labels=(_label(255),),
            xyxys=((0, 0, 8, 8),),
            groups=(0,),
        )
        bundle = GroupedInMemoryPatchDataRecipe(
            train=train,
            val=val,
            in_channels=1,
            patch_size=8,
            train_batch_size=2,
            valid_batch_size=1,
            sampler=GroupStratifiedSampler(batch_size=2, seed=3),
            normalization=ClipMaxDiv255Normalization(),
        ).build(augment=_neutral_augment())
        objective = _RecordingObjective()
        model = _TrainableDownsampleBiasModel(init_bias=1.0)
        runtime = _assemble_patch_training(
            Experiment(
                name="stratified_runtime",
                data="data",
                model=model,
                loss=_QuadraticRegionLoss(),
                objective=objective,
                runtime=_LoopRuntime(None, grad_accum=1),
                augment=None,
            ),
            bundle,
        )
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")

        result = runtime.run_epoch(optimizer_setup=optimizer_setup)

        self.assertEqual(result.batches, 3)
        self.assertEqual(len(objective.calls), 3)
        self.assertEqual(
            [sorted(int(group_idx) for group_idx in call["group_idx"].tolist()) for call in objective.calls],
            [[0, 1], [0, 1], [0, 1]],
        )

    def test_run_training_epoch_accepts_flat_patch_data_batches(self):
        train = InMemoryPatchSamples(
            images=(_image(16), _image(32)),
            labels=(_label(255), _label(0)),
        )
        val = InMemoryPatchSamples(
            images=(_image(64),),
            labels=(_label(255),),
            xyxys=((0, 0, 8, 8),),
        )
        bundle = InMemoryPatchDataRecipe(
            train=train,
            val=val,
            in_channels=1,
            patch_size=8,
            train_batch_size=2,
            valid_batch_size=1,
            normalization=ClipMaxDiv255Normalization(),
        ).build(augment=_neutral_augment())
        objective = _RecordingObjective()
        model = _TrainableDownsampleBiasModel(init_bias=1.0)
        runtime = _assemble_patch_training(
            Experiment(
                name="flat_runtime",
                data="data",
                model=model,
                loss=_QuadraticRegionLoss(),
                objective=objective,
                runtime=_LoopRuntime(None, grad_accum=1),
                augment=None,
            ),
            bundle,
        )
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")

        result = runtime.run_epoch(optimizer_setup=optimizer_setup)

        self.assertEqual(result.batches, 1)
        self.assertEqual(len(objective.calls), 1)
        self.assertIsNone(objective.calls[0]["group_idx"])

    def test_run_training_loads_init_checkpoint_model_weights(self):
        runtime, model, _objective = self._make_runtime(grad_accum=1, epochs=0)
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")

        with tempfile.TemporaryDirectory() as tmpdir:
            init_ckpt_path = Path(tmpdir) / "init.pt"
            torch.save({"model": {"bias": torch.tensor(7.0)}}, init_ckpt_path)
            runtime.init_ckpt_path = str(init_ckpt_path)

            result = runtime.run(optimizer_setup=optimizer_setup)

        self.assertEqual(len(result.train_epochs), 0)
        self.assertAlmostEqual(float(model.bias.detach().item()), 7.0)

    def test_run_training_resume_checkpoint_restores_optimizer_and_epoch_offset(self):
        model = _TrainableBiasModel(init_bias=0.0)
        objective = _RecordingObjective()
        runtime = _assemble_patch_training(
            Experiment(
                name="resume_runtime",
                data="data",
                model=model,
                loss=_QuadraticRegionLoss(),
                objective=objective,
                runtime=_LoopRuntime(None, grad_accum=1),
                augment=None,
                trainer=PatchTrainer(epochs=2),
            ),
            DataBundle(
                train_loader=[],
                eval_loader=[],
                in_channels=1,
                group_counts=[1],
            ),
        )
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")

        with tempfile.TemporaryDirectory() as tmpdir:
            resume_optimizer = torch.optim.SGD(model.parameters(), lr=0.7, momentum=0.3)
            resume_ckpt_path = Path(tmpdir) / "resume.pt"
            torch.save(
                {
                    "model": {"bias": torch.tensor(5.0)},
                    "optimizer": resume_optimizer.state_dict(),
                    "epoch": 4,
                },
                resume_ckpt_path,
            )
            runtime.resume_ckpt_path = str(resume_ckpt_path)
            run_fs = RunFS(
                Path(tmpdir) / "run",
                Experiment(
                    name=runtime.experiment.name,
                    data="data",
                    model="model",
                    loss="loss",
                    objective="objective",
                    runtime="runtime",
                    augment=None,
                ),
            )

            result = runtime.run(
                optimizer_setup=optimizer_setup,
                run_fs=run_fs,
            )

            history_lines = run_fs.history_path.read_text(encoding="utf-8").strip().splitlines()
            history = [json.loads(line) for line in history_lines]
            last_payload = torch.load(run_fs.ckpt_dir / "last.pt", map_location="cpu")

        self.assertEqual(len(result.train_epochs), 2)
        self.assertEqual([entry["epoch"] for entry in history], [5, 6])
        self.assertAlmostEqual(float(model.bias.detach().item()), 5.0)
        self.assertAlmostEqual(float(optimizer_setup.optimizer.param_groups[0]["lr"]), 0.7)
        self.assertEqual(last_payload["epoch"], 6)

    def test_run_training_logs_metrics_and_local_paths_to_wandb(self):
        runtime, model, _objective = self._make_runtime(grad_accum=1, epochs=2, eval_every=1)
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")
        runtime.experiment.runtime.wandb = WandbLogger(
            enabled=True,
            project="ink-tests",
            run_name="wandb-run",
            tags=("unit",),
            log_train_every_n_steps=1,
        )
        fake_run = _FakeWandbRun()
        init_calls = []

        def fake_init(**kwargs):
            init_calls.append(dict(kwargs))
            return fake_run

        with tempfile.TemporaryDirectory() as tmpdir:
            run_fs = RunFS(
                Path(tmpdir) / "run",
                Experiment(
                    name=runtime.experiment.name,
                    data="data",
                    model="model",
                    loss="loss",
                    objective="objective",
                    runtime="runtime",
                    augment=None,
                ),
            )

            with patch.dict("sys.modules", {"wandb": SimpleNamespace(init=fake_init, Image=_FakeWandbImage)}):
                class _Evaluator:
                    def evaluate(self, _model, _loader, *, device=None):
                        del _model, _loader, device
                        return EvalReport(summary={"val/dice": 0.75})

                runtime.experiment = replace(runtime.experiment, evaluator=_Evaluator())
                runtime.run(
                    optimizer_setup=optimizer_setup,
                    run_fs=run_fs,
                )

            logged_keys = {key for entry in fake_run.logged for key in entry["payload"].keys()}

        self.assertEqual(len(init_calls), 1)
        self.assertEqual(init_calls[0]["project"], "ink-tests")
        self.assertEqual(init_calls[0]["name"], "wandb-run")
        self.assertIn("config", init_calls[0])
        self.assertEqual(fake_run.summary["local/run_dir"], str(run_fs.run_dir))
        self.assertEqual(fake_run.summary["local/checkpoints_dir"], str(run_fs.ckpt_dir))
        self.assertIn("train/loss", logged_keys)
        self.assertIn("train_epoch/loss", logged_keys)
        self.assertIn("trainer/global_step", logged_keys)
        self.assertIn("train/lr", logged_keys)
        self.assertIn("val/dice", logged_keys)
        step_entries = [entry for entry in fake_run.logged if "train/loss" in entry["payload"]]
        epoch_entries = [entry for entry in fake_run.logged if "train_epoch/loss" in entry["payload"]]
        self.assertTrue(step_entries)
        self.assertTrue(epoch_entries)
        self.assertTrue(all("trainer/global_step" in entry["payload"] for entry in step_entries))
        self.assertTrue(all("trainer/global_step" not in entry["payload"] for entry in epoch_entries))
        self.assertTrue(fake_run.finished)

    def test_run_training_logs_step_progress_on_logger_cadence(self):
        runtime, model, _objective = self._make_runtime(grad_accum=1, epochs=1, eval_every=1)
        messages: list[str] = []
        runtime.log = messages.append
        runtime.log_every_n_steps = 2
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")

        runtime.run(optimizer_setup=optimizer_setup)

        self.assertTrue(any("step=2/3" in message for message in messages))
        self.assertTrue(any("it_s=" in message for message in messages))

    def test_run_training_can_log_eval_group_and_segment_metrics_to_wandb(self):
        runtime, model, _objective = self._make_runtime(grad_accum=1, epochs=1, eval_every=1)
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")
        runtime.experiment.runtime.wandb = WandbLogger(
            enabled=True,
            project="ink-tests",
            run_name="wandb-run",
            log_eval_by_group=True,
            log_eval_by_segment=True,
        )
        fake_run = _FakeWandbRun()

        def fake_init(**kwargs):
            del kwargs
            return fake_run

        with tempfile.TemporaryDirectory() as tmpdir:
            run_fs = RunFS(
                Path(tmpdir) / "run",
                Experiment(
                    name=runtime.experiment.name,
                    data="data",
                    model="model",
                    loss="loss",
                    objective="objective",
                    runtime="runtime",
                    augment=None,
                ),
            )

            with patch.dict("sys.modules", {"wandb": SimpleNamespace(init=fake_init, Image=_FakeWandbImage)}):
                class _Evaluator:
                    def evaluate(self, _model, _loader, *, device=None):
                        del _model, _loader, device
                        return EvalReport(
                            summary={"val/stitch/Dice": 0.75},
                            by_group={"0": {"val/stitch/Dice": 0.5}},
                            by_segment={"segA#component_0": {"val/stitch/Dice": 1.0}},
                        )

                runtime.experiment = replace(runtime.experiment, evaluator=_Evaluator())
                runtime.run(
                    optimizer_setup=optimizer_setup,
                    run_fs=run_fs,
                )

        logged_keys = {key for entry in fake_run.logged for key in entry["payload"].keys()}
        self.assertIn("val/stitch/Dice", logged_keys)
        self.assertIn("group/0/val/stitch/Dice", logged_keys)
        self.assertIn("segment/segA#component_0/val/stitch/Dice", logged_keys)

    def test_run_training_logs_stitched_eval_preview_images_to_wandb(self):
        runtime, model, _objective = self._make_runtime(grad_accum=1, epochs=1, eval_every=1)
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")
        runtime.experiment.runtime.wandb = WandbLogger(
            enabled=True,
            project="ink-tests",
            run_name="wandb-run",
            media_downsample=2,
        )
        fake_run = _FakeWandbRun()

        def fake_init(**kwargs):
            del kwargs
            return fake_run

        class _Store:
            def __init__(self):
                self.root_dir = Path("initial")
                self.downsample = 1

            def segment_ids(self):
                return ("segA",)

            def full_segment_prob_preview_u8(self, *, segment_id, media_downsample=1):
                del segment_id, media_downsample
                return np.array([[64, 128], [192, 255]], dtype=np.uint8)

            def write_full_segment_preview_png(self, *, segment_id, media_downsample=1, image_u8=None):
                del media_downsample, image_u8
                path = Path(self.root_dir) / f"{str(segment_id).replace('/', '__')}__prob.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"png")
                return str(path)

        store = _Store()

        class _Evaluator:
            stage_prefix = "val"

            def __init__(self):
                self.stitch_inference = SimpleNamespace(store=store, stitch_runtime=SimpleNamespace(train=None))

            def prepare_run_artifacts(self, *, run_fs=None, epoch=None):
                if run_fs is not None:
                    root_dir = run_fs.artifacts_dir / "stitch_eval"
                    if epoch is not None:
                        root_dir = root_dir / f"epoch_{int(epoch):04d}"
                    self.stitch_inference.store.root_dir = root_dir

            def export_logged_images(self, *, media_downsample):
                del media_downsample
                self.stitch_inference.store.write_full_segment_preview_png(
                    segment_id="segA",
                    media_downsample=1,
                    image_u8=np.array([[127]], dtype=np.uint8),
                )
                return {
                    "stitch_eval/segA": {
                        "image": np.array([[127]], dtype=np.uint8),
                        "caption": "segA (val ds=1)",
                    }
                }

            def evaluate(self, _model, _loader, *, device=None):
                del _model, _loader, device
                return EvalReport(summary={"val/dice": 0.75})

        runtime.experiment = replace(runtime.experiment, evaluator=_Evaluator())

        with tempfile.TemporaryDirectory() as tmpdir:
            run_fs = RunFS(
                Path(tmpdir) / "run",
                Experiment(
                    name=runtime.experiment.name,
                    data="data",
                    model="model",
                    loss="loss",
                    objective="objective",
                    runtime="runtime",
                    augment=None,
                ),
            )
            with patch.dict("sys.modules", {"wandb": SimpleNamespace(init=fake_init, Image=_FakeWandbImage)}):
                runtime.run(
                    optimizer_setup=optimizer_setup,
                    run_fs=run_fs,
                )

        preview_entries = [
            entry["payload"]["stitch_eval/segA"]
            for entry in fake_run.logged
            if "stitch_eval/segA" in entry["payload"]
        ]
        self.assertEqual(len(preview_entries), 1)
        self.assertIsInstance(preview_entries[0], _FakeWandbImage)
        self.assertEqual(preview_entries[0].caption, "segA (val ds=1)")

    def test_run_training_logs_epochs_and_persists_last_and_best_checkpoints(self):
        runtime, model, _objective = self._make_runtime(grad_accum=1, epochs=3, eval_every=1)
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_fs = RunFS(
                Path(tmpdir) / "run",
                Experiment(
                    name=runtime.experiment.name,
                    data="data",
                    model="model",
                    loss="loss",
                    objective="objective",
                    runtime="runtime",
                    augment=None,
                ),
            )
            eval_calls = {"count": 0}

            class _Evaluator:
                def evaluate(self, bound_model, eval_loader, *, device=None):
                    del bound_model, eval_loader, device
                    eval_calls["count"] += 1
                    return EvalReport(summary={"val/dice": 0.3 + (0.1 * eval_calls["count"])})

            runtime.experiment = replace(runtime.experiment, evaluator=_Evaluator())
            result = runtime.run(
                optimizer_setup=optimizer_setup,
                run_fs=run_fs,
                checkpoint_extra_state=lambda epoch: {"tag": f"epoch-{epoch}"},
            )

            history_lines = run_fs.history_path.read_text(encoding="utf-8").strip().splitlines()
            history = [json.loads(line) for line in history_lines]

            self.assertEqual(len(result.train_epochs), 3)
            self.assertEqual(len(result.eval_epochs), 3)
            self.assertEqual([entry["split"] for entry in history], ["train", "val", "train", "val", "train", "val"])
            self.assertTrue((run_fs.ckpt_dir / "last.pt").is_file())
            self.assertTrue((run_fs.ckpt_dir / "best.pt").is_file())
            self.assertAlmostEqual(run_fs.best_metric, 0.6)

            last_payload = torch.load(run_fs.ckpt_dir / "last.pt", map_location="cpu")
            best_payload = torch.load(run_fs.ckpt_dir / "best.pt", map_location="cpu")
            self.assertEqual(last_payload["epoch"], 2)
            self.assertEqual(last_payload["tag"], "epoch-2")
            self.assertEqual(best_payload["epoch"], 2)
            self.assertEqual(best_payload["tag"], "epoch-2")

    def test_run_training_auto_selects_first_eval_summary_key_for_best_checkpoint(self):
        runtime, model, _objective = self._make_runtime(grad_accum=1, epochs=3, eval_every=1)
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_fs = RunFS(
                Path(tmpdir) / "run",
                Experiment(
                    name=runtime.experiment.name,
                    data="data",
                    model="model",
                    loss="loss",
                    objective="objective",
                    runtime="runtime",
                    augment=None,
                ),
            )
            eval_calls = {"count": 0}

            class _Evaluator:
                def evaluate(self, bound_model, eval_loader, *, device=None):
                    del bound_model, eval_loader, device
                    eval_calls["count"] += 1
                    return EvalReport(
                        summary={
                            "val/dice": 0.3 + (0.1 * eval_calls["count"]),
                            "val/balanced_accuracy": 0.1,
                        }
                    )

            runtime.experiment = replace(runtime.experiment, evaluator=_Evaluator())
            runtime.run(
                optimizer_setup=optimizer_setup,
                run_fs=run_fs,
                checkpoint_extra_state=lambda epoch: {"tag": f"epoch-{epoch}"},
            )

            self.assertTrue((run_fs.ckpt_dir / "best.pt").is_file())
            self.assertAlmostEqual(run_fs.best_metric, 0.6)
            best_payload = torch.load(run_fs.ckpt_dir / "best.pt", map_location="cpu")
            self.assertEqual(best_payload["epoch"], 2)
            self.assertEqual(best_payload["tag"], "epoch-2")

    def test_run_training_can_save_checkpoints_on_epoch_cadence(self):
        runtime, model, _objective = self._make_runtime(
            grad_accum=1,
            epochs=3,
            eval_every=1,
            save_every_n_epochs=2,
        )
        optimizer_setup, _scheduler = self._optimizer_setup(model, interval="step")

        with tempfile.TemporaryDirectory() as tmpdir:
            run_fs = RunFS(
                Path(tmpdir) / "run",
                Experiment(
                    name=runtime.experiment.name,
                    data="data",
                    model="model",
                    loss="loss",
                    objective="objective",
                    runtime="runtime",
                    augment=None,
                ),
            )

            runtime.run(
                optimizer_setup=optimizer_setup,
                run_fs=run_fs,
            )

            self.assertTrue((run_fs.ckpt_dir / "epoch_0001.pt").is_file())
            self.assertFalse((run_fs.ckpt_dir / "epoch_0000.pt").exists())
            self.assertFalse((run_fs.ckpt_dir / "epoch_0002.pt").exists())


class ValidationEvaluatorTests(unittest.TestCase):
    def test_patch_eval_requires_batch_inputs(self):
        evaluator = PatchEval(metrics=(_SimpleBatchMetric(),))
        model = _IdentityModel()
        eval_loader = [
            (
                torch.ones((1, 1, 1, 1), dtype=torch.float32),
                torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                torch.ones((1, 1, 1, 1), dtype=torch.float32),
                torch.tensor([[0, 0, 1, 1]], dtype=torch.long),
            ),
        ]

        with self.assertRaisesRegex(TypeError, "validation batch must be Batch"):
            ValidationEvaluator(patch=evaluator).evaluate(model, eval_loader)

    def test_patch_eval_supports_simple_compute_metrics(self):
        evaluator = PatchEval(metrics=(_SimpleBatchMetric(),))
        model = _IdentityModel()
        eval_loader = [
            Batch(
                x=torch.ones((1, 1, 1, 1), dtype=torch.float32),
                y=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                meta=BatchMeta(segment_ids=[], group_idx=None),
            ),
            Batch(
                x=2.0 * torch.ones((1, 1, 1, 1), dtype=torch.float32),
                y=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                meta=BatchMeta(segment_ids=[], group_idx=None),
            ),
        ]

        report = ValidationEvaluator(patch=evaluator).evaluate(model, eval_loader)

        self.assertAlmostEqual(report.stages["patch"].summary["metrics/val/simple_batch_metric"], 1.5)

    def test_patch_eval_logs_eval_progress_on_logger_cadence(self):
        messages: list[str] = []
        evaluator = ValidationEvaluator(
            patch=PatchEval(metrics=(_SimpleBatchMetric(),)),
            log_every_n_steps=2,
        )
        object.__setattr__(evaluator, "_logger", messages.append)
        model = _IdentityModel()
        eval_loader = [
            Batch(
                x=torch.ones((1, 1, 1, 1), dtype=torch.float32),
                y=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                meta=BatchMeta(segment_ids=[], group_idx=None),
            ),
            Batch(
                x=2.0 * torch.ones((1, 1, 1, 1), dtype=torch.float32),
                y=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                meta=BatchMeta(segment_ids=[], group_idx=None),
            ),
            Batch(
                x=3.0 * torch.ones((1, 1, 1, 1), dtype=torch.float32),
                y=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                meta=BatchMeta(segment_ids=[], group_idx=None),
            ),
        ]

        evaluator.evaluate(model, eval_loader)

        self.assertTrue(any("step=2/3" in message for message in messages))
        self.assertTrue(any("step=3/3" in message for message in messages))
        self.assertTrue(all("it_s=" in message for message in messages))

    def test_patch_eval_delegates_batch_iteration_and_report_assembly_to_metric_recipe(self):
        final_report = MetricReport(
            summary={"Dice": 0.75},
            by_group={"0": {"val/loss": 2.0, "val/count": 1.0}},
            by_segment={"segA": {"val/dice": 0.5, "val/count": 1.0}},
        )
        metric = _RecordingMetricRecipe(outputs=["batch-1", "batch-2"], final_report=final_report)
        evaluator = PatchEval(metrics=(metric,), n_groups=2)
        model = _ModeTrackingModel()
        eval_loader = [
            Batch(
                x=torch.zeros((2, 1, 1, 1), dtype=torch.float32),
                y=torch.zeros((2, 1, 1, 1), dtype=torch.float32),
                meta=BatchMeta(
                    segment_ids=["segA", "segB"],
                    valid_mask=torch.ones((2, 1, 1, 1), dtype=torch.float32),
                    group_idx=torch.tensor([0, 1], dtype=torch.long),
                ),
            ),
            Batch(
                x=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                y=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                meta=BatchMeta(
                    segment_ids=["segB"],
                    valid_mask=torch.ones((1, 1, 1, 1), dtype=torch.float32),
                    group_idx=torch.tensor([1], dtype=torch.long),
                ),
            ),
        ]

        report = ValidationEvaluator(patch=evaluator).evaluate(model, eval_loader)

        self.assertEqual(report.stages["patch"].summary["Dice"], 0.75)
        self.assertEqual(report.stages["patch"].by_group["0"]["val/loss"], 2.0)
        self.assertEqual(report.stages["patch"].by_segment["segA"]["val/dice"], 0.5)
        self.assertEqual(len(metric.calls), 2)
        self.assertEqual(metric.calls[0]["segment_ids"], ("segA", "segB"))
        self.assertEqual(metric.calls[1]["segment_ids"], ("segB",))
        self.assertTrue(torch.equal(metric.calls[0]["group_idx"], torch.tensor([0, 1], dtype=torch.long)))
        self.assertTrue(torch.equal(metric.calls[1]["group_idx"], torch.tensor([1], dtype=torch.long)))
        self.assertEqual(metric.states[0]["n_groups"], 2)
        self.assertEqual(metric.states[-1]["outputs"], ["batch-1", "batch-2"])

    def test_patch_eval_restores_model_train_mode(self):
        evaluator = PatchEval(
            metrics=(
                _RecordingMetricRecipe(
                    outputs=[],
                    final_report=MetricReport(summary={"Dice": 0.0}),
                ),
            )
        )
        model = _ModeTrackingModel()

        report = ValidationEvaluator(patch=evaluator).evaluate(model, [])

        self.assertTrue(model.training)
        self.assertEqual(model.eval_calls, 1)
        self.assertEqual(model.train_calls, 1)
        self.assertEqual(report.stages["patch"].summary["Dice"], 0.0)

    def test_validation_evaluator_reports_epoch_metrics_without_objective_dispatch(self):
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
        objective = _RecordingObjective()
        runtime = _assemble_patch_training(
            Experiment(
                name="val_runtime",
                data="data",
                model=_FixedLogitModel(logits),
                loss=DiceBCEBatch(),
                objective=objective,
                runtime=_LoopRuntime(None),
                augment=None,
                evaluator=ValidationEvaluator(
                    patch=PatchEval(
                        metrics=(
                            Dice(),
                            BalancedAccuracy(),
                            Dice(threshold=96.0 / 255.0, name="DiceHist96_255"),
                            BalancedAccuracy(threshold=96.0 / 255.0, name="BalancedAccuracyHist96_255"),
                        ),
                    ),
                ),
            ),
            DataBundle(
                train_loader=[],
                eval_loader=[batch],
                in_channels=1,
                group_counts=[1, 1],
            ),
        )

        raw_report = runtime.experiment.evaluator.evaluate(runtime.experiment.model, runtime.eval_loader)
        report = flatten_eval_report(raw_report, stage_prefix=runtime.experiment.evaluator.stage_prefix)

        self.assertEqual(objective.calls, [])
        self.assertEqual(len(runtime.experiment.runtime.precision_context_calls), 1)
        self.assertAlmostEqual(report.summary["val/patch/Dice"], 1.0)
        self.assertAlmostEqual(report.summary["val/patch/BalancedAccuracy"], 1.0)
        self.assertAlmostEqual(report.summary["val/patch/DiceHist96_255"], 1.0)
        self.assertAlmostEqual(report.summary["val/patch/BalancedAccuracyHist96_255"], 1.0)

    def test_validation_evaluator_object_integrates_with_run_training(self):
        logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]], dtype=torch.float32)
        targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
        batch = Batch(
            x=torch.zeros_like(logits),
            y=targets,
            meta=BatchMeta(
                segment_ids=["segA"],
                valid_mask=torch.ones_like(targets),
                patch_xyxy=torch.tensor([[0, 0, 2, 2]], dtype=torch.long),
                group_idx=torch.tensor([0], dtype=torch.long),
            ),
        )
        model = _TrainableBiasModel(init_bias=1.0)
        optimizer_setup, _scheduler = TrainingLoopOrchestrationTests()._optimizer_setup(model, interval="step")
        runtime = _assemble_patch_training(
            Experiment(
                name="val_object",
                data="data",
                model=model,
                loss=DiceBCEBatch(),
                objective=_RecordingObjective(),
                runtime=_LoopRuntime(optimizer_setup),
                augment=None,
                trainer=PatchTrainer(epochs=2, eval_every=1),
                evaluator=ValidationEvaluator(
                    patch=PatchEval(
                        metrics=(
                            Dice(),
                            BalancedAccuracy(),
                            Dice(threshold=96.0 / 255.0, name="DiceHist96_255"),
                            BalancedAccuracy(threshold=96.0 / 255.0, name="BalancedAccuracyHist96_255"),
                        ),
                    ),
                ),
            ),
            DataBundle(
                train_loader=[
                    Batch(
                        x=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                        y=torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                        meta=BatchMeta(
                            segment_ids=[],
                            valid_mask=torch.ones((1, 1, 1, 1), dtype=torch.float32),
                            group_idx=torch.tensor([0], dtype=torch.long),
                        ),
                    )
                ],
                eval_loader=[batch],
                in_channels=1,
                group_counts=[1],
            ),
        )

        result = runtime.run(optimizer_setup=optimizer_setup)

        self.assertEqual(len(result.eval_epochs), 2)
        self.assertIn("val/patch/Dice", result.eval_epochs[0].report.summary)


if __name__ == "__main__":
    unittest.main()
