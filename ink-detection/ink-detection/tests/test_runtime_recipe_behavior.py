from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from ink.core import DataBundle
from ink.recipes.runtime import (
    AdamWOptimizer,
    CosineScheduler,
    GradualWarmupV2Scheduler,
    MuonOptimizer,
    OneCycleScheduler,
    SGDOptimizer,
    TrainRuntime,
)
from ink.recipes.trainers.support import WandbLogger


class _OptimizationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.norm = nn.LayerNorm(3)

    def forward(self, x):
        return self.norm(self.linear(x))


class _MixedShapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(4)
        self.head = nn.Linear(4 * 4 * 4, 2)

    def forward(self, x):
        x = self.norm(self.conv(x))
        x = x.flatten(1)
        return self.head(x)


def _runtime_bundle(loader_length: int) -> DataBundle:
    return DataBundle(
        train_loader=list(range(loader_length)),
        eval_loader="eval_loader",
        in_channels=62,
        extras={},
    )


def _optimizer_group_signature(optimizer):
    signature = []
    for group in optimizer.param_groups:
        signature.append(
            (
                float(group["weight_decay"]),
                len(group["params"]),
                tuple(sorted(int(getattr(param, "ndim", 0)) for param in group["params"])),
            )
        )
    return signature


class TrainRuntimeContractTests(unittest.TestCase):
    def test_build_resolves_precision_and_budget_from_loader_length(self):
        recipe = TrainRuntime(grad_accum=4, use_amp=False, grad_clip_norm=7.5)

        bound = recipe.build(data=_runtime_bundle(10))

        self.assertEqual(bound.precision, "32-true")
        self.assertEqual(bound.steps_per_epoch, 3)
        self.assertEqual(bound.grad_accum, 4)
        self.assertEqual(bound.grad_clip_norm, 7.5)
        self.assertIsInstance(bound.optimizer, AdamWOptimizer)
        self.assertIsInstance(bound.scheduler, OneCycleScheduler)

    def test_build_preserves_explicit_precision_and_step_budget_override(self):
        recipe = TrainRuntime(grad_accum=8, precision="bf16-mixed", steps_per_epoch=7)

        bound = recipe.build(data=_runtime_bundle(100))

        self.assertEqual(bound.precision, "bf16-mixed")
        self.assertEqual(bound.steps_per_epoch, 7)

    def test_build_normalizes_wandb_logger_config(self):
        recipe = TrainRuntime(
            wandb=WandbLogger(
                enabled=True,
                project="ink-tests",
                tags=("unit", "runtime"),
                run_name="runtime-run",
                media_downsample=4,
            ),
        )

        bound = recipe.build(data=_runtime_bundle(2))

        self.assertIsInstance(bound.wandb, WandbLogger)
        self.assertEqual(bound.wandb.project, "ink-tests")
        self.assertEqual(bound.wandb.tags, ("unit", "runtime"))
        self.assertEqual(bound.wandb.run_name, "runtime-run")
        self.assertEqual(bound.wandb.media_downsample, 4)

    def test_build_rejects_invalid_wandb_logger_config_type(self):
        recipe = TrainRuntime(wandb=object())

        with self.assertRaises(AssertionError):
            recipe.build(data=_runtime_bundle(2))

    def test_adamw_optimizer_setup_uses_param_groups_and_onecycle_budget(self):
        recipe = TrainRuntime(
            steps_per_epoch=6,
            optimizer=AdamWOptimizer(
                lr=2e-4,
                weight_decay=1e-2,
                beta2=0.95,
                eps=1e-7,
                exclude_weight_decay_bias_norm=True,
            ),
            scheduler=OneCycleScheduler(
                pct_start=0.2,
                div_factor=10.0,
                final_div_factor=1e3,
            ),
        )

        setup = recipe.build(data=_runtime_bundle(6)).build_optimizer_setup(_OptimizationModel(), epochs=5)

        self.assertIsInstance(setup.optimizer, torch.optim.AdamW)
        self.assertEqual(setup.scheduler_interval, "step")
        self.assertEqual(setup.optimizer.defaults["lr"], 2e-4)
        self.assertEqual(setup.optimizer.defaults["betas"], (0.9, 0.95))
        self.assertEqual(setup.optimizer.defaults["eps"], 1e-7)
        self.assertEqual(
            _optimizer_group_signature(setup.optimizer),
            [(0.01, 1, (2,)), (0.0, 3, (1, 1, 1))],
        )
        self.assertIsInstance(setup.scheduler, torch.optim.lr_scheduler.OneCycleLR)
        self.assertEqual(setup.scheduler.total_steps, 30)

    def test_sgd_optimizer_setup_uses_cosine_with_warmup(self):
        recipe = TrainRuntime(
            steps_per_epoch=5,
            optimizer=SGDOptimizer(
                lr=1e-3,
                weight_decay=1e-2,
                momentum=0.85,
                nesterov=True,
                exclude_weight_decay_bias_norm=False,
            ),
            scheduler=CosineScheduler(
                warmup_pct=0.25,
                min_lr=1e-5,
                warmup_factor=5.0,
            ),
        )

        setup = recipe.build(data=_runtime_bundle(5)).build_optimizer_setup(_OptimizationModel(), epochs=4)
        scheduler = setup.scheduler

        self.assertIsInstance(setup.optimizer, torch.optim.SGD)
        self.assertEqual(setup.scheduler_interval, "step")
        self.assertEqual(setup.optimizer.defaults["momentum"], 0.85)
        self.assertEqual(setup.optimizer.defaults["nesterov"], True)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        self.assertEqual(list(scheduler._milestones), [5])
        self.assertEqual(
            [type(inner).__name__ for inner in scheduler._schedulers],
            ["LinearLR", "CosineAnnealingLR"],
        )
        self.assertEqual(scheduler._schedulers[0].total_iters, 5)
        self.assertEqual(scheduler._schedulers[1].T_max, 15)
        self.assertEqual(scheduler._schedulers[1].eta_min, 1e-5)

    def test_gradual_warmup_scheduler_setup_uses_epoch_interval(self):
        recipe = TrainRuntime(
            steps_per_epoch=8,
            optimizer=AdamWOptimizer(),
            scheduler=GradualWarmupV2Scheduler(),
        )

        setup = recipe.build(data=_runtime_bundle(8)).build_optimizer_setup(_OptimizationModel(), epochs=30)
        scheduler = setup.scheduler

        self.assertEqual(setup.scheduler_interval, "epoch")
        self.assertEqual(type(scheduler).__name__, "GradualWarmupSchedulerV2")
        self.assertEqual(scheduler.multiplier, 1.0)
        self.assertEqual(scheduler.total_epoch, 1)
        self.assertIsInstance(scheduler.after_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        self.assertEqual(scheduler.after_scheduler.T_max, 50)
        self.assertEqual(scheduler.after_scheduler.eta_min, 1e-6)

    def test_muon_optimizer_builds_and_steps_on_mixed_parameter_shapes(self):
        recipe = TrainRuntime(
            steps_per_epoch=3,
            optimizer=MuonOptimizer(
                lr=1e-3,
                weight_decay=1e-4,
                momentum=0.9,
                nesterov=True,
                ns_steps=3,
                adamw_lr=5e-4,
            ),
            scheduler=OneCycleScheduler(max_lr=1e-3),
        )

        model = _MixedShapeModel()
        setup = recipe.build(data=_runtime_bundle(3)).build_optimizer_setup(model, epochs=2)
        x = torch.randn(2, 3, 4, 4)
        y = torch.randn(2, 2)
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        setup.optimizer.step()

        self.assertEqual(type(setup.optimizer).__name__, "Muon")
        self.assertEqual(setup.scheduler_interval, "step")
        self.assertEqual(setup.optimizer.defaults["lr"], 1e-3)
        self.assertEqual(setup.optimizer.defaults["adamw_lr"], 5e-4)
        self.assertEqual(setup.optimizer.defaults["momentum"], 0.9)
        self.assertEqual(setup.optimizer.defaults["ns_steps"], 3)
        self.assertIsInstance(setup.scheduler, torch.optim.lr_scheduler.OneCycleLR)

    def test_invalid_optimizer_and_scheduler_raise_clear_errors(self):
        runtime = TrainRuntime(optimizer=object(), steps_per_epoch=3).build(data=_runtime_bundle(3))
        with self.assertRaises(AttributeError):
            runtime.build_optimizer_setup(_OptimizationModel(), epochs=1)

        runtime = TrainRuntime(scheduler=object(), steps_per_epoch=3).build(data=_runtime_bundle(3))
        with self.assertRaises(AttributeError):
            runtime.build_optimizer_setup(_OptimizationModel(), epochs=1)


if __name__ == "__main__":
    unittest.main()
