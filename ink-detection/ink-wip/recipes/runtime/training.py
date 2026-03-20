from __future__ import annotations

from contextlib import nullcontext
import math
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from ink.core.types import DataBundle
from ink.recipes.runtime.optimizers import AdamWOptimizer
from ink.recipes.runtime.schedulers import OneCycleScheduler, SchedulerSetup
from ink.recipes.trainers.support.logging import WandbLogger


@dataclass(frozen=True)
class OptimizerSetup:
    optimizer: Any
    scheduler: Any
    scheduler_interval: str


@dataclass(frozen=True)
class TrainRuntime:
    grad_accum: int = 1
    use_amp: bool = True
    precision: str | int | None = None
    grad_clip_norm: float | None = 100.0
    wandb: WandbLogger | None = None
    steps_per_epoch: int | None = None
    optimizer: Any = field(default_factory=AdamWOptimizer)
    scheduler: Any = field(default_factory=OneCycleScheduler)

    def resolve_steps_per_epoch(self, data: DataBundle) -> int | None:
        """Infer scheduler steps per epoch after gradient accumulation is applied."""
        if self.steps_per_epoch is not None:
            return int(self.steps_per_epoch)

        try:
            loader_length = len(data.train_loader)
        except TypeError:
            return None

        accum = max(1, int(self.grad_accum))
        return int(math.ceil(int(loader_length) / accum))

    def build(self, *, data: DataBundle, augment=None) -> TrainRuntime:
        """Bind runtime fields that depend on the prepared data bundle."""
        del augment
        assert self.wandb is None or isinstance(self.wandb, WandbLogger)
        return replace(
            self,
            precision=_resolve_training_precision(self.precision, use_amp=bool(self.use_amp)),
            steps_per_epoch=self.resolve_steps_per_epoch(data),
        )

    def build_optimizer_setup(self, model, *, epochs: int) -> OptimizerSetup:
        """Instantiate the optimizer and scheduler pair expected by the trainer."""
        optimizer = self._build_optimizer(model)
        scheduler_setup = self._build_scheduler(optimizer, epochs=int(epochs))
        return OptimizerSetup(
            optimizer=optimizer,
            scheduler=scheduler_setup.scheduler,
            scheduler_interval=scheduler_setup.interval,
        )

    def _build_optimizer(self, model):
        return self.optimizer.build(model)

    def _build_scheduler(self, optimizer, *, epochs: int):
        setup = self.scheduler.build(
            optimizer,
            steps_per_epoch=self._require_steps_per_epoch(),
            epochs=int(epochs),
        )
        assert isinstance(setup, SchedulerSetup)
        return setup

    def _require_steps_per_epoch(self) -> int:
        if self.steps_per_epoch is None:
            raise ValueError("steps_per_epoch must be resolved before building optimizer setup")
        return int(self.steps_per_epoch)

    def precision_context(self, *, device=None):
        """Return the autocast context for the configured precision and device."""
        precision = str(self.precision).strip().lower()
        if precision in {"", "32", "32-true"}:
            return nullcontext()
        if device is None:
            return nullcontext()

        device_type = torch.device(device).type
        if precision in {"16", "16-mixed", "fp16", "float16"}:
            if device_type == "cpu":
                return nullcontext()
            return torch.autocast(device_type=device_type, dtype=torch.float16)
        if precision in {"bf16", "bf16-mixed", "bfloat16"}:
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        raise ValueError(f"unsupported precision: {self.precision!r}")


def _resolve_training_precision(precision: str | int | None, *, use_amp: bool) -> str | int | None:
    """Map legacy use_amp behavior onto an explicit precision setting."""
    if precision is not None:
        precision_text = str(precision).strip().lower()
        if precision_text and precision_text != "auto":
            return precision
    return "16-mixed" if use_amp else "32-true"
