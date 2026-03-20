from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _resolve_max_lr(optimizer, max_lr: float | None):
    if max_lr is not None:
        return float(max_lr)
    if len(optimizer.param_groups) == 1:
        return float(optimizer.param_groups[0]["lr"])
    return [float(group["lr"]) for group in optimizer.param_groups]


class GradualWarmupSchedulerV2(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = float(multiplier)
        self.total_epoch = int(total_epoch)
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler is not None:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            if self.total_epoch <= 0:
                return list(self.base_lrs)
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        if self.total_epoch <= 0:
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [
            base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: int | None = None):
        if self.finished and self.after_scheduler is not None:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
            return

        super().step(epoch)

        if self.last_epoch > self.total_epoch and self.after_scheduler is not None:
            if not self.finished:
                self.after_scheduler.base_lrs = [
                    base_lr * self.multiplier for base_lr in self.base_lrs
                ]
                self.finished = True
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()


@dataclass(frozen=True)
class SchedulerSetup:
    scheduler: Any
    interval: str


@dataclass(frozen=True)
class OneCycleScheduler:
    max_lr: float | None = None
    pct_start: float = 0.15
    div_factor: float = 25.0
    final_div_factor: float = 1e2

    def build(self, optimizer, *, steps_per_epoch: int, epochs: int) -> SchedulerSetup:
        return SchedulerSetup(
            scheduler=torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=_resolve_max_lr(optimizer, self.max_lr),
                pct_start=float(self.pct_start),
                steps_per_epoch=int(steps_per_epoch),
                epochs=int(epochs),
                div_factor=float(self.div_factor),
                final_div_factor=float(self.final_div_factor),
            ),
            interval="step",
        )


@dataclass(frozen=True)
class CosineScheduler:
    warmup_pct: float = 0.15
    min_lr: float = 1e-6
    warmup_factor: float = 10.0

    def build(self, optimizer, *, steps_per_epoch: int, epochs: int) -> SchedulerSetup:
        total_steps = max(1, int(steps_per_epoch) * int(epochs))
        warmup_pct = float(self.warmup_pct or 0.0)
        warmup_pct = max(0.0, min(1.0, warmup_pct))
        warmup_steps = int(round(total_steps * warmup_pct))
        warmup_steps = max(0, min(warmup_steps, total_steps - 1))
        eta_min = float(self.min_lr)

        if warmup_steps > 0:
            warmup_factor = float(self.warmup_factor or 1.0)
            if warmup_factor <= 0.0:
                raise ValueError(f"warmup_factor must be > 0, got {warmup_factor!r}")
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=float(1.0 / warmup_factor),
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(total_steps - warmup_steps),
                eta_min=eta_min,
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(total_steps),
                eta_min=eta_min,
            )

        return SchedulerSetup(scheduler=scheduler, interval="step")


@dataclass(frozen=True)
class GradualWarmupV2Scheduler:
    multiplier: float = 1.0
    total_epoch: int = 1
    cosine_t_max: int = 50
    eta_min: float = 1e-6

    def build(self, optimizer, *, steps_per_epoch: int, epochs: int) -> SchedulerSetup:
        del steps_per_epoch, epochs
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.cosine_t_max),
            eta_min=float(self.eta_min),
        )
        return SchedulerSetup(
            scheduler=GradualWarmupSchedulerV2(
                optimizer,
                multiplier=float(self.multiplier),
                total_epoch=int(self.total_epoch),
                after_scheduler=cosine,
            ),
            interval="epoch",
        )
