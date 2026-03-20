from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from ink.core.device import move_batch_to_device
from ink.core.types import Batch, DataBundle, EvalReport
from ink.recipes.metrics import MetricBatch, merge_metric_reports


@dataclass(frozen=True, kw_only=True)
class PatchEval:
    metrics: tuple[Any, ...] = ()
    n_groups: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if self.n_groups is not None:
            object.__setattr__(self, "n_groups", int(self.n_groups))

    def build(self, *, data: DataBundle, runtime=None, stitch=None, logger=None) -> PatchEval:
        if not self.metrics:
            raise ValueError("PatchEval requires at least one metric")
        group_counts = dict(getattr(data, "extras", {}) or {}).get("group_counts")
        metrics = tuple(
            metric.build(
                data=data,
                runtime=runtime,
                stitch=stitch,
                logger=logger,
            )
            for metric in self.metrics
        )
        return replace(
            self,
            metrics=metrics,
            n_groups=None if group_counts is None else int(len(group_counts)),
        )

    def evaluate(self, model, val_loader, *, device=None, batch_observer=None) -> EvalReport:
        if not callable(model):
            raise TypeError("evaluation model must be callable")
        if not self.metrics:
            raise ValueError("PatchEval requires at least one metric")
        if batch_observer is not None and not callable(batch_observer):
            raise TypeError("batch_observer must be callable")
        if device is not None and hasattr(model, "to"):
            model.to(device)

        states = [metric.empty_state(n_groups=self.n_groups) for metric in self.metrics]

        was_training = bool(getattr(model, "training", False))
        if callable(getattr(model, "eval", None)):
            model.eval()

        try:
            with torch.inference_mode():
                for batch in val_loader:
                    if not isinstance(batch, Batch):
                        raise TypeError("validation batch must be Batch")
                    batch = move_batch_to_device(batch, device=device)
                    if batch.y is None:
                        raise ValueError("validation batch requires batch.y")

                    logits = model(batch.x)
                    metric_batch = MetricBatch(
                        logits=logits,
                        targets=batch.y,
                        valid_mask=batch.meta.valid_mask,
                        group_idx=batch.meta.group_idx,
                        segment_ids=tuple(batch.meta.segment_ids),
                        patch_xyxy=batch.meta.patch_xyxy,
                    )
                    if batch_observer is not None:
                        batch_observer(metric_batch)
                    shared = {}
                    for idx, metric in enumerate(self.metrics):
                        states[idx] = metric.update(states[idx], metric_batch, shared=shared)
        finally:
            if was_training and callable(getattr(model, "train", None)):
                model.train()

        reports = [metric.finalize(state) for metric, state in zip(self.metrics, states)]
        return merge_metric_reports(reports)
