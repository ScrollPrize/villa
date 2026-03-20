from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from ink.core.device import move_batch_to_device
from ink.core.types import Batch, DataBundle, EvalReport
from ink.recipes.eval.metrics import MetricReport, loss_values as resolve_loss_values, merge_metric_reports
from ink.recipes.losses.reporting import resolve_train_output


def _to_per_sample_values(values, *, batch_size: int, key: str, device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        values = values.to(device=device, dtype=torch.float32)
    else:
        values = torch.as_tensor(values, dtype=torch.float32, device=device)
    if values.ndim == 0:
        return values.expand(batch_size).clone()
    if values.ndim > 1 and int(values.shape[0]) == batch_size:
        return values.reshape(batch_size, -1).mean(dim=1)
    values = values.reshape(-1)
    if int(values.numel()) == 1 and batch_size > 1:
        return values.expand(batch_size).clone()
    if int(values.numel()) != batch_size:
        raise ValueError(f"{key} must resolve to scalar or {batch_size} per-sample values, got {tuple(values.shape)}")
    return values


def _patch_loss_key(patch_loss) -> str:
    key = getattr(patch_loss, "__name__", None)
    if not callable(patch_loss) or not isinstance(key, str) or not key:
        key = getattr(type(patch_loss), "__name__", "") or "loss"
    return f"val/{key}"


def _report_metric_key(key) -> str:
    key = str(key)
    if key.startswith("metrics/") or key.startswith("val/"):
        return key
    return f"val/{key}"


def _accumulate_metric_total(by_key, *, key: str, values: torch.Tensor) -> None:
    totals = by_key.setdefault(key, {"total": 0.0, "count": 0.0})
    totals["total"] += float(values.sum().item())
    totals["count"] += float(values.numel())


def _totals_report(by_key) -> MetricReport:
    return MetricReport(
        summary={
            str(key): 0.0 if float(totals["count"]) <= 0.0 else float(totals["total"] / totals["count"])
            for key, totals in by_key.items()
        }
    )


@dataclass(frozen=True)
class PatchValidationEvaluator:
    metrics: tuple[Any, ...] = ()
    patch_loss: Any = None
    n_groups: int | None = None

    def __post_init__(self) -> None:
        if self.n_groups is not None:
            object.__setattr__(self, "n_groups", int(self.n_groups))

    def build(self, *, data: DataBundle, runtime=None, stitch=None, logger=None, patch_loss=None) -> PatchValidationEvaluator:
        patch_loss = self.patch_loss if self.patch_loss is not None else patch_loss
        if not self.metrics and patch_loss is None:
            raise ValueError("PatchValidationEvaluator requires metrics or patch_loss in Experiment")
        group_counts = dict(getattr(data, "extras", {}) or {}).get("group_counts")
        metrics = tuple(
            metric.build(
                data=data,
                runtime=runtime,
                stitch=stitch,
                logger=logger,
                patch_loss=patch_loss,
            )
            for metric in self.metrics
        )
        return replace(
            self,
            metrics=metrics,
            patch_loss=patch_loss,
            n_groups=None if group_counts is None else int(len(group_counts)),
        )

    def evaluate(self, model, val_loader, *, device=None) -> EvalReport:
        if not callable(model):
            raise TypeError("evaluation model must be callable")
        if device is not None and hasattr(model, "to"):
            model.to(device)
        if not self.metrics and self.patch_loss is None:
            raise ValueError("PatchValidationEvaluator requires metrics or patch_loss in Experiment")

        metrics = self.metrics
        states = [metric.empty_state(n_groups=self.n_groups) for metric in metrics]
        auto_loss_totals = None
        patch_loss_key = None
        if self.patch_loss is not None:
            patch_loss_key = _patch_loss_key(self.patch_loss)
            auto_loss_totals = {}

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
                    valid_mask = batch.meta.valid_mask
                    shared = {}
                    if auto_loss_totals is not None:
                        output = resolve_train_output(
                            self.patch_loss,
                            logits,
                            batch.y,
                            valid_mask=valid_mask,
                        )
                        batch_size = int(batch.y.shape[0])

                        loss_values = resolve_loss_values(
                            self.patch_loss,
                            logits,
                            batch.y,
                            valid_mask=valid_mask,
                            shared=shared,
                        )
                        loss_values = _to_per_sample_values(
                            loss_values,
                            batch_size=batch_size,
                            key="loss_values",
                            device=logits.device,
                        )
                        _accumulate_metric_total(auto_loss_totals, key=patch_loss_key, values=loss_values)

                        for key, value in output.metrics.items():
                            metric_values = _to_per_sample_values(
                                value,
                                batch_size=batch_size,
                                key=str(key),
                                device=logits.device,
                            )
                            _accumulate_metric_total(
                                auto_loss_totals,
                                key=_report_metric_key(key),
                                values=metric_values,
                            )

                    for idx, metric in enumerate(metrics):
                        kwargs = {
                            "valid_mask": valid_mask,
                            "group_idx": batch.meta.group_idx,
                            "segment_ids": batch.meta.segment_ids,
                        }
                        if bool(getattr(metric, "uses_shared_batch", False)):
                            kwargs["shared"] = shared
                        states[idx] = metric.update(
                            states[idx],
                            logits,
                            batch.y,
                            **kwargs,
                        )
        finally:
            if was_training and callable(getattr(model, "train", None)):
                model.train()

        reports = [metric.finalize(state) for metric, state in zip(metrics, states)]
        if auto_loss_totals is not None:
            reports.append(_totals_report(auto_loss_totals))
        return merge_metric_reports(reports)
