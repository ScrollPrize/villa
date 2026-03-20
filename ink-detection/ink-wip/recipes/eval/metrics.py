from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch

from ink.core.types import EvalReport
from ink.recipes.losses.reporting import loss_values as resolve_loss_values


@dataclass(frozen=True)
class MetricReport:
    summary: dict[str, float] = field(default_factory=dict)
    by_group: dict[str, dict[str, float]] = field(default_factory=dict)
    by_segment: dict[str, dict[str, float]] = field(default_factory=dict)


def _merge_metric_map(
    current: dict[str, float],
    incoming: dict[str, float],
    *,
    where: str,
) -> dict[str, float]:
    merged = dict(current)
    for key, value in incoming.items():
        key = str(key)
        if key in merged:
            raise ValueError(f"duplicate metric key {key!r} while merging {where}")
        merged[key] = float(value)
    return merged


def _merge_nested_metric_map(
    current: dict[str, dict[str, float]],
    incoming: dict[str, dict[str, float]],
    *,
    where: str,
) -> dict[str, dict[str, float]]:
    merged = {str(k): dict(v) for k, v in current.items()}
    for entity, metrics in incoming.items():
        entity_key = str(entity)
        merged[entity_key] = _merge_metric_map(
            merged.get(entity_key, {}),
            metrics,
            where=f"{where}[{entity_key!r}]",
        )
    return merged


def merge_metric_reports(reports: list[MetricReport]) -> EvalReport:
    summary: dict[str, float] = {}
    by_group: dict[str, dict[str, float]] = {}
    by_segment: dict[str, dict[str, float]] = {}
    for report in reports:
        summary = _merge_metric_map(summary, report.summary, where="summary")
        by_group = _merge_nested_metric_map(by_group, report.by_group, where="by_group")
        by_segment = _merge_nested_metric_map(by_segment, report.by_segment, where="by_segment")
    return EvalReport(summary=summary, by_group=by_group, by_segment=by_segment)


def _safe_div(numer: torch.Tensor, denom: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return numer / (denom + eps)


def _safe_average(total: float, count: float) -> float:
    if count <= 0.0:
        return 0.0
    return float(total / count)


@dataclass(frozen=True)
class ConfusionCounts:
    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor
    tn: torch.Tensor


def zero_confusion_counts(*, device=None) -> ConfusionCounts:
    tensor_kwargs = {}
    if device is not None:
        tensor_kwargs["device"] = device
    return ConfusionCounts(
        tp=torch.zeros((), dtype=torch.float64, **tensor_kwargs),
        fp=torch.zeros((), dtype=torch.float64, **tensor_kwargs),
        fn=torch.zeros((), dtype=torch.float64, **tensor_kwargs),
        tn=torch.zeros((), dtype=torch.float64, **tensor_kwargs),
    )


def add_confusion_counts(left: ConfusionCounts, right: ConfusionCounts) -> ConfusionCounts:
    return ConfusionCounts(
        tp=left.tp + right.tp,
        fp=left.fp + right.fp,
        fn=left.fn + right.fn,
        tn=left.tn + right.tn,
    )


def confusion_counts(preds: torch.Tensor, targets: torch.Tensor) -> ConfusionCounts:
    preds = preds.bool()
    targets = targets.bool()
    return ConfusionCounts(
        tp=(preds & targets).sum(dtype=torch.float64),
        fp=(preds & ~targets).sum(dtype=torch.float64),
        fn=(~preds & targets).sum(dtype=torch.float64),
        tn=(~preds & ~targets).sum(dtype=torch.float64),
    )


def dice_from_counts(counts: ConfusionCounts) -> torch.Tensor:
    return _safe_div(2.0 * counts.tp, 2.0 * counts.tp + counts.fp + counts.fn)


def _recall_or_nan(tp_like: torch.Tensor, fn_like: torch.Tensor) -> torch.Tensor:
    denom = tp_like + fn_like
    return torch.where(denom > 0.0, tp_like / denom, torch.full_like(denom, torch.nan))


def balanced_accuracy_from_counts(counts: ConfusionCounts) -> torch.Tensor:
    pos_recall = _recall_or_nan(counts.tp, counts.fn)
    neg_recall = _recall_or_nan(counts.tn, counts.fp)
    recalls = torch.stack((pos_recall, neg_recall))
    valid = ~torch.isnan(recalls)
    if bool(valid.any()):
        return recalls[valid].mean()
    return torch.zeros((), device=recalls.device, dtype=recalls.dtype)


def _masked_confusion_counts(logits, targets, *, threshold: float, valid_mask=None) -> ConfusionCounts:
    if tuple(logits.shape) != tuple(targets.shape):
        raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")

    logits = logits.detach()
    targets = targets.detach()
    if valid_mask is not None:
        valid_mask = valid_mask.detach().bool()
        if tuple(valid_mask.shape) != tuple(targets.shape):
            raise ValueError(f"valid_mask shape mismatch: {tuple(valid_mask.shape)} vs {tuple(targets.shape)}")
        logits = logits[valid_mask]
        targets = targets[valid_mask]

    if int(targets.numel()) == 0:
        return zero_confusion_counts(device=logits.device)

    preds = torch.sigmoid(logits).to(dtype=torch.float32) >= float(threshold)
    return confusion_counts(preds, targets.to(dtype=torch.float32) >= 0.5)


def _cached_reporting_values(shared, *, patch_loss, cache_tag: str, compute_fn):
    if shared is None:
        return compute_fn()
    cache = shared.setdefault("patch_loss_values", {})
    cache_key = (int(id(patch_loss)), str(cache_tag))
    if cache_key not in cache:
        cache[cache_key] = compute_fn()
    return cache[cache_key]


def loss_values(patch_loss, logits, targets, *, valid_mask=None, shared=None) -> torch.Tensor:
    return _cached_reporting_values(
        shared,
        patch_loss=patch_loss,
        cache_tag="loss_values",
        compute_fn=lambda: resolve_loss_values(
            patch_loss,
            logits,
            targets,
            valid_mask=valid_mask,
        ),
    )
region_loss_values = loss_values


def _resolve_metric_values(patch_loss, value_fn, logits, targets, *, valid_mask=None, shared=None) -> list[float]:
    if not callable(value_fn):
        raise TypeError("region metric value_fn must be callable")
    values = value_fn(
        patch_loss,
        logits,
        targets,
        valid_mask=valid_mask,
        shared=shared,
    )
    if not isinstance(values, torch.Tensor):
        values = torch.as_tensor(values, dtype=torch.float32, device=logits.device)
    values = values.detach().reshape(-1)
    expected = int(targets.shape[0])
    if int(values.numel()) != expected:
        raise ValueError(f"region metric value_fn output must contain {expected} values, got {int(values.numel())}")
    return [float(v) for v in values.tolist()]


def _resolve_group_values(group_idx, *, batch_size: int) -> list[int] | None:
    if group_idx is None:
        return None
    values = group_idx.detach().reshape(-1)
    if int(values.numel()) != int(batch_size):
        raise ValueError(f"group_idx length must match batch size, got {int(values.numel())} vs {batch_size}")
    return [int(v) for v in values.tolist()]


def _group_ids(*, by_group: dict[int, Any], n_groups: int | None):
    if n_groups is None:
        return sorted(by_group.keys())
    return range(int(n_groups))


def _resolve_batch_size(targets) -> int | None:
    if hasattr(targets, "shape") and len(targets.shape) > 0:
        return int(targets.shape[0])
    return None


@dataclass(frozen=True)
class _RunningTotal:
    total: float = 0.0
    count: float = 0.0


def _add_running_total(left: _RunningTotal, right: _RunningTotal) -> _RunningTotal:
    return _RunningTotal(
        total=float(left.total + right.total),
        count=float(left.count + right.count),
    )


@dataclass(frozen=True)
class PatchRegionAverageMetricState:
    total: float = 0.0
    count: float = 0.0


@dataclass(frozen=True)
class PatchRegionAverageMetric:
    uses_shared_batch = True

    key: str
    value_fn: Any
    patch_loss: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(self.key))
        if not callable(self.value_fn):
            raise TypeError("value_fn must be callable and accept patch loss batch inputs")

    def build(self, *, data, runtime=None, stitch=None, logger=None, patch_loss=None) -> PatchRegionAverageMetric:
        del data, runtime, stitch, logger
        return replace(self, patch_loss=self.patch_loss if self.patch_loss is not None else patch_loss)

    def empty_state(self, *, n_groups: int | None = None) -> PatchRegionAverageMetricState:
        del n_groups
        return PatchRegionAverageMetricState()

    def update(
        self,
        state: PatchRegionAverageMetricState,
        logits,
        targets,
        *,
        valid_mask=None,
        group_idx=None,
        segment_ids=(),
        shared=None,
    ) -> PatchRegionAverageMetricState:
        del group_idx, segment_ids
        if not isinstance(state, PatchRegionAverageMetricState):
            raise TypeError("patch region average metric state must be PatchRegionAverageMetricState")
        values = _resolve_metric_values(
            self.patch_loss,
            self.value_fn,
            logits,
            targets,
            valid_mask=valid_mask,
            shared=shared,
        )
        return PatchRegionAverageMetricState(
            total=float(state.total + sum(values)),
            count=float(state.count + len(values)),
        )

    def finalize(self, state: PatchRegionAverageMetricState) -> MetricReport:
        if not isinstance(state, PatchRegionAverageMetricState):
            raise TypeError("patch region average metric state must be PatchRegionAverageMetricState")
        return MetricReport(summary={str(self.key): _safe_average(float(state.total), float(state.count))})


@dataclass(frozen=True)
class PatchRegionGroupMetricState:
    by_group: dict[int, _RunningTotal] = field(default_factory=dict)
    n_groups: int | None = None


@dataclass(frozen=True)
class PatchRegionWorstGroupMetric:
    requires_group_idx = True

    uses_shared_batch = True

    key: str
    value_fn: Any
    worst_fn: Any = max
    patch_loss: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(self.key))
        if not callable(self.value_fn):
            raise TypeError("value_fn must be callable and accept patch loss batch inputs")
        if not callable(self.worst_fn):
            raise TypeError("worst_fn must be callable")

    def build(self, *, data, runtime=None, stitch=None, logger=None, patch_loss=None) -> PatchRegionWorstGroupMetric:
        del data, runtime, stitch, logger
        return replace(self, patch_loss=self.patch_loss if self.patch_loss is not None else patch_loss)

    def empty_state(self, *, n_groups: int | None = None) -> PatchRegionGroupMetricState:
        return PatchRegionGroupMetricState(
            n_groups=None if n_groups is None else int(n_groups),
        )

    def update(
        self,
        state: PatchRegionGroupMetricState,
        logits,
        targets,
        *,
        valid_mask=None,
        group_idx=None,
        segment_ids=(),
        shared=None,
    ) -> PatchRegionGroupMetricState:
        del segment_ids
        if not isinstance(state, PatchRegionGroupMetricState):
            raise TypeError("patch region group metric state must be PatchRegionGroupMetricState")

        values = _resolve_metric_values(
            self.patch_loss,
            self.value_fn,
            logits,
            targets,
            valid_mask=valid_mask,
            shared=shared,
        )
        groups = _resolve_group_values(group_idx, batch_size=len(values))
        if groups is None:
            return state

        by_group = dict(state.by_group)
        for idx, group_i in enumerate(groups):
            if group_i < 0:
                continue
            by_group[group_i] = _add_running_total(
                by_group.get(group_i, _RunningTotal()),
                _RunningTotal(total=float(values[idx]), count=1.0),
            )
        return PatchRegionGroupMetricState(by_group=by_group, n_groups=state.n_groups)

    def finalize(self, state: PatchRegionGroupMetricState) -> MetricReport:
        if not isinstance(state, PatchRegionGroupMetricState):
            raise TypeError("patch region group metric state must be PatchRegionGroupMetricState")
        values = []
        for group_i in _group_ids(by_group=state.by_group, n_groups=state.n_groups):
            totals = state.by_group.get(int(group_i), _RunningTotal())
            if float(totals.count) <= 0.0:
                continue
            values.append(_safe_average(float(totals.total), float(totals.count)))
        metric_value = 0.0 if not values else float(self.worst_fn(values))
        return MetricReport(summary={str(self.key): metric_value})


@dataclass(frozen=True)
class PatchRegionByGroupMetric:
    requires_group_idx = True

    uses_shared_batch = True

    key: str
    value_fn: Any
    patch_loss: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(self.key))
        if not callable(self.value_fn):
            raise TypeError("value_fn must be callable and accept patch loss batch inputs")

    def build(self, *, data, runtime=None, stitch=None, logger=None, patch_loss=None) -> PatchRegionByGroupMetric:
        del data, runtime, stitch, logger
        return replace(self, patch_loss=self.patch_loss if self.patch_loss is not None else patch_loss)

    def empty_state(self, *, n_groups: int | None = None) -> PatchRegionGroupMetricState:
        return PatchRegionGroupMetricState(
            n_groups=None if n_groups is None else int(n_groups),
        )

    def update(
        self,
        state: PatchRegionGroupMetricState,
        logits,
        targets,
        *,
        valid_mask=None,
        group_idx=None,
        segment_ids=(),
        shared=None,
    ) -> PatchRegionGroupMetricState:
        del segment_ids
        if not isinstance(state, PatchRegionGroupMetricState):
            raise TypeError("patch region group metric state must be PatchRegionGroupMetricState")

        values = _resolve_metric_values(
            self.patch_loss,
            self.value_fn,
            logits,
            targets,
            valid_mask=valid_mask,
            shared=shared,
        )
        groups = _resolve_group_values(group_idx, batch_size=len(values))
        if groups is None:
            return state

        by_group = dict(state.by_group)
        for idx, group_i in enumerate(groups):
            if group_i < 0:
                continue
            by_group[group_i] = _add_running_total(
                by_group.get(group_i, _RunningTotal()),
                _RunningTotal(total=float(values[idx]), count=1.0),
            )
        return PatchRegionGroupMetricState(by_group=by_group, n_groups=state.n_groups)

    def finalize(self, state: PatchRegionGroupMetricState) -> MetricReport:
        if not isinstance(state, PatchRegionGroupMetricState):
            raise TypeError("patch region group metric state must be PatchRegionGroupMetricState")

        by_group: dict[str, dict[str, float]] = {}
        for group_i in _group_ids(by_group=state.by_group, n_groups=state.n_groups):
            totals = state.by_group.get(int(group_i), _RunningTotal())
            by_group[str(group_i)] = {
                str(self.key): _safe_average(float(totals.total), float(totals.count)),
            }
        return MetricReport(by_group=by_group)


@dataclass(frozen=True)
class PatchRegionGroupCountMetricState:
    by_group: dict[int, float] = field(default_factory=dict)
    n_groups: int | None = None


@dataclass(frozen=True)
class PatchRegionByGroupCountMetric:
    requires_group_idx = True

    key: str = "val/count"

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(self.key))

    def build(self, *, data, runtime=None, stitch=None, logger=None, patch_loss=None) -> PatchRegionByGroupCountMetric:
        del data, runtime, stitch, logger, patch_loss
        return self

    def empty_state(self, *, n_groups: int | None = None) -> PatchRegionGroupCountMetricState:
        return PatchRegionGroupCountMetricState(
            n_groups=None if n_groups is None else int(n_groups),
        )

    def update(
        self,
        state: PatchRegionGroupCountMetricState,
        logits,
        targets,
        *,
        valid_mask=None,
        group_idx=None,
        segment_ids=(),
    ) -> PatchRegionGroupCountMetricState:
        del logits, targets, valid_mask, segment_ids
        if not isinstance(state, PatchRegionGroupCountMetricState):
            raise TypeError("patch region group-count metric state must be PatchRegionGroupCountMetricState")
        if group_idx is None:
            return state

        batch_size = _resolve_batch_size(group_idx)
        groups = _resolve_group_values(group_idx, batch_size=int(batch_size or group_idx.numel()))
        if groups is None:
            return state

        by_group = dict(state.by_group)
        for group_i in groups:
            if group_i < 0:
                continue
            by_group[group_i] = float(by_group.get(group_i, 0.0) + 1.0)
        return PatchRegionGroupCountMetricState(by_group=by_group, n_groups=state.n_groups)

    def finalize(self, state: PatchRegionGroupCountMetricState) -> MetricReport:
        if not isinstance(state, PatchRegionGroupCountMetricState):
            raise TypeError("patch region group-count metric state must be PatchRegionGroupCountMetricState")

        by_group: dict[str, dict[str, float]] = {}
        for group_i in _group_ids(by_group=state.by_group, n_groups=state.n_groups):
            by_group[str(group_i)] = {
                str(self.key): float(state.by_group.get(int(group_i), 0.0)),
            }
        return MetricReport(by_group=by_group)


@dataclass(frozen=True)
class PatchRegionSegmentMetricState:
    by_segment: dict[str, _RunningTotal] = field(default_factory=dict)


@dataclass(frozen=True)
class PatchRegionBySegmentMetric:
    uses_shared_batch = True

    key: str
    value_fn: Any
    patch_loss: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(self.key))
        if not callable(self.value_fn):
            raise TypeError("value_fn must be callable and accept patch loss batch inputs")

    def build(self, *, data, runtime=None, stitch=None, logger=None, patch_loss=None) -> PatchRegionBySegmentMetric:
        del data, runtime, stitch, logger
        return replace(self, patch_loss=self.patch_loss if self.patch_loss is not None else patch_loss)

    def empty_state(self, *, n_groups: int | None = None) -> PatchRegionSegmentMetricState:
        del n_groups
        return PatchRegionSegmentMetricState()

    def update(
        self,
        state: PatchRegionSegmentMetricState,
        logits,
        targets,
        *,
        valid_mask=None,
        group_idx=None,
        segment_ids=(),
        shared=None,
    ) -> PatchRegionSegmentMetricState:
        del group_idx
        if not isinstance(state, PatchRegionSegmentMetricState):
            raise TypeError("patch region segment metric state must be PatchRegionSegmentMetricState")

        values = _resolve_metric_values(
            self.patch_loss,
            self.value_fn,
            logits,
            targets,
            valid_mask=valid_mask,
            shared=shared,
        )
        segments = list(segment_ids)
        if len(segments) != len(values):
            return state

        by_segment = dict(state.by_segment)
        for idx, segment_id in enumerate(segments):
            segment_key = str(segment_id)
            by_segment[segment_key] = _add_running_total(
                by_segment.get(segment_key, _RunningTotal()),
                _RunningTotal(total=float(values[idx]), count=1.0),
            )
        return PatchRegionSegmentMetricState(by_segment=by_segment)

    def finalize(self, state: PatchRegionSegmentMetricState) -> MetricReport:
        if not isinstance(state, PatchRegionSegmentMetricState):
            raise TypeError("patch region segment metric state must be PatchRegionSegmentMetricState")

        by_segment = {
            segment_id: {
                str(self.key): _safe_average(float(totals.total), float(totals.count)),
            }
            for segment_id, totals in state.by_segment.items()
        }
        return MetricReport(by_segment=by_segment)


@dataclass(frozen=True)
class PatchRegionSegmentCountMetricState:
    by_segment: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PatchRegionBySegmentCountMetric:
    key: str = "val/count"

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", str(self.key))

    def build(self, *, data, runtime=None, stitch=None, logger=None, patch_loss=None) -> PatchRegionBySegmentCountMetric:
        del data, runtime, stitch, logger, patch_loss
        return self

    def empty_state(self, *, n_groups: int | None = None) -> PatchRegionSegmentCountMetricState:
        del n_groups
        return PatchRegionSegmentCountMetricState()

    def update(
        self,
        state: PatchRegionSegmentCountMetricState,
        logits,
        targets,
        *,
        valid_mask=None,
        group_idx=None,
        segment_ids=(),
    ) -> PatchRegionSegmentCountMetricState:
        del logits, valid_mask, group_idx
        if not isinstance(state, PatchRegionSegmentCountMetricState):
            raise TypeError("patch region segment-count metric state must be PatchRegionSegmentCountMetricState")

        batch_size = _resolve_batch_size(targets)
        segments = list(segment_ids)
        if batch_size is not None and len(segments) != int(batch_size):
            return state

        by_segment = dict(state.by_segment)
        for segment_id in segments:
            segment_key = str(segment_id)
            by_segment[segment_key] = float(by_segment.get(segment_key, 0.0) + 1.0)
        return PatchRegionSegmentCountMetricState(by_segment=by_segment)

    def finalize(self, state: PatchRegionSegmentCountMetricState) -> MetricReport:
        if not isinstance(state, PatchRegionSegmentCountMetricState):
            raise TypeError("patch region segment-count metric state must be PatchRegionSegmentCountMetricState")
        by_segment = {
            segment_id: {str(self.key): float(count)}
            for segment_id, count in state.by_segment.items()
        }
        return MetricReport(by_segment=by_segment)


@dataclass(frozen=True)
class PatchConfusionMetricState:
    counts: ConfusionCounts = field(default_factory=zero_confusion_counts)


@dataclass(frozen=True)
class PatchConfusionMetric:
    key: str = "metrics/val/dice"
    threshold: float = 0.5
    score_fn: Any = dice_from_counts

    def __post_init__(self) -> None:
        threshold = float(self.threshold)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
        if not callable(self.score_fn):
            raise TypeError("score_fn must be callable and accept ConfusionCounts")
        object.__setattr__(self, "key", str(self.key))
        object.__setattr__(self, "threshold", threshold)

    def build(self, *, data, runtime=None, stitch=None, logger=None, patch_loss=None) -> PatchConfusionMetric:
        del data, runtime, stitch, logger, patch_loss
        return self

    def empty_state(self, *, n_groups: int | None = None) -> PatchConfusionMetricState:
        del n_groups
        return PatchConfusionMetricState()

    def update(
        self,
        state: PatchConfusionMetricState,
        logits,
        targets,
        *,
        valid_mask=None,
        group_idx=None,
        segment_ids=(),
    ) -> PatchConfusionMetricState:
        del group_idx, segment_ids
        if not isinstance(state, PatchConfusionMetricState):
            raise TypeError("patch confusion metric state must be PatchConfusionMetricState")
        mask = None if valid_mask is None else (valid_mask >= 0.5)
        counts = _masked_confusion_counts(
            logits,
            targets,
            threshold=float(self.threshold),
            valid_mask=mask,
        )
        return PatchConfusionMetricState(
            counts=add_confusion_counts(state.counts, counts),
        )

    def finalize(self, state: PatchConfusionMetricState) -> MetricReport:
        if not isinstance(state, PatchConfusionMetricState):
            raise TypeError("patch confusion metric state must be PatchConfusionMetricState")
        raw_value = self.score_fn(state.counts)
        if hasattr(raw_value, "item"):
            value = float(raw_value.item())
        else:
            value = float(raw_value)
        return MetricReport(
            summary={
                str(self.key): value,
            }
        )


class StreamingBinarySegmentationMetrics:
    def __init__(self, *, threshold: float = 0.5) -> None:
        threshold = float(threshold)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._counts = zero_confusion_counts()

    def update(self, *, logits, targets, mask=None) -> None:
        self._counts = add_confusion_counts(
            self._counts,
            _masked_confusion_counts(
                logits,
                targets,
                threshold=float(self.threshold),
                valid_mask=mask,
            ),
        )

    def compute(self) -> dict[str, float]:
        return {
            "dice": float(dice_from_counts(self._counts).item()),
            "balanced_accuracy": float(balanced_accuracy_from_counts(self._counts).item()),
        }
