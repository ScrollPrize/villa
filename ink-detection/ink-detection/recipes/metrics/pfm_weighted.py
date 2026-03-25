from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ink.recipes.metrics.confusion import _resolve_metric_name
from ink.recipes.metrics.reports import MetricReport
from ink.recipes.metrics.stitching import (
    StitchMetricBatch,
    stitch_component_arrays,
)
from ink.recipes.stitch.artifact_primitives import (
    as_bool_2d as _as_bool_2d,
    normalize_skeleton_method as _normalize_skeleton_method,
    pseudo_weight_maps,
    skeletonize_binary,
)

_EPS = 1e-8


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator + _EPS)


def weighted_pseudo_fmeasure_from_weights(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    recall_weights: np.ndarray,
    recall_weights_sum: float,
    precision_weights: np.ndarray,
) -> float:
    pred = _as_bool_2d(pred)
    gt = _as_bool_2d(gt)
    recall_weights = np.asarray(recall_weights, dtype=np.float64)
    precision_weights = np.asarray(precision_weights, dtype=np.float64)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    if recall_weights.shape != gt.shape:
        raise ValueError(f"recall_weights/gt shape mismatch: {tuple(recall_weights.shape)} vs {tuple(gt.shape)}")
    if precision_weights.shape != gt.shape:
        raise ValueError(
            f"precision_weights/gt shape mismatch: {tuple(precision_weights.shape)} vs {tuple(gt.shape)}"
        )

    gw_sum = float(recall_weights_sum)
    if gw_sum <= 0.0:
        raise ValueError("invalid weighted pseudo-recall map: sum is non-positive")
    recall = float((pred.astype(np.float64) * recall_weights).sum() / gw_sum)

    weighted_pred = pred.astype(np.float64) * precision_weights
    weighted_pred_sum = float(weighted_pred.sum())
    if weighted_pred_sum <= 0.0:
        if int(pred.sum()) == 0:
            return 0.0
        raise ValueError("invalid weighted pseudo-precision denominator: predicted weighted foreground sum is non-positive")
    precision = float((gt.astype(np.float64) * weighted_pred).sum() / weighted_pred_sum)
    return _safe_div(2.0 * recall * precision, recall + precision)


@dataclass(frozen=True)
class _PFMWeightedState:
    total: float = 0.0
    count: int = 0


@dataclass(frozen=True, kw_only=True)
class PFMWeighted:
    threshold: float = 0.5
    skeleton_method: str = "guo_hall"
    name: str | None = None

    def __post_init__(self) -> None:
        threshold = float(self.threshold)
        explicit_name = None if self.name is None else str(self.name).strip()
        resolved_name = _resolve_metric_name(
            explicit_name=explicit_name,
            base_name="PFMWeighted",
            threshold=threshold,
        )
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "skeleton_method", _normalize_skeleton_method(self.skeleton_method))
        object.__setattr__(self, "name", resolved_name)

    def metric_name(self) -> str:
        return str(self.name)

    def empty_state(self, *, n_groups: int | None = None) -> _PFMWeightedState:
        del n_groups
        return _PFMWeightedState()

    def update(self, state: _PFMWeightedState, batch, *, shared=None) -> _PFMWeightedState:
        del shared
        if not isinstance(batch, StitchMetricBatch):
            raise TypeError("PFMWeighted requires StitchMetricBatch from StitchEval")
        arrays = stitch_component_arrays(batch, threshold=float(self.threshold))
        prepared = batch.prepared
        prepared_gt = None if prepared is None else prepared.selected_component_gt()
        if prepared_gt is not None and np.array_equal(prepared_gt, arrays.gt_bin):
            recall_weights, precision_weights = prepared.pfm_weights(method=str(self.skeleton_method))
        else:
            skeleton = skeletonize_binary(
                arrays.gt_bin,
                method=str(self.skeleton_method),
            )
            recall_weights, precision_weights = pseudo_weight_maps(
                arrays.gt_bin,
                connectivity=int(arrays.connectivity),
                skel_gt=skeleton,
            )
        value = weighted_pseudo_fmeasure_from_weights(
            arrays.pred_bin,
            arrays.gt_bin,
            recall_weights=recall_weights.astype("float32", copy=False),
            recall_weights_sum=float(recall_weights.sum(dtype="float64")),
            precision_weights=precision_weights.astype("float32", copy=False),
        )
        return _PFMWeightedState(
            total=float(state.total) + float(value),
            count=int(state.count) + 1,
        )

    def finalize(self, state: _PFMWeightedState) -> MetricReport:
        value = 0.0 if int(state.count) <= 0 else float(state.total) / float(state.count)
        return MetricReport(summary={str(self.metric_name()): value})
