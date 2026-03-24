from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve

from ink.recipes.metrics.confusion import _resolve_metric_name
from ink.recipes.metrics.reports import MetricReport
from ink.recipes.metrics.stitching import (
    StitchMetricBatch,
    stitch_component_arrays,
)

_EPS = 1e-8


def _as_bool_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={tuple(array.shape)}")
    return array.astype(bool, copy=False)


def drd(pred: np.ndarray, gt: np.ndarray, *, block_size: int = 8) -> float:
    pred_u8 = _as_bool_2d(pred).astype(np.uint8)
    gt_u8 = _as_bool_2d(gt).astype(np.uint8)
    if pred_u8.shape != gt_u8.shape:
        raise ValueError(f"pred/gt shape mismatch: {tuple(pred_u8.shape)} vs {tuple(gt_u8.shape)}")

    height, width = gt_u8.shape
    block = int(block_size)
    if block < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size!r}")

    blocks_y = (height + block - 1) // block
    blocks_x = (width + block - 1) // block
    pad_h = blocks_y * block - height
    pad_w = blocks_x * block - width
    gt_pad = np.pad(gt_u8, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    gt_blocks = gt_pad.reshape(blocks_y, block, blocks_x, block)
    block_sum = gt_blocks.sum(axis=(1, 3))
    nubn = int(np.logical_and(block_sum > 0, block_sum < (block * block)).sum())
    nubn = max(nubn, 1)

    mismatched = pred_u8 != gt_u8
    if not bool(mismatched.any()):
        return 0.0

    weights = np.zeros((5, 5), dtype=np.float64)
    for iy, dy in enumerate(range(-2, 3)):
        for ix, dx in enumerate(range(-2, 3)):
            if dy == 0 and dx == 0:
                continue
            weights[iy, ix] = 1.0 / math.sqrt(float(dy * dy + dx * dx))
    weights /= float(weights.sum() + _EPS)

    gt_weight_sum = convolve(gt_u8.astype(np.float64), weights, mode="constant", cval=0.0)
    drd_map = np.where(pred_u8.astype(bool), 1.0 - gt_weight_sum, gt_weight_sum)
    return float(drd_map[mismatched].sum() / float(nubn))


@dataclass(frozen=True)
class _DRDState:
    total: float = 0.0
    count: int = 0


@dataclass(frozen=True, kw_only=True)
class DRD:
    threshold: float = 0.5
    block_size: int = 8
    name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "block_size", int(self.block_size))
        explicit_name = None if self.name is None else str(self.name).strip()
        object.__setattr__(self, "name", None if explicit_name is None or not explicit_name else explicit_name)

    def metric_name(self) -> str:
        base_name = _resolve_metric_name(
            explicit_name=self.name,
            base_name="drd",
            threshold=float(self.threshold),
        )
        if int(self.block_size) == 8 or self.name is not None:
            return base_name
        return f"{base_name}_bs_{int(self.block_size)}"

    def empty_state(self, *, n_groups: int | None = None) -> _DRDState:
        del n_groups
        return _DRDState()

    def update(self, state: _DRDState, batch, *, shared=None) -> _DRDState:
        del shared
        if not isinstance(batch, StitchMetricBatch):
            raise TypeError("DRD requires StitchMetricBatch from StitchEval")
        arrays = stitch_component_arrays(batch, threshold=float(self.threshold))
        value = drd(
            arrays.pred_bin,
            arrays.gt_bin,
            block_size=int(self.block_size),
        )
        return _DRDState(
            total=float(state.total) + float(value),
            count=int(state.count) + 1,
        )

    def finalize(self, state: _DRDState) -> MetricReport:
        value = 0.0 if int(state.count) <= 0 else float(state.total) / float(state.count)
        return MetricReport(summary={str(self.metric_name()): value})
