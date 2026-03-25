from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve

from koine_machines.evaluation.metrics.base import BinaryImageMetric
from koine_machines.evaluation.metrics.confusion import _resolve_metric_name

@dataclass(frozen=True, kw_only=True)
class DRD(BinaryImageMetric):
    _EPS = 1e-8

    block_size: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_size", int(self.block_size))
        BinaryImageMetric.__post_init__(self)

    def default_name(self) -> str:
        base_name = _resolve_metric_name(
            explicit_name=None,
            base_name="DRD",
            threshold=float(self.threshold),
        )
        if int(self.block_size) == 8:
            return base_name
        return f"{base_name}_bs_{int(self.block_size)}"

    @staticmethod
    def _as_bool_2d(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim != 2:
            raise ValueError(f"expected 2D array, got shape={tuple(array.shape)}")
        return array.astype(bool, copy=False)

    @classmethod
    def _weights(cls) -> np.ndarray:
        weights = np.zeros((5, 5), dtype=np.float64)
        for iy, dy in enumerate(range(-2, 3)):
            for ix, dx in enumerate(range(-2, 3)):
                if dy == 0 and dx == 0:
                    continue
                weights[iy, ix] = 1.0 / math.sqrt(float(dy * dy + dx * dx))
        weights /= float(weights.sum() + cls._EPS)
        return weights

    @classmethod
    def _compute_drd(cls, pred: np.ndarray, gt: np.ndarray, *, block_size: int) -> float:
        pred_u8 = cls._as_bool_2d(pred).astype(np.uint8)
        gt_u8 = cls._as_bool_2d(gt).astype(np.uint8)
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

        gt_weight_sum = convolve(gt_u8.astype(np.float64), cls._weights(), mode="constant", cval=0.0)
        drd_map = np.where(pred_u8.astype(bool), 1.0 - gt_weight_sum, gt_weight_sum)
        return float(drd_map[mismatched].sum() / float(nubn))

    def compute_binary(self, pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
        return self._compute_drd(
            pred_bin,
            gt_bin,
            block_size=int(self.block_size),
        )
