from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from koine_machines.evaluation.metrics.base import BinaryImageMetric
from koine_machines.evaluation.metrics.confusion import _resolve_metric_name


@dataclass(frozen=True, kw_only=True)
class PFMWeighted(BinaryImageMetric):
    _EPS = 1e-8

    skeleton_method: str = "guo_hall"

    def __post_init__(self) -> None:
        object.__setattr__(self, "skeleton_method", self._normalize_skeleton_method(self.skeleton_method))
        BinaryImageMetric.__post_init__(self)

    def default_name(self) -> str:
        return _resolve_metric_name(
            explicit_name=None,
            base_name="PFMWeighted",
            threshold=float(self.threshold),
        )

    @staticmethod
    def _normalize_skeleton_method(method: str) -> str:
        normalized = str(method).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "guohall": "guo_hall",
            "guo_hall": "guo_hall",
            "zhangsuen": "zhang_suen",
            "zhang_suen": "zhang_suen",
        }
        if normalized not in aliases:
            raise ValueError(f"unsupported skeleton_method {method!r}")
        return aliases[normalized]

    @staticmethod
    def _as_bool_2d(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim != 2:
            raise ValueError(f"expected 2D array, got shape={tuple(array.shape)}")
        return array.astype(bool, copy=False)

    def _safe_div(self, numerator: float, denominator: float) -> float:
        return float(numerator) / float(denominator + self._EPS)

    def _skeletonize_binary(self, mask: np.ndarray) -> np.ndarray:
        mask_u8 = self._as_bool_2d(mask).astype(np.uint8) * 255
        try:
            import cv2
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PFMWeighted requires opencv-contrib-python-headless for skeletonization") from exc
        if not hasattr(cv2, "ximgproc"):
            raise RuntimeError("PFMWeighted requires cv2.ximgproc.thinning")
        thinning_type = (
            cv2.ximgproc.THINNING_GUOHALL
            if str(self.skeleton_method) == "guo_hall"
            else cv2.ximgproc.THINNING_ZHANGSUEN
        )
        return cv2.ximgproc.thinning(mask_u8, thinningType=thinning_type).astype(bool)

    def _pseudo_weight_maps(self, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gt_bool = self._as_bool_2d(gt)
        skeleton = self._skeletonize_binary(gt_bool)
        recall_weights = skeleton.astype(np.float32, copy=False)
        if not bool(recall_weights.any()):
            recall_weights = gt_bool.astype(np.float32, copy=False)
        distance = distance_transform_edt(~gt_bool)
        precision_weights = (1.0 / (1.0 + distance)).astype(np.float32)
        precision_weights[gt_bool] = 1.0
        return recall_weights, precision_weights

    def _weighted_pseudo_fmeasure_from_weights(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        *,
        recall_weights: np.ndarray,
        recall_weights_sum: float,
        precision_weights: np.ndarray,
    ) -> float:
        pred = self._as_bool_2d(pred)
        gt = self._as_bool_2d(gt)
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
        return self._safe_div(2.0 * recall * precision, recall + precision)

    def compute_binary(self, pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
        recall_weights, precision_weights = self._pseudo_weight_maps(gt_bin)
        return self._weighted_pseudo_fmeasure_from_weights(
            pred_bin,
            gt_bin,
            recall_weights=recall_weights.astype(np.float32, copy=False),
            recall_weights_sum=float(recall_weights.sum(dtype=np.float64)),
            precision_weights=precision_weights.astype(np.float32, copy=False),
        )
