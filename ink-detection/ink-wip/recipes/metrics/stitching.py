from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from ink.core.types import ModelOutputBatch

if TYPE_CHECKING:
    from ink.recipes.eval.stitch_prepared import PreparedStitchEvalArtifacts


@dataclass(frozen=True)
class StitchMetricBatch:
    logits: torch.Tensor
    targets: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
    connectivity: int = 2
    prepared: PreparedStitchEvalArtifacts | None = None

    @classmethod
    def from_model_output_batch(
        cls,
        batch: ModelOutputBatch,
        *,
        connectivity: int = 2,
        prepared: PreparedStitchEvalArtifacts | None = None,
    ) -> StitchMetricBatch:
        if not isinstance(batch, ModelOutputBatch):
            raise TypeError("stitched metric batches require ModelOutputBatch")
        return cls(
            logits=batch.logits,
            targets=batch.targets,
            valid_mask=batch.valid_mask,
            connectivity=int(connectivity),
            prepared=prepared,
        )

    def require_targets(self) -> torch.Tensor:
        if self.targets is None:
            raise ValueError("stitched metric batch requires targets")
        return self.targets


def _as_bool_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={tuple(array.shape)}")
    return array.astype(bool, copy=False)


def _tensor_plane(tensor: torch.Tensor | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().to(device="cpu").numpy()
    else:
        array = np.asarray(tensor)
    if array.ndim == 4:
        if tuple(array.shape[:2]) != (1, 1):
            raise ValueError(f"{name} must have shape (1,1,H,W) or (H,W), got {tuple(array.shape)}")
        return np.asarray(array[0, 0])
    if array.ndim == 2:
        return np.asarray(array)
    raise ValueError(f"{name} must have shape (1,1,H,W) or (H,W), got {tuple(array.shape)}")


def _sigmoid_numpy(array: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(array, dtype=np.float32), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


@dataclass(frozen=True)
class StitchComponentArrays:
    pred_prob: np.ndarray
    pred_bin: np.ndarray
    gt_bin: np.ndarray
    valid_mask: np.ndarray
    connectivity: int


def stitch_component_arrays(batch: StitchMetricBatch, *, threshold: float) -> StitchComponentArrays:
    if not isinstance(batch, StitchMetricBatch):
        raise TypeError("stitched component metrics require StitchMetricBatch")

    logits = _tensor_plane(batch.logits, name="logits")
    targets = _tensor_plane(batch.require_targets(), name="targets")
    if logits.shape != targets.shape:
        raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    if batch.valid_mask is None:
        valid_mask = np.ones(logits.shape, dtype=bool)
    else:
        valid_mask = _tensor_plane(batch.valid_mask, name="valid_mask").astype(bool, copy=False)
        if valid_mask.shape != logits.shape:
            raise ValueError(f"valid_mask/logits shape mismatch: {tuple(valid_mask.shape)} vs {tuple(logits.shape)}")

    pred_prob = _sigmoid_numpy(logits)
    pred_prob = pred_prob.copy()
    pred_prob[~valid_mask] = 0.0

    binary_targets = np.logical_or(np.isclose(targets, 0.0), np.isclose(targets, 1.0))
    if not bool(binary_targets.all()):
        raise ValueError("stitched component metrics require binary targets encoded as 0/1")
    gt_bin = np.asarray(targets, dtype=bool)
    gt_bin = gt_bin.copy()
    gt_bin[~valid_mask] = False

    pred_bin = pred_prob >= float(threshold)
    pred_bin = np.asarray(pred_bin, dtype=bool)
    pred_bin[~valid_mask] = False

    return StitchComponentArrays(
        pred_prob=pred_prob,
        pred_bin=pred_bin,
        gt_bin=gt_bin,
        valid_mask=valid_mask,
        connectivity=int(batch.connectivity),
    )
