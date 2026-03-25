from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, kw_only=True)
class BaseMetric(ABC):
    name: str | None = None
    per_sample: bool = False

    def __post_init__(self) -> None:
        explicit_name = None if self.name is None else str(self.name).strip()
        object.__setattr__(self, "name", explicit_name or self.default_name())
        object.__setattr__(self, "per_sample", bool(self.per_sample))

    @abstractmethod
    def default_name(self) -> str:
        raise NotImplementedError

    def metric_name(self) -> str:
        return str(self.name)

    def compute(self, batch):
        if self.per_sample:
            return self.compute_per_sample(batch)
        return self.compute_batch(batch)

    @abstractmethod
    def compute_batch(self, batch):
        raise NotImplementedError

    def compute_per_sample(self, batch):
        raise NotImplementedError(f"{type(self).__name__} does not implement per-sample computation")


@dataclass(frozen=True, kw_only=True)
class BinaryImageMetric(BaseMetric):
    threshold: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "threshold", float(self.threshold))
        BaseMetric.__post_init__(self)

    @staticmethod
    def _require_targets(batch):
        if hasattr(batch, "require_targets"):
            return batch.require_targets()
        targets = getattr(batch, "targets", None)
        if targets is None:
            raise ValueError(f"{type(batch).__name__} does not provide targets")
        return targets

    @staticmethod
    def _tensor_plane(tensor, *, name: str) -> np.ndarray:
        if hasattr(tensor, "detach"):
            array = tensor.detach().to(device="cpu").numpy()
        else:
            array = np.asarray(tensor)
        if array.ndim == 4:
            if tuple(array.shape[:2]) != (1, 1):
                raise ValueError(f"{name} must have shape (1,1,H,W), (1,H,W), or (H,W), got {tuple(array.shape)}")
            return np.asarray(array[0, 0])
        if array.ndim == 3:
            if int(array.shape[0]) != 1:
                raise ValueError(f"{name} must have shape (1,1,H,W), (1,H,W), or (H,W), got {tuple(array.shape)}")
            return np.asarray(array[0])
        if array.ndim == 2:
            return np.asarray(array)
        raise ValueError(f"{name} must have shape (1,1,H,W), (1,H,W), or (H,W), got {tuple(array.shape)}")

    @staticmethod
    def _sigmoid_numpy(array: np.ndarray) -> np.ndarray:
        clipped = np.clip(np.asarray(array, dtype=np.float32), -80.0, 80.0)
        return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)

    @staticmethod
    def _slice_sample(value, index: int):
        if value is None:
            return None
        return value[index : index + 1]

    def _iter_sample_batches(self, batch):
        logits = batch.logits
        if hasattr(logits, "ndim"):
            ndim = int(logits.ndim)
            n_samples = int(logits.shape[0]) if ndim >= 1 else 0
        else:
            array = np.asarray(logits)
            ndim = int(array.ndim)
            n_samples = int(array.shape[0]) if ndim >= 1 else 0
        if ndim in (2, 3):
            yield batch
            return
        if ndim != 4:
            raise ValueError(f"{type(self).__name__} expects logits with shape (H,W), (1,H,W), or (N,1,H,W)")
        for index in range(n_samples):
            yield replace(
                batch,
                logits=self._slice_sample(batch.logits, index),
                targets=self._slice_sample(getattr(batch, "targets", None), index),
                valid_mask=self._slice_sample(getattr(batch, "valid_mask", None), index),
            )

    def _prepare_binary_arrays(self, batch) -> tuple[np.ndarray, np.ndarray]:
        logits = self._tensor_plane(batch.logits, name="logits")
        targets = self._tensor_plane(self._require_targets(batch), name="targets")
        if logits.shape != targets.shape:
            raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")

        valid_mask_tensor = getattr(batch, "valid_mask", None)
        if valid_mask_tensor is None:
            valid_mask = np.ones(logits.shape, dtype=bool)
        else:
            valid_mask = self._tensor_plane(valid_mask_tensor, name="valid_mask").astype(bool, copy=False)
            if valid_mask.shape != logits.shape:
                raise ValueError(f"valid_mask/logits shape mismatch: {tuple(valid_mask.shape)} vs {tuple(logits.shape)}")

        pred_prob = self._sigmoid_numpy(logits)
        pred_prob = pred_prob.copy()
        pred_prob[~valid_mask] = 0.0

        binary_targets = np.logical_or(np.isclose(targets, 0.0), np.isclose(targets, 1.0))
        if not bool(binary_targets.all()):
            raise ValueError(f"{type(self).__name__} requires binary targets encoded as 0/1")
        gt_bin = np.asarray(targets, dtype=bool)
        gt_bin = gt_bin.copy()
        gt_bin[~valid_mask] = False

        pred_bin = pred_prob >= float(self.threshold)
        pred_bin = np.asarray(pred_bin, dtype=bool)
        pred_bin[~valid_mask] = False
        return pred_bin, gt_bin

    @abstractmethod
    def compute_binary(self, pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
        raise NotImplementedError

    def compute_batch(self, batch) -> float:
        pred_bin, gt_bin = self._prepare_binary_arrays(batch)
        return float(self.compute_binary(pred_bin, gt_bin))

    def compute_per_sample(self, batch) -> float:
        values = [self.compute_batch(sample_batch) for sample_batch in self._iter_sample_batches(batch)]
        if not values:
            return 0.0
        return sum(values) / float(len(values))
