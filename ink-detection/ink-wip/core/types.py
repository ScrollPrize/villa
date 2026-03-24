from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import torch


@dataclass
class BatchMeta:
    segment_ids: list[str]
    valid_mask: torch.Tensor | None = None
    patch_xyxy: torch.Tensor | None = None
    group_idx: torch.Tensor | None = None


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor | None
    meta: BatchMeta


@dataclass(frozen=True)
class ModelOutputBatch:
    logits: torch.Tensor
    segment_ids: Sequence[str] = field(default_factory=tuple)
    patch_xyxy: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None
    group_idx: torch.Tensor | None = None

    @classmethod
    def from_batch_and_logits(cls, batch: Batch, logits: torch.Tensor) -> "ModelOutputBatch":
        if not isinstance(batch, Batch):
            raise TypeError("model output batch requires Batch")
        return cls(
            logits=logits,
            segment_ids=tuple(batch.meta.segment_ids),
            patch_xyxy=batch.meta.patch_xyxy,
            targets=batch.y,
            valid_mask=batch.meta.valid_mask,
            group_idx=batch.meta.group_idx,
        )

    def require_targets(self) -> torch.Tensor:
        if self.targets is None:
            raise ValueError("model output batch requires targets")
        return self.targets

    def require_patch_xyxy(self) -> torch.Tensor:
        if self.patch_xyxy is None:
            raise ValueError("model output batch requires patch_xyxy")
        return self.patch_xyxy


@dataclass
class DataBundle:
    train_loader: Any
    eval_loader: Any
    in_channels: int
    augment: Any = None
    group_counts: list[int] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    summary: dict[str, float]
    by_group: dict[str, dict[str, float]] = field(default_factory=dict)
    by_segment: dict[str, dict[str, float]] = field(default_factory=dict)
    stages: dict[str, "EvalReport"] = field(default_factory=dict)
