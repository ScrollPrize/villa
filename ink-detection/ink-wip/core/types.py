from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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


@dataclass
class DataBundle:
    train_loader: Any
    val_loader: Any
    in_channels: int
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    summary: dict[str, float]
    by_group: dict[str, dict[str, float]] = field(default_factory=dict)
    by_segment: dict[str, dict[str, float]] = field(default_factory=dict)
