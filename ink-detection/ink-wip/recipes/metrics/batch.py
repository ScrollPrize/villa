from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass(frozen=True)
class MetricBatch:
    logits: torch.Tensor
    targets: torch.Tensor
    valid_mask: torch.Tensor | None = None
    group_idx: torch.Tensor | None = None
    segment_ids: Sequence[str] = field(default_factory=tuple)
    patch_xyxy: torch.Tensor | None = None

