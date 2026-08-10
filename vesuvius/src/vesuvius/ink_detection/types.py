"""Shared immutable vocabulary for ink data and validation numerics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vesuvius.ink_detection.config import DatasetSource, InkDataConfig


if TYPE_CHECKING:
    import torch


BBoxZYX = tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class Segment:
    """One surface representation and its resolved label assets."""

    data_config: InkDataConfig
    source: DatasetSource
    dataset_idx: int
    segment_relpath: str
    segment_dir: Path
    segment_name: str
    image_volume: str | Path
    inklabels: Path | None = None
    supervision_mask: Path | None = None
    validation_mask: Path | None = None

    @property
    def scale(self) -> int:
        return self.source.volume_scale

    @property
    def patch_size(self) -> tuple[int, int, int]:
        return self.data_config.patch_size

    @property
    def cache_key(self) -> tuple[int, str, int, str, str, str]:
        return (
            self.dataset_idx,
            self.segment_relpath,
            self.scale,
            "" if self.inklabels is None else str(self.inklabels),
            "" if self.supervision_mask is None else str(self.supervision_mask),
            "" if self.validation_mask is None else str(self.validation_mask),
        )

    def with_labels(
        self,
        *,
        inklabels: Path | None,
        supervision_mask: Path | None,
        validation_mask: Path | None,
    ) -> "Segment":
        return replace(
            self,
            inklabels=inklabels,
            supervision_mask=supervision_mask,
            validation_mask=validation_mask,
        )


@dataclass(frozen=True)
class Patch:
    """One fixed ZYX crop and its training/validation role."""

    segment: Segment
    bbox: BBoxZYX
    is_validation: bool = False
    is_unlabeled: bool = False
    supervision_mask_override: Path | str | None = None

    @property
    def image_volume(self) -> str | Path:
        return self.segment.image_volume

    @property
    def supervision_mask(self) -> Path | str | None:
        return self.supervision_mask_override or self.segment.supervision_mask

    @property
    def inklabels(self) -> Path | None:
        return self.segment.inklabels

    @property
    def segment_dir(self) -> Path:
        return self.segment.segment_dir


@dataclass(frozen=True)
class MetricBatch:
    """Logits and optional binary targets/mask with matching tensor shapes."""

    logits: torch.Tensor
    targets: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None

    def require_targets(self) -> torch.Tensor:
        if self.targets is None:
            raise ValueError("MetricBatch does not provide targets")
        return self.targets


@dataclass(frozen=True)
class ConfusionCounts:
    tp: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor
    tn: torch.Tensor


@dataclass(frozen=True)
class SamplingPolicy:
    """The DataLoader arguments selected by one sampling strategy."""

    strategy: str
    shuffle: bool = False
    generator: torch.Generator | None = None
    sampler: Any = None
    batch_sampler: Any = None
    audit: dict[str, Any] | None = None
