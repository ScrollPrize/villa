from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    try:
        from .segment import Segment
    except ImportError:
        from segment import Segment


@dataclass
class Patch:
    segment: "Segment"
    bbox: tuple  # (z0, y0, x0, z1, y1, x1)
    is_validation: bool = False
    supervision_mask_override: Any = None

    @property
    def image_volume(self):
        return self.segment.image_volume

    @property
    def supervision_mask(self):
        if self.supervision_mask_override is not None:
            return self.supervision_mask_override
        return self.segment.supervision_mask

    @property
    def inklabels(self):
        return self.segment.inklabels

    @property
    def segment_dir(self):
        return self.segment.segment_dir
