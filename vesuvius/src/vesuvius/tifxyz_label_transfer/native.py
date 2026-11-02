"""Label-transfer reporting over vc_reader's prepared native surface sampler."""
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from vc_reader.resident import SurfaceTransfer

@dataclass(frozen=True)
class NativeTileResult:
    source_indices: NDArray[np.int64]
    validity: NDArray[np.uint8]
    distances: NDArray[np.float32]
    target_surface_valid: int
    measured_pixels: int
    seam_filled_pixels: int
    seam_blocked_pixels: int
    inherited_filled_pixels: int


def resolve_rasterizer(requested):
    if requested not in {"auto", "native"}:
        raise ValueError("rasterizer must be 'auto' or 'native'")
    return "native"


class NativeRasterizer:
    """Prepared vc_reader sampling with the label-transfer result schema."""
    def __init__(self, **kwargs):
        self.sampler = SurfaceTransfer(**kwargs)

    def rasterize(self, bounds):
        indices, valid, distances, stats = self.sampler.rasterize(bounds)
        return NativeTileResult(indices, valid, distances, *stats)
