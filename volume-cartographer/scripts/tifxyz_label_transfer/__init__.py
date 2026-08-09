"""Geometry-driven label transfer between TIFXYZ surfaces."""

from .core import (
    AffineChoice,
    MappingStats,
    Surface,
    SurfaceMapper,
    apply_affine,
    choose_affine_direction,
    infer_output_shape,
    load_affine,
    transfer_array,
)

__all__ = [
    "AffineChoice",
    "MappingStats",
    "Surface",
    "SurfaceMapper",
    "apply_affine",
    "choose_affine_direction",
    "infer_output_shape",
    "load_affine",
    "transfer_array",
]
