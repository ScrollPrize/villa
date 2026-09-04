"""Tifxyz.full_resolution_shape must match the canvas VC3D renders.

Tracer meshes (vc_grow_seg_from_seed, published w-series segments) store
``scale`` as the float32 value 0.05, which serialises as 0.05000000074505806.
volume-cartographer's ``QuadSurface::size()`` divides in float32 and truncates,
which yields the rounded canvas (152 / 0.05f -> 3040). The float64 quotient is
3039.99995..., so truncating it is one pixel short.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesuvius.tifxyz.types import Tifxyz, _full_resolution_extent

FLOAT32_SCALE = float(np.float32(0.05))  # 0.05000000074505806


def _surface(stored_h: int, stored_w: int, scale: float) -> Tifxyz:
    zeros = np.zeros((stored_h, stored_w), dtype=np.float32)
    return Tifxyz(
        _x=zeros,
        _y=zeros.copy(),
        _z=zeros.copy(),
        _scale=(scale, scale),
        path=None,
    )


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        # vc_grow_seg_from_seed patches on PHerc0358/0813/0826 (issue #1694)
        (152, 3040),
        # published PHerc0139 w035: 5820x5240 surface volume
        (291, 5820),
        (262, 5240),
        # PHerc1447 20250703025628 mesh/intermediate/tifxyz_original (202x215)
        (202, 4040),
        (215, 4300),
    ],
)
def test_float32_rounded_scale_matches_cpp_canvas(stored: int, expected: int) -> None:
    assert int(stored / FLOAT32_SCALE) == expected - 1, "float64 truncation is short"
    assert _full_resolution_extent(stored, FLOAT32_SCALE) == expected


def test_exact_scale_is_unchanged() -> None:
    assert _full_resolution_extent(205, 0.05) == 4100
    assert _full_resolution_extent(213, 0.05) == 4260
    assert _full_resolution_extent(84300, 1.0) == 84300


def test_shape_properties_agree_and_use_float32_division() -> None:
    surface = _surface(202, 215, FLOAT32_SCALE)
    assert surface.full_resolution_shape == (4040, 4300)
    assert surface.shape == (202, 215)
    assert surface.use_full_resolution().shape == (4040, 4300)
    assert surface.full_resolution_shape == surface.shape


def test_zero_scale_falls_back_to_stored_shape() -> None:
    surface = _surface(10, 12, 0.0)
    assert surface.full_resolution_shape == (10, 12)
