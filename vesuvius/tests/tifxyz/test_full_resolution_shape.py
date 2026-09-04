"""Tifxyz.full_resolution_shape must match the canvas vc_render_tifxyz renders.

Tracer meshes (vc_grow_seg_from_seed, published w-series segments) store
``scale`` as the float32 value 0.05, which serialises as 0.05000000074505806.
The float64 quotient 152 / 0.05000000074505806 is 3039.99995..., so truncating
it is one pixel short of the 3040 the renderer uses. vc_render_tifxyz sizes
the canvas as ``lround(stored * (render_scale / float32(scale)))`` in double,
which is what ``_full_resolution_extent`` reproduces, including for the
non-round scales vc_obj2tifxyz writes and for exact halves (std::lround rounds
half away from zero, Python's round() half to even).
"""

from __future__ import annotations

import numpy as np
import pytest

from vesuvius.tifxyz.types import Tifxyz, _full_resolution_extent

FLOAT32_SCALE = float(np.float32(0.05))  # 0.05000000074505806


def _surface(stored_h: int, stored_w: int, scale_y: float, scale_x: float | None = None) -> Tifxyz:
    zeros = np.zeros((stored_h, stored_w), dtype=np.float32)
    return Tifxyz(
        _x=zeros,
        _y=zeros.copy(),
        _z=zeros.copy(),
        _scale=(scale_y, scale_y if scale_x is None else scale_x),
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
def test_float32_rounded_scale_matches_rendered_canvas(stored: int, expected: int) -> None:
    assert int(stored / FLOAT32_SCALE) == expected - 1, "float64 truncation is short"
    assert _full_resolution_extent(stored, FLOAT32_SCALE) == expected


@pytest.mark.parametrize(
    ("stored_wh", "scale_xy", "rendered_wh"),
    [
        # vc_obj2tifxyz outputs measured against `vc_render_tifxyz --scale 1 -g 0`
        # ("rendering WxH" line) by @Bullo27 on PR #1699; all W x H.
        ((272, 185), (0.0499802, 0.0498272), (5442, 3713)),
        ((14, 10), (0.0023985, 0.0024457), (5837, 4089)),
        ((1085, 737), (0.1998486, 0.1992154), (5429, 3700)),
        ((10841, 7361), (1.9987041, 1.9921989), (5424, 3695)),
        ((5421, 3681), (0.9992994, 0.9960730), (5425, 3696)),
        ((1994, 2001), (0.3675560, 0.5414373), (5425, 3696)),
        ((272, 185), (FLOAT32_SCALE, FLOAT32_SCALE), (5440, 3700)),
    ],
)
def test_non_round_obj2tifxyz_scales_match_vc_render_tifxyz(stored_wh, scale_xy, rendered_wh) -> None:
    (w, h), (sx, sy), (rw, rh) = stored_wh, scale_xy, rendered_wh
    assert (_full_resolution_extent(w, sx), _full_resolution_extent(h, sy)) == (rw, rh)
    surface = _surface(h, w, sy, sx)
    assert surface.full_resolution_shape == (rh, rw)


def test_exact_half_rounds_away_from_zero_like_lround() -> None:
    # 7 x 5 grid at scale 2.0: 3.5 -> 4 and 2.5 -> 3 (round() would give 4 and 2)
    assert _full_resolution_extent(7, 2.0) == 4
    assert _full_resolution_extent(5, 2.0) == 3
    assert round(2.5) == 2  # documents why round() is not used


def test_exact_scale_is_unchanged() -> None:
    assert _full_resolution_extent(205, 0.05) == 4100
    assert _full_resolution_extent(213, 0.05) == 4260
    assert _full_resolution_extent(84300, 1.0) == 84300


def test_extent_is_at_least_one() -> None:
    assert _full_resolution_extent(1, 4.0) == 1


def test_shape_properties_agree() -> None:
    surface = _surface(202, 215, FLOAT32_SCALE)
    assert surface.full_resolution_shape == (4040, 4300)
    assert surface.shape == (202, 215)
    assert surface.use_full_resolution().shape == (4040, 4300)
    assert surface.full_resolution_shape == surface.shape


def test_zero_scale_falls_back_to_stored_shape() -> None:
    surface = _surface(10, 12, 0.0)
    assert surface.full_resolution_shape == (10, 12)
