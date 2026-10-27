"""The Python tifxyz writer records how isometric a grid is.

ScrollPrize/villa issue #2279: a segment's meta.json has its area but nothing
about how isometric its grid is, so segments cannot be compared for stretch.
These tests pin the metric definitions on grids whose isometry is known by
construction, and pin the writer wiring.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from vesuvius.tifxyz import (
    Tifxyz,
    compute_grid_isometry,
    update_meta_isometry,
    write_tifxyz,
)


def _flat_grid(height: int, width: int, *, du: float, dv: float):
    """A flat grid in the xy-plane: x = col*du, y = row*dv, z = 0."""
    rows, cols = np.mgrid[0:height, 0:width]
    x = (cols * du).astype(np.float32)
    y = (rows * dv).astype(np.float32)
    z = np.zeros((height, width), dtype=np.float32)
    return x, y, z


def test_exact_grid_is_isometric() -> None:
    """A grid whose edges equal the promised step reports perfect isometry."""
    x, y, z = _flat_grid(20, 24, du=5.0, dv=5.0)
    block = compute_grid_isometry(x, y, z, scale=(0.2, 0.2))  # promised step 5 voxels

    assert block["version"] == 1
    assert block["step_vx"] == [5.0, 5.0]
    assert block["edges"] == 20 * 23 + 19 * 24
    assert block["within_10pct"] == 1.0
    assert block["within_25pct"] == 1.0
    assert block["median_abs_log_ratio"] == 0.0
    assert block["stretched_nodes"] == 0
    assert block["stretched_area_vx2"] == 0.0
    assert math.isclose(block["area_ratio"], 1.0, rel_tol=1e-6)
    assert math.isclose(block["angle_deg"]["p50"], 90.0, abs_tol=1e-4)
    assert math.isclose(block["l2_mean"], 1.0, rel_tol=1e-6)
    assert math.isclose(block["linf"], 1.0, rel_tol=1e-6)
    assert block["area_error"] == pytest.approx(0.0, abs=1e-9)


def test_stretched_grid_is_flagged() -> None:
    """A grid whose edges are 1.6x the promised step is stretched everywhere."""
    x, y, z = _flat_grid(20, 24, du=8.0, dv=8.0)
    block = compute_grid_isometry(x, y, z, scale=(0.2, 0.2))  # promised step 5 voxels

    assert block["within_10pct"] == 0.0
    assert block["within_25pct"] == 0.0
    assert math.isclose(block["median_abs_log_ratio"], math.log(1.6), rel_tol=1e-3)
    assert block["stretched_nodes"] == 20 * 24
    assert math.isclose(block["area_ratio"], (8.0 * 8.0) / (5.0 * 5.0), rel_tol=1e-3)
    assert math.isclose(block["l2_mean"], 1.6, rel_tol=1e-3)
    assert math.isclose(block["linf"], 1.6, rel_tol=1e-3)


def test_invalid_sentinel_corner_is_skipped() -> None:
    """An (-1, -1, -1) corner contributes no edge, exactly like the area metric."""
    x, y, z = _flat_grid(3, 3, du=5.0, dv=5.0)
    x[1, 1] = y[1, 1] = z[1, 1] = -1.0
    block = compute_grid_isometry(x, y, z, scale=(0.2, 0.2))

    # 4 horizontal + 4 vertical edges survive; the 4 that touched (1, 1) do not.
    assert block["edges"] == 8
    assert block["within_10pct"] == 1.0


def test_non_finite_corner_is_skipped() -> None:
    # A NaN corner drops the 2 edges that touch it (an interior sentinel drops 4).
    x, y, z = _flat_grid(3, 3, du=5.0, dv=5.0)
    x[0, 0] = np.nan
    block = compute_grid_isometry(x, y, z, scale=(0.2, 0.2))
    assert block["edges"] == 10


def test_mismatched_shapes_raise() -> None:
    x, y, z = _flat_grid(4, 4, du=1.0, dv=1.0)
    with pytest.raises(ValueError):
        compute_grid_isometry(x, y, z[:-1], scale=(1.0, 1.0))


def test_within_bands_are_multiplicative() -> None:
    """The band is a *factor* of the step: 0.90x is outside 1.1, 0.91x is inside."""
    inside = compute_grid_isometry(*_flat_grid(4, 4, du=5.0 * 0.91, dv=5.0), scale=(0.2, 0.2))
    assert inside["within_10pct"] == 1.0  # 0.91 >= 1/1.1 = 0.9091

    outside = compute_grid_isometry(*_flat_grid(4, 4, du=5.0 * 0.90, dv=5.0), scale=(0.2, 0.2))
    # only the v-edges (ratio exactly 1.0) count: 12 of 24
    assert outside["within_10pct"] == pytest.approx(0.5)


def _surface(height: int = 16, width: int = 16, *, step: float = 5.0) -> Tifxyz:
    """An isometric 16x16 grid with z > 0, so the format's validity test keeps it."""
    rows, cols = np.mgrid[0:height, 0:width]
    x = (cols * step).astype(np.float32)
    y = (rows * step).astype(np.float32)
    z = np.full((height, width), 3.0, dtype=np.float32)
    return Tifxyz(_x=x, _y=y, _z=z, uuid="segment", _scale=(1.0 / step, 1.0 / step), path=None)


def test_writer_emits_isometry_block(tmp_path) -> None:
    written = write_tifxyz(tmp_path / "seg.tifxyz", _surface())
    meta = json.loads((written / "meta.json").read_text(encoding="utf-8"))

    assert "isometry" in meta
    iso = meta["isometry"]
    assert iso["version"] == 1
    assert iso["step_vx"] == [5.0, 5.0]
    assert iso["within_10pct"] == 1.0
    assert iso["stretched_nodes"] == 0


def test_writer_can_skip_isometry(tmp_path) -> None:
    written = write_tifxyz(tmp_path / "seg.tifxyz", _surface(), write_isometry=False)
    meta = json.loads((written / "meta.json").read_text(encoding="utf-8"))
    assert "isometry" not in meta


def test_update_meta_backfill_stamps_meta_without_touching_tiffs(tmp_path) -> None:
    """The backfill adds the block to an existing segment; the TIFFs are untouched."""
    written = write_tifxyz(tmp_path / "seg.tifxyz", _surface(), write_isometry=False)
    assert "isometry" not in json.loads((written / "meta.json").read_text(encoding="utf-8"))
    before = {name: (written / name).read_bytes() for name in ("x.tif", "y.tif", "z.tif")}

    block = update_meta_isometry(written)

    meta = json.loads((written / "meta.json").read_text(encoding="utf-8"))
    assert meta["isometry"] == block
    assert block["within_10pct"] == 1.0
    for name, content in before.items():
        assert (written / name).read_bytes() == content


def test_update_meta_backfill_needs_a_segment(tmp_path) -> None:
    empty = tmp_path / "not_a_segment"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        update_meta_isometry(empty)
