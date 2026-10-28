"""Tests for the tifxyz writer's grid isometry block (#2279)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import tifffile

from vesuvius.tifxyz import (
    Tifxyz,
    compute_grid_isometry,
    read_tifxyz,
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
    assert isinstance(block["angle_deg"], list) and len(block["angle_deg"]) == 3
    assert math.isclose(block["angle_deg"][1], 90.0, abs_tol=1e-4)
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
    # #2279: stretched_area_vx2 is nodes times ONE promised cell area (25),
    # not the summed 3D area of touched cells (19 * 23 * 64 here).
    assert block["stretched_area_vx2"] == 20 * 24 * 25.0
    assert math.isclose(block["area_ratio"], (8.0 * 8.0) / (5.0 * 5.0), rel_tol=1e-3)
    assert math.isclose(block["l2_mean"], 1.6, rel_tol=1e-3)
    assert math.isclose(block["linf"], 1.6, rel_tol=1e-3)


def test_stretched_area_counts_one_cell_per_node() -> None:
    """One pulled-out node: flagged area is the node count times one cell area."""
    x, y, z = _flat_grid(5, 5, du=5.0, dv=5.0)
    x[2, 2] += 10.0
    block = compute_grid_isometry(x, y, z, scale=(0.2, 0.2))
    assert block["stretched_nodes"] > 0
    assert block["stretched_area_vx2"] == block["stretched_nodes"] * 5.0 * 5.0


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


# ---------------------------------------------------------------------------
# Issue 1 of the review: a re-written segment must not carry its stale block.
# ---------------------------------------------------------------------------

def test_rewrite_after_grid_edit_recomputes_isometry(tmp_path) -> None:
    """Write, read, change the grid, write again: the block follows the grid."""
    seg = tmp_path / "seg.tifxyz"
    write_tifxyz(seg, _surface())
    first = json.loads((seg / "meta.json").read_text(encoding="utf-8"))["isometry"]
    assert first["within_10pct"] == 1.0

    loaded = read_tifxyz(seg)
    assert "isometry" in loaded.extra  # the reader keeps the block in `extra`
    loaded._x[...] *= 1.6  # stretch the grid, so the stored block becomes stale
    loaded.extra["keep_me"] = 7  # an unrelated extra field must survive

    write_tifxyz(seg, loaded, overwrite=True)

    meta = json.loads((seg / "meta.json").read_text(encoding="utf-8"))
    second = meta["isometry"]
    assert second != first
    assert second["within_10pct"] < 1.0
    assert second["stretched_nodes"] > 0
    assert second == compute_grid_isometry(
        loaded._x,
        loaded._y,
        loaded._z,
        valid=loaded.valid_vertex_mask,
        scale=loaded._scale,
    )
    assert meta["keep_me"] == 7


def test_writer_skip_isometry_drops_an_inherited_block(tmp_path) -> None:
    seg = tmp_path / "seg.tifxyz"
    write_tifxyz(seg, _surface())
    assert "isometry" in json.loads((seg / "meta.json").read_text(encoding="utf-8"))

    loaded = read_tifxyz(seg)
    assert "isometry" in loaded.extra
    write_tifxyz(seg, loaded, overwrite=True, write_isometry=False)

    assert "isometry" not in json.loads((seg / "meta.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Issue 2 of the review: the backfill follows the documented loader rules
# (lasagna/tifxyz_format.md 4.2 and 4.3).
# ---------------------------------------------------------------------------

def _write_segment(tmp_path, *, height: int = 6, width: int = 6) -> object:
    """Write a plain, all-valid segment (no mask) and return its directory."""
    rows, cols = np.mgrid[0:height, 0:width]
    surface = Tifxyz(
        _x=(cols * 5.0).astype(np.float32),
        _y=(rows * 5.0).astype(np.float32),
        _z=np.full((height, width), 3.0, dtype=np.float32),
        uuid="seg",
        _scale=(0.2, 0.2),
        path=None,
    )
    return write_tifxyz(tmp_path / "seg.tifxyz", surface, write_mask=False)


def _edges_for(valid: np.ndarray) -> int:
    """Valid grid edges, the same u-then-v count compute_grid_isometry reports."""
    u = (valid[:, :-1] & valid[:, 1:]).sum()
    v = (valid[:-1, :] & valid[1:, :]).sum()
    return int(u + v)


def test_backfill_treats_nonpositive_z_as_invalid(tmp_path) -> None:
    """The documented Z <= 0 rule (4.2) applies to the backfill."""
    seg = _write_segment(tmp_path)
    z = tifffile.imread(str(seg / "z.tif"))
    z[5, :] = 0.0  # an entire invalid row
    tifffile.imwrite(str(seg / "z.tif"), z)

    block = update_meta_isometry(seg)

    valid = np.ones((6, 6), dtype=bool)
    valid[5, :] = False
    assert block["edges"] == _edges_for(valid)


def test_backfill_mask_keeps_only_values_at_least_255(tmp_path) -> None:
    """A mask value below 255 invalidates (4.3); a `> 0` test would keep it."""
    seg = _write_segment(tmp_path)
    tifffile.imwrite(str(seg / "mask.tif"), np.full((6, 6), 100, dtype=np.uint8))

    assert update_meta_isometry(seg)["edges"] == 0  # every value 100 is invalid

    tifffile.imwrite(str(seg / "mask.tif"), np.full((6, 6), 255, dtype=np.uint8))
    block = update_meta_isometry(seg)
    assert block["edges"] == _edges_for(np.ones((6, 6), dtype=bool))


def test_backfill_mask_at_integer_multiple_scale(tmp_path) -> None:
    """A 2x mask maps each grid node to a 2x2 mask block (4.3)."""
    seg = _write_segment(tmp_path)
    mask = np.full((12, 12), 255, dtype=np.uint8)
    mask[4:6, 4:6] = 0  # the block that maps to grid node (2, 2)

    tifffile.imwrite(str(seg / "mask.tif"), mask)
    block = update_meta_isometry(seg)

    valid = np.ones((6, 6), dtype=bool)
    valid[2, 2] = False
    assert block["edges"] == _edges_for(valid)


def test_backfill_mask_with_noninteger_scale_is_ignored(tmp_path) -> None:
    """A mask that is not an integer multiple of the grid is skipped (4.3)."""
    seg = _write_segment(tmp_path)
    tifffile.imwrite(str(seg / "mask.tif"), np.zeros((3, 3), dtype=np.uint8))

    assert update_meta_isometry(seg)["edges"] == _edges_for(np.ones((6, 6), dtype=bool))


def test_backfill_mask_uses_sample_zero(tmp_path) -> None:
    """Only sample 0 counts: other channels of 255 cannot rescue a zero sample 0."""
    seg = _write_segment(tmp_path)
    mask = np.zeros((6, 6, 3), dtype=np.uint8)
    mask[..., 1] = 255
    mask[..., 2] = 255

    tifffile.imwrite(str(seg / "mask.tif"), mask, photometric="rgb")
    assert update_meta_isometry(seg)["edges"] == 0


def test_backfill_mask_uses_first_page(tmp_path) -> None:
    """A multi-page mask.tif uses page 0 (the C++ writer stores mask then image)."""
    seg = _write_segment(tmp_path)
    pages = np.stack(
        [np.zeros((6, 6), dtype=np.uint8), np.full((6, 6), 255, dtype=np.uint8)]
    )

    tifffile.imwrite(str(seg / "mask.tif"), pages)
    assert update_meta_isometry(seg)["edges"] == 0
