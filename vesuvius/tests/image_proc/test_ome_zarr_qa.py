"""Unit tests for the OME-Zarr pyramid/metadata QA tool.

Each test builds a *tiny* synthetic zarr pyramid in ``tmp_path`` and asserts the
QA checks catch (or pass) a specific class of real villa bug. No network, no slow
marks — these run by default. numpy + zarr only; arrays stay a few dozen voxels.

The key dimension used below is an axis of size 9: ``round(9 * 0.5) == 4`` under
Python's banker's rounding, while ``ceil(9 / 2) == 5`` — so the ``round()``-built
pyramid is one voxel short on that axis, reproducing villa #2135 at unit scale.
"""

import json

import numpy as np
import pytest

from vesuvius.scripts.ome_zarr_qa import (
    FAIL,
    PASS,
    SKIP,
    WARN,
    check_chunk_object_sizes,
    check_metadata_json_keys,
    check_ome_scale_metadata,
    check_pyramid_level_shapes,
    discover_levels,
    read_array_meta,
    run_all_checks,
)

zarr = pytest.importorskip("zarr")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _write_level(group_path, level, shape, chunks=(4, 4, 4), dtype="uint8", fill=False):
    """Create one zarr v2 level array (nested chunk dirs) under the group."""
    import numcodecs

    group_path.mkdir(parents=True, exist_ok=True)
    zgroup = group_path / ".zgroup"
    if not zgroup.exists():
        zgroup.write_text(json.dumps({"zarr_format": 2}))

    level_path = group_path / str(level)
    chunks = tuple(min(c, s) for c, s in zip(chunks, shape))
    arr = zarr.open(
        str(level_path),
        mode="w",
        shape=tuple(shape),
        chunks=chunks,
        dtype=dtype,
        dimension_separator="/",
        zarr_format=2,
        compressor=numcodecs.Blosc(cname="zstd", clevel=1),
    )
    if fill:
        # Write a cheap non-fill pattern so chunk files actually land on disk.
        arr[...] = np.ones(tuple(shape), dtype=dtype)
    return arr


def _ceil_div(n, level):
    return (n + (1 << level) - 1) >> level


def _build_ceil_pyramid(group_path, base_shape=(9, 8, 8), num_levels=3, **kw):
    """A correctly built pyramid: every level == ceil(base / 2**level) per axis."""
    for lvl in range(num_levels):
        shape = tuple(_ceil_div(d, lvl) for d in base_shape)
        _write_level(group_path, lvl, shape, **kw)
    return group_path


def _build_round_pyramid(group_path, base_shape=(9, 8, 8), num_levels=3, **kw):
    """A pyramid built with banker's rounding (the #2135 bug) — one voxel short."""
    prev = base_shape
    for lvl in range(num_levels):
        if lvl == 0:
            shape = base_shape
        else:
            # Exactly what build_pyramid does: max(1, round(0.5 * dim)) off the prev level.
            shape = tuple(max(1, int(round(0.5 * d))) for d in prev)
        _write_level(group_path, lvl, shape, **kw)
        prev = shape
    return group_path


def _write_multiscales(group_path, axes_names=("z", "y", "x"), num_levels=3,
                       scales=None, units=None):
    """Write an OME multiscales block into the group's .zattrs."""
    axes = []
    for i, a in enumerate(axes_names):
        entry = {"name": a, "type": "space"}
        if units is not None:
            entry["unit"] = units[i]
        axes.append(entry)
    datasets = []
    for lvl in range(num_levels):
        d = {"path": str(lvl)}
        if scales is not None:
            d["coordinateTransformations"] = [{"type": "scale", "scale": list(scales[lvl])}]
        datasets.append(d)
    zattrs = group_path / ".zattrs"
    zattrs.write_text(json.dumps({"multiscales": [{"version": "0.4", "axes": axes, "datasets": datasets}]}))


def _by_name(results):
    return {r.name: r for r in results}


# ---------------------------------------------------------------------------
# pyramid_level_shapes
# ---------------------------------------------------------------------------


def test_ceil_pyramid_passes_all_checks(tmp_path):
    group = tmp_path / "vol.zarr"
    _build_ceil_pyramid(group, fill=True)
    # Give it good metadata so nothing downgrades to WARN unexpectedly.
    _write_multiscales(
        group,
        scales=[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
        units=["micrometer", "micrometer", "micrometer"],
    )
    (group / "metadata.json").write_text(
        json.dumps({"scan": "s1", "voxelsize": 7.91, "z_direction_is_top_to_bottom": True})
    )

    results = _by_name(run_all_checks(group))
    assert results["pyramid_level_shapes"].status == PASS
    assert results["ome_scale_metadata"].status == PASS
    assert results["metadata_json_keys"].status == PASS
    assert results["level_z_consistency"].status == PASS
    # No check should be FAIL on a well-formed pyramid.
    assert not [r for r in results.values() if r.status == FAIL]


def test_round_pyramid_fails_pyramid_level_shapes(tmp_path):
    """Reproduces villa #2135: round() leaves the size-9 z axis one voxel short."""
    group = tmp_path / "vol.zarr"
    _build_round_pyramid(group, base_shape=(9, 8, 8))

    metas = [read_array_meta(l, group / str(l)) for l in discover_levels(group)]
    # Sanity: the builder really did produce the short axis.
    assert metas[1].shape[0] == 4  # round(9*0.5) == 4
    assert _ceil_div(9, 1) == 5    # VC3D expects 5

    result = check_pyramid_level_shapes(metas)
    assert result.status == FAIL
    viol = result.details["violations"][0]
    assert viol["axis"] == 0 and viol["expected"] == 5 and viol["actual"] == 4


def test_missing_last_z_slice_fails(tmp_path):
    """A level that is short a z slice (missing last z chunk) must FAIL (#2092)."""
    group = tmp_path / "vol.zarr"
    _write_level(group, 0, (8, 8, 8))
    _write_level(group, 1, (3, 4, 4))  # z should be ceil(8/2)==4, not 3

    metas = [read_array_meta(l, group / str(l)) for l in discover_levels(group)]
    result = check_pyramid_level_shapes(metas)
    assert result.status == FAIL
    viol = result.details["violations"][0]
    assert viol["axis"] == 0 and viol["expected"] == 4 and viol["actual"] == 3


def test_single_level_skips_shape_check(tmp_path):
    group = tmp_path / "vol.zarr"
    _write_level(group, 0, (8, 8, 8))
    metas = [read_array_meta(0, group / "0")]
    assert check_pyramid_level_shapes(metas).status == SKIP


# ---------------------------------------------------------------------------
# ome_scale_metadata
# ---------------------------------------------------------------------------


def test_absent_multiscales_warns_not_crashes(tmp_path):
    group = tmp_path / "vol.zarr"
    _build_ceil_pyramid(group)
    metas = [read_array_meta(l, group / str(l)) for l in discover_levels(group)]
    result = check_ome_scale_metadata(group, metas)
    assert result.status == WARN


def test_inconsistent_scale_ratio_fails(tmp_path):
    """A scale that does not halve between levels is a FAIL (#497 class)."""
    group = tmp_path / "vol.zarr"
    _build_ceil_pyramid(group)
    _write_multiscales(
        group,
        scales=[[1.0, 1.0, 1.0], [3.0, 3.0, 3.0], [9.0, 9.0, 9.0]],  # 3x, not 2x
        units=["micrometer", "micrometer", "micrometer"],
    )
    metas = [read_array_meta(l, group / str(l)) for l in discover_levels(group)]
    result = check_ome_scale_metadata(group, metas)
    assert result.status == FAIL


def test_missing_axis_unit_warns(tmp_path):
    group = tmp_path / "vol.zarr"
    _build_ceil_pyramid(group)
    _write_multiscales(
        group,
        scales=[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
        units=None,  # names present, units absent
    )
    metas = [read_array_meta(l, group / str(l)) for l in discover_levels(group)]
    result = check_ome_scale_metadata(group, metas)
    assert result.status == WARN
    assert "unit" in result.message


# ---------------------------------------------------------------------------
# chunk_object_sizes
# ---------------------------------------------------------------------------


def test_oversized_chunk_object_fails(tmp_path):
    """An 8x-too-large level-0 chunk file must FAIL (#1965)."""
    group = tmp_path / "vol.zarr"
    arr = _write_level(group, 0, (8, 8, 8), chunks=(4, 4, 4), fill=True)
    _write_level(group, 1, (4, 4, 4), fill=True)

    # declared chunk bytes = 4*4*4*1 = 64. Inflate one chunk file to 8x = 512 bytes.
    level0 = group / "0"
    chunk_files = [p for p in level0.rglob("*") if p.is_file() and not p.name.startswith(".")]
    assert chunk_files
    chunk_files[0].write_bytes(b"\x01" * (64 * 8))

    metas = [read_array_meta(l, group / str(l)) for l in discover_levels(group)]
    result = check_chunk_object_sizes(metas)
    assert result.status == FAIL
    assert result.details["max_ratio"] >= 2.0


def test_normal_chunks_pass_or_skip(tmp_path):
    group = tmp_path / "vol.zarr"
    _write_level(group, 0, (8, 8, 8), chunks=(4, 4, 4), fill=True)
    metas = [read_array_meta(0, group / "0")]
    result = check_chunk_object_sizes(metas)
    # Compressed ones-array chunks are tiny; never larger than declared.
    assert result.status in (PASS, SKIP)
    assert result.status != FAIL


# ---------------------------------------------------------------------------
# metadata_json_keys
# ---------------------------------------------------------------------------


def test_metadata_json_missing_scan_fails(tmp_path):
    """metadata.json without `scan` must FAIL (#2078)."""
    group = tmp_path / "vol.zarr"
    _write_level(group, 0, (8, 8, 8))
    (group / "metadata.json").write_text(
        json.dumps({"voxelsize": 7.91, "z_direction_is_top_to_bottom": True})
    )
    result = check_metadata_json_keys(group)
    assert result.status == FAIL
    assert "scan" in result.details["missing"]


def test_metadata_json_null_z_direction_warns(tmp_path):
    """metadata.json with null z-direction must WARN (#2106)."""
    group = tmp_path / "vol.zarr"
    _write_level(group, 0, (8, 8, 8))
    (group / "metadata.json").write_text(
        json.dumps({"scan": "s1", "voxelsize": 7.91, "z_direction_is_top_to_bottom": None})
    )
    result = check_metadata_json_keys(group)
    assert result.status == WARN
    assert "null" in result.message


def test_metadata_json_absent_skips(tmp_path):
    group = tmp_path / "vol.zarr"
    _write_level(group, 0, (8, 8, 8))
    assert check_metadata_json_keys(group).status == SKIP


# ---------------------------------------------------------------------------
# level_z_consistency + end-to-end CLI
# ---------------------------------------------------------------------------


def test_level_gap_fails(tmp_path):
    group = tmp_path / "vol.zarr"
    _write_level(group, 0, (8, 8, 8))
    _write_level(group, 2, (2, 2, 2))  # level 1 missing
    results = _by_name(run_all_checks(group))
    assert results["level_z_consistency"].status == FAIL
    assert results["level_z_consistency"].details["gaps"] == [1]


def test_cli_exit_code_nonzero_on_fail(tmp_path, capsys):
    from vesuvius.scripts.ome_zarr_qa import main

    group = tmp_path / "vol.zarr"
    _build_round_pyramid(group, base_shape=(9, 8, 8))
    rc = main([str(group)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "pyramid_level_shapes" in out and "FAIL" in out


def test_cli_json_output(tmp_path, capsys):
    from vesuvius.scripts.ome_zarr_qa import main

    group = tmp_path / "vol.zarr"
    _build_ceil_pyramid(group, fill=True)
    rc = main([str(group), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert "results" in payload and "summary" in payload
    assert rc == 0  # no FAILs (only WARN/SKIP allowed without --strict)


def test_cli_strict_fails_on_warn(tmp_path):
    from vesuvius.scripts.ome_zarr_qa import main

    group = tmp_path / "vol.zarr"
    _build_ceil_pyramid(group, fill=True)  # no multiscales => WARN
    assert main([str(group)]) == 0
    assert main([str(group), "--strict"]) == 2
