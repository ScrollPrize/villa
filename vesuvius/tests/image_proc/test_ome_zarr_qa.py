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

from vesuvius.scripts import ome_zarr_qa as qa
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


def test_declared_shape_short_z_axis_fails(tmp_path):
    """A declared shape that is short a z slice must FAIL."""
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


# Review regressions: path identity, dimensions, semantics, bounded I/O, coverage.


def _good_store(group, shape=(8, 8, 8), levels=3):
    _build_ceil_pyramid(group, base_shape=shape, num_levels=levels,
                        chunks=(8,) * len(shape), fill=True)
    _write_multiscales(group, num_levels=levels,
                      scales=[[2**l] * len(shape) for l in range(levels)],
                      units=["micrometer"] * len(shape))
    (group / "metadata.json").write_text(json.dumps({
        "scan": "s1", "voxelsize": 7.91, "z_direction_is_top_to_bottom": True}))
    return group


def _edit_ms(group, edit):
    attrs = json.loads((group / ".zattrs").read_text())
    edit(attrs["multiscales"][0])
    (group / ".zattrs").write_text(json.dumps(attrs))


@pytest.mark.parametrize("paths", [["0", "1", "99"], ["0", "0", "0"], ["1", "0", "2"]],
                         ids=["missing", "duplicate", "reordered"])
def test_dataset_paths_missing_duplicate_reordered_rejected(tmp_path, paths):
    group = _good_store(tmp_path / "vol.zarr")
    _edit_ms(group, lambda ms: [d.update(path=p) for d, p in zip(ms["datasets"], paths)])
    results = _by_name(run_all_checks(group))
    assert results["ome_scale_metadata"].status == FAIL
    assert results["level_z_consistency"].status == FAIL


@pytest.mark.parametrize("names", [("c", "z", "y", "x"), ("t", "c", "z", "y", "x")])
def test_channel_time_axes_without_units_pass_strict(tmp_path, names):
    group = tmp_path / "vol.zarr"
    spatial = [n in "zyx" for n in names]
    base = [2 if n in "tc" else 8 for n in names]
    for l in range(3):
        shape = [(_ceil_div(n, l) if sp else n) for n, sp in zip(base, spatial)]
        _write_level(group, l, shape, chunks=(8,) * len(names), fill=True)
    _write_multiscales(group, axes_names=names,
                      scales=[[2**l if sp else 1 for sp in spatial] for l in range(3)])
    def edit(ms):
        for a in ms["axes"]:
            a["type"] = {"c": "channel", "t": "time"}.get(a["name"], "space")
            if a["type"] == "space":
                a["unit"] = "micrometer"
    _edit_ms(group, edit)
    assert not [r for r in run_all_checks(group) if r.status in (WARN, FAIL)]
    assert qa.main([str(group), "--strict"]) == 0


@pytest.mark.parametrize("width", [2, 4], ids=["short", "oversized"])
def test_scale_vector_rank_mismatch_fails_without_crash(tmp_path, width):
    group = _good_store(tmp_path / "vol.zarr")
    _edit_ms(group, lambda ms: [d.update(coordinateTransformations=[{
        "type": "scale", "scale": [2**i] * width}]) for i, d in enumerate(ms["datasets"])])
    result = qa.check_ome_scale_metadata(group, [read_array_meta(i, group / str(i)) for i in range(3)])
    assert result.status == FAIL
    assert "rank" in result.message


@pytest.mark.parametrize("version,status", [("0.4", FAIL), ("0.10", FAIL), ("0.3", WARN)])
def test_axes_omitted_reports_version_appropriate_status(tmp_path, version, status):
    group = _good_store(tmp_path / "vol.zarr")
    def edit(ms):
        ms.pop("axes")
        ms["version"] = version
    _edit_ms(group, edit)
    assert _by_name(run_all_checks(group))["ome_scale_metadata"].status == status


def test_axes_wrong_length_fails(tmp_path):
    group = _good_store(tmp_path / "vol.zarr")
    _edit_ms(group, lambda ms: ms["axes"].pop())
    assert _by_name(run_all_checks(group))["ome_scale_metadata"].status == FAIL


@pytest.mark.parametrize("omit_types", [False, True])
def test_singleton_spatial_axis_scale_doubles_without_warning(tmp_path, omit_types):
    group = _good_store(tmp_path / "vol.zarr", shape=(1, 8, 8))
    if omit_types:
        _edit_ms(group, lambda ms: [a.pop("type") for a in ms["axes"]])
    assert not [r for r in run_all_checks(group) if r.status in (WARN, FAIL)]


def test_channel_shape_must_remain_constant(tmp_path):
    group = _good_store(tmp_path / "vol.zarr")
    _edit_ms(group, lambda ms: ms["axes"][0].update(name="c", type="channel"))
    assert _by_name(run_all_checks(group))["pyramid_level_shapes"].status == FAIL


def test_chunk_sampling_one_is_bounded_and_deterministic(tmp_path, monkeypatch):
    root = tmp_path / "chunks"
    for i in range(100):
        directory = root / str(i) / "0"
        directory.mkdir(parents=True)
        for j in range(10):
            (directory / str(j)).write_bytes(b"1")
    real = qa.os.scandir
    visited = []
    def scan(path):
        visited.append(str(path))
        return real(path)
    monkeypatch.setattr(qa.os, "scandir", scan)
    first = qa._chunk_files(root, 1)
    assert len(visited) == 3  # root, first branch, its leaf directory
    visited.clear()
    assert qa._chunk_files(root, 1) == first == [root / "0" / "0" / "0"]
    assert len(visited) == 3
    assert qa._chunk_files(root, 0) == []


@pytest.mark.parametrize("layout,expected", [
    ({"scan": {"voxelsize": "7.91"}}, 7.91),
    ({"scan": {"tomo": {"acquisition": {"detector": {"samplePixelSize": "0.00791"}}}}}, 7.91),
    ({"scan": {}, "tomo": {"acquisition": {"detector": {"samplePixelSize": 0.00791}}}}, 7.91),
    ({"scan": {}, "source": {"resolution": "2", "metadata": {
        "tomo": {"acquisition": {"detector": {"samplePixelSize": 0.00791}}}}}}, 31.64),
])
def test_voxel_size_resolver_supported_nested_sources(tmp_path, layout, expected):
    group = tmp_path / "vol.zarr"
    group.mkdir()
    layout["z_direction_is_top_to_bottom"] = True
    (group / "metadata.json").write_text(json.dumps(layout))
    result = check_metadata_json_keys(group)
    assert result.status == PASS
    assert result.details["voxel_size"]["value_um"] == pytest.approx(expected)
    assert "metadata.json:" in result.details["voxel_size"]["source"]


@pytest.mark.parametrize("value", [True, False, 0, -1, "nan", "inf", "bad"])
def test_voxel_size_unresolvable_values_fail(tmp_path, value):
    group = tmp_path / "vol.zarr"
    group.mkdir()
    (group / "metadata.json").write_text(json.dumps({
        "scan": {}, "voxelsize": value, "z_direction_is_top_to_bottom": True}))
    assert check_metadata_json_keys(group).status == FAIL


@pytest.mark.parametrize("key", ["voxelsize", "voxel_size_um", "voxelSizeUm", "pixel_size_um", "pixelSizeUm", "resolution_um"])
def test_voxel_size_meta_json_precedence_and_aliases(tmp_path, key):
    group = _good_store(tmp_path / "vol.zarr")
    (group / "meta.json").write_text(json.dumps({key: "12.5"}))
    result = check_metadata_json_keys(group)
    assert result.status == PASS
    assert result.details["voxel_size"]["value_um"] == 12.5
    assert "meta.json:" in result.details["voxel_size"]["source"]


def test_levels_limit_applies_to_datasets_and_cli(tmp_path):
    group = _good_store(tmp_path / "vol.zarr", shape=(32, 32, 32), levels=6)
    results = run_all_checks(group, levels=4)
    assert all(r.status == PASS for r in results), results
    assert qa.main([str(group), "--levels", "4", "--strict"]) == 0


@pytest.mark.parametrize("encoding", ["v2-dot", "v2-slash", "v3-default", "v3-dot", "v3-v2"])
@pytest.mark.parametrize("source_present", [True, False], ids=["missing_object_2092", "fill_region"])
def test_chunk_coverage_missing_object_and_fill_region(tmp_path, encoding, source_present):
    group = tmp_path / "vol.zarr"
    group.mkdir()
    for level, shape in enumerate([(8, 8, 8), (4, 4, 4)]):
        directory = group / str(level)
        directory.mkdir()
        sep = "." if encoding in ("v2-dot", "v3-dot", "v3-v2") else "/"
        if encoding.startswith("v2"):
            (directory / ".zarray").write_text(json.dumps({
                "shape": shape, "chunks": [4, 4, 4], "dtype": "|u1",
                **({"dimension_separator": sep} if sep == "/" else {})}))
        else:
            enc = {"name": "v2"} if encoding == "v3-v2" else {"name": "default"}
            if encoding == "v3-dot":
                enc["configuration"] = {"separator": "."}
            (directory / "zarr.json").write_text(json.dumps({
                "node_type": "array", "shape": shape, "data_type": "uint8",
                "chunk_grid": {"configuration": {"chunk_shape": [4, 4, 4]}},
                "chunk_key_encoding": enc}))
    metas = [read_array_meta(l, group / str(l)) for l in range(2)]
    if source_present:
        key = qa._chunk_key(metas[0], (0, 0, 0))
        path = group / "0" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source")
    result = qa.check_chunk_coverage(metas)
    assert result.status == (WARN if source_present else PASS)
    if source_present:
        assert result.details["offenders"] == [{"level": 1, "chunk": qa._chunk_key(metas[1], (0, 0, 0))}]


def test_chunk_coverage_targeted_lookups_bounded(tmp_path, monkeypatch):
    group = _good_store(tmp_path / "vol.zarr")
    metas = [read_array_meta(l, group / str(l)) for l in range(3)]
    calls = []
    real = qa.Path.is_file
    def is_file(path):
        calls.append(str(path))
        return real(path)
    monkeypatch.setattr(qa.Path, "is_file", is_file)
    result = qa.check_chunk_coverage(metas, sample=1)
    assert result.details["examined"] == 2
    assert len(calls) == 2
    assert qa.check_chunk_coverage(metas[:1]).status == SKIP


def test_chunk_coverage_2092_deleted_written_chunk_correct_shape(tmp_path):
    group = _good_store(tmp_path / "vol.zarr")
    coarse = group / "1" / "0" / "0" / "0"
    assert coarse.is_file()
    coarse.unlink()
    results = _by_name(run_all_checks(group))
    assert results["pyramid_level_shapes"].status == PASS
    assert results["chunk_coverage"].status == WARN
    assert {"level": 1, "chunk": "0/0/0"} in results["chunk_coverage"].details["offenders"]


def test_levels_limit_retains_store_wide_gap_check(tmp_path):
    group = _good_store(tmp_path / "vol.zarr")
    _write_level(group, 4, (1, 1, 1))
    result = _by_name(run_all_checks(group, levels=2))["level_z_consistency"]
    assert result.status == FAIL
    assert result.details["gaps"] == [3]


def test_voxel_size_meta_unresolvable_falls_back_and_scan_precedes_detector(tmp_path):
    group = _good_store(tmp_path / "vol.zarr")
    (group / "meta.json").write_text("{broken")
    (group / "metadata.json").write_text(json.dumps({
        "scan": {"voxelsize": "12.5", "tomo": {"acquisition": {
            "detector": {"samplePixelSize": 0.00791}}}},
        "z_direction_is_top_to_bottom": True}))
    result = check_metadata_json_keys(group)
    assert result.status == PASS
    assert result.details["voxel_size"]["value_um"] == 12.5
    assert result.details["voxel_size"]["source"].endswith(":scan.voxelsize")


@pytest.mark.parametrize("resolution", [None, True, -1, 1.5, 32, "nan"])
def test_voxel_size_source_requires_valid_resolution_level(tmp_path, resolution):
    group = _good_store(tmp_path / "vol.zarr")
    (group / "metadata.json").write_text(json.dumps({
        "scan": {}, "source": {"resolution": resolution, "metadata": {
            "tomo": {"acquisition": {"detector": {"samplePixelSize": 0.00791}}}}},
        "z_direction_is_top_to_bottom": True}))
    assert check_metadata_json_keys(group).status == FAIL


@pytest.mark.parametrize("sample", [2, 5, 24, 32])
def test_chunk_coverage_sample_coordinates_deterministic(sample):
    grid = (2, 3, 4)
    selected = list(qa._sample_chunk_coordinates(grid, sample))
    assert selected == list(qa._sample_chunk_coordinates(grid, sample))
    linear = [(z * 3 + y) * 4 + x for z, y, x in selected]
    count = min(24, sample)
    assert linear == [i * 23 // (count - 1) for i in range(count)]
    assert selected[0] == (0, 0, 0)
    assert selected[-1] == (1, 2, 3)
    assert len(set(selected)) == count


def test_chunk_coverage_sample_single_and_huge_grid():
    assert list(qa._sample_chunk_coordinates((2, 3, 4), 1)) == [(0, 0, 0)]
    assert list(qa._sample_chunk_coordinates((10**12, 10**12), 2)) == [
        (0, 0), (10**12 - 1, 10**12 - 1)]


def test_chunk_coverage_sample_detects_deleted_last_chunk(tmp_path):
    group = tmp_path / "vol.zarr"
    source = _write_level(group, 0, (16, 16, 16), fill=True)
    assert source.shape == (16, 16, 16)
    _write_level(group, 1, (8, 8, 8), fill=True)
    metas = [read_array_meta(l, group / str(l)) for l in range(2)]
    (group / "1" / "1" / "1" / "1").unlink()
    result = qa.check_chunk_coverage(metas, sample=3)
    assert result.status == WARN
    assert result.details["offenders"] == [{"level": 1, "chunk": "1/1/1"}]
    assert result.details["per_level"] == [{"level": 1, "examined": 3, "total": 8}]
