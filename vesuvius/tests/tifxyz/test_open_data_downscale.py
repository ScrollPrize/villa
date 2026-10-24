"""Open-data ``original_volume_downscale`` for ``tifxyz_normalized`` meshes.

Normalized meshes of some open-data segments are stored at 1/original_volume_downscale
of the original volume (x4 for PHercParis4 and PHerc0139, x2 for PHerc0500P2 on the
2026-10-07 catalog). The factor is in the bucket catalog
(samples[s].segments[g].properties.original_volume_downscale), not in meta.json.
The fixtures below reproduce that layout and catalog shape with synthetic coordinates.
"""

from __future__ import annotations

import gzip
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import tifffile

import vesuvius.tifxyz as tx
from vesuvius.tifxyz import list_tifxyz, read_tifxyz, write_tifxyz

# New names are looked up as tx.<name> inside each test (not imported at module level), so on a
# tree without this change every test fails on its own instead of one collection error.

SCALE = 0.05000000074505806  # float32 0.05 as the tracer meshes store it


def _grid(rows: int = 12, cols: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, c = np.mgrid[0:rows, 0:cols].astype(np.float32)
    x = 1000.0 + 20.0 * c + 0.25 * r
    y = 2000.0 + 20.0 * r
    z = 3000.0 + 1.5 * c - 0.75 * r
    x[0, 0] = y[0, 0] = z[0, 0] = -1.0  # holes, as published meshes have
    x[5, 7] = y[5, 7] = z[5, 7] = -1.0
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def _write_mesh(d: Path, bbox_from_points: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d.mkdir(parents=True)
    x, y, z = _grid()
    for name, a in (("x", x), ("y", y), ("z", z)):
        tifffile.imwrite(str(d / f"{name}.tif"), a)
    v = z > 0
    bbox = [[float(x[v].min()), float(y[v].min()), float(z[v].min())],
            [float(x[v].max()), float(y[v].max()), float(z[v].max())]]
    # the same keys a published tifxyz_normalized meta.json carries -- no downscale field
    (d / "meta.json").write_text(json.dumps(
        {"format": "tifxyz", "type": "seg", "uuid": d.parts[-4], "scale": [SCALE, SCALE], "bbox": bbox}))
    return x, y, z


def _normalized_dir(root: Path, sample: str, seg_dir: str) -> Path:
    return root / sample / "segments" / seg_dir / "mesh" / "intermediate" / "tifxyz_normalized"


CATALOG = {
    "samples": {
        "PHercTest": {
            "segments": {
                "20250108000004": {
                    "id": "20250108000004",
                    "long_id": "20250108000004-w029_test",
                    "properties": {"original_volume_downscale": 4},
                },
                "20250101000000": {
                    "id": "20250101000000",
                    "long_id": "20250101000000-w001_test",
                    "properties": {"original_volume_downscale": 1},
                },
                "20250102000000": {
                    "id": "20250102000000",
                    "long_id": "20250102000000-legacy",
                    "properties": {},
                },
            }
        }
    }
}


@pytest.fixture()
def ds4(tmp_path):
    d = _normalized_dir(tmp_path, "PHercTest", "20250108000004-w029_test")
    return d, _write_mesh(d)


def test_downscale_scales_valid_points_bbox_and_grid_scale(ds4) -> None:
    d, (x, y, z) = ds4
    plain = read_tifxyz(d, downscale=1)
    s = read_tifxyz(d, downscale=4)
    v = z > 0
    assert np.array_equal(s._valid_mask, v)
    np.testing.assert_array_equal(s._x[v], 4 * x[v])
    np.testing.assert_array_equal(s._y[v], 4 * y[v])
    np.testing.assert_array_equal(s._z[v], 4 * z[v])
    assert (s._x[~v] == -1).all() and (s._z[~v] == -1).all()  # holes keep the -1 marker
    assert s.bbox == tuple(4 * b for b in plain.bbox)
    assert s.get_scale_tuple() == (plain._scale[0] / 4, plain._scale[1] / 4)
    # one full-resolution pixel is still one voxel -- of the original volume now
    assert s.full_resolution_shape == (4 * plain.full_resolution_shape[0], 4 * plain.full_resolution_shape[1])
    assert s.extra["original_volume_downscale_applied"] == 4.0
    assert plain.extra["original_volume_downscale_applied"] == 1.0


def test_normalized_path_without_downscale_warns(ds4) -> None:
    d, (x, _, _) = ds4
    with pytest.warns(tx.OpenDataDownscaleWarning, match="original_volume_downscale"):
        s = read_tifxyz(d)
    np.testing.assert_array_equal(s._x[1:], x[1:])  # default still applies nothing


def test_explicit_downscale_and_other_paths_do_not_warn(ds4, tmp_path) -> None:
    d, _ = ds4
    other = tmp_path / "some" / "segment"
    _write_mesh(other)
    with warnings.catch_warnings():
        warnings.simplefilter("error", tx.OpenDataDownscaleWarning)
        read_tifxyz(d, downscale=1)
        read_tifxyz(d, downscale=4)
        read_tifxyz(other)


def test_catalog_lookup_by_id_and_long_id(tmp_path) -> None:
    assert tx.original_volume_downscale(CATALOG, "PHercTest", "20250108000004") == 4.0
    assert tx.original_volume_downscale(CATALOG, "PHercTest", "20250108000004-w029_test") == 4.0
    assert tx.original_volume_downscale(CATALOG, "PHercTest", "20250101000000-w001_test") == 1.0
    assert tx.original_volume_downscale(CATALOG, "PHercTest", "20250102000000") == 1.0  # absent -> 1
    with pytest.raises(KeyError):
        tx.original_volume_downscale(CATALOG, "PHercTest", "29990101000000")
    with pytest.raises(KeyError):
        tx.original_volume_downscale(CATALOG, "PHercNope", "20250108000004")
    gz = tmp_path / "metadata.json"
    gz.write_bytes(gzip.compress(json.dumps(CATALOG).encode()))  # the bucket serves it gzip-compressed
    assert tx.load_open_data_catalog(gz) == CATALOG
    assert tx.original_volume_downscale(gz, "PHercTest", "20250108000004") == 4.0
    plain = tmp_path / "metadata-plain.json"
    plain.write_text(json.dumps(CATALOG))
    assert tx.original_volume_downscale(str(plain), "PHercTest", "20250108000004") == 4.0


def test_read_open_data_normalized_uses_catalog_factor(ds4, tmp_path) -> None:
    d, (x, y, z) = ds4
    s = tx.read_open_data_normalized(d, CATALOG)
    v = z > 0
    np.testing.assert_array_equal(s._z[v], 4 * z[v])
    d1 = _normalized_dir(tmp_path, "PHercTest", "20250101000000-w001_test")
    _, _, z1 = _write_mesh(d1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", tx.OpenDataDownscaleWarning)
        s1 = tx.read_open_data_normalized(d1, CATALOG)
    np.testing.assert_array_equal(s1._z[z1 > 0], z1[z1 > 0])
    with pytest.raises(ValueError, match="cannot infer"):
        tx.read_open_data_normalized(tmp_path, CATALOG)
    s2 = tx.read_open_data_normalized(tmp_path / "PHercTest" / "segments" / "20250108000004-w029_test" / "mesh"
                                   / "intermediate" / "tifxyz_normalized", CATALOG,
                                   sample="PHercTest", segment="20250108000004")
    np.testing.assert_array_equal(s2._z[v], 4 * z[v])


def test_written_scaled_mesh_is_not_scaled_twice(ds4, tmp_path) -> None:
    d, (_, _, z) = ds4
    out = tmp_path / "rescaled"
    write_tifxyz(out, read_tifxyz(d, downscale=4), overwrite=True)
    again = read_tifxyz(out)
    np.testing.assert_array_equal(again._z[z > 0], 4 * z[z > 0])
    assert again.extra["original_volume_downscale_applied"] == 4.0
    with pytest.raises(ValueError, match="already"):
        read_tifxyz(out, downscale=4)
    # downscale=1 on an already-scaled mesh changes nothing and keeps the marker
    same = read_tifxyz(out, downscale=1)
    np.testing.assert_array_equal(same._z[z > 0], 4 * z[z > 0])
    assert same.extra["original_volume_downscale_applied"] == 4.0
    # a mesh written after an explicit downscale=1 is also marked; a later x4 is refused
    out1 = tmp_path / "marked-1"
    write_tifxyz(out1, read_tifxyz(d, downscale=1), overwrite=True)
    with pytest.raises(ValueError, match="already"):
        read_tifxyz(out1, downscale=4)


def test_vc3d_scaled_cache_marker_is_respected(tmp_path) -> None:
    # VC3D's Open Data cache scales a cached segment itself and records
    # vc_open_data_coordinates_scaled_to_original_volume in meta.json.
    d = _normalized_dir(tmp_path, "PHercTest", "20250108000004-w029_test")
    _, _, z = _write_mesh(d)
    meta = json.loads((d / "meta.json").read_text())
    meta["vc_open_data_coordinates_scaled_to_original_volume"] = 4.0
    (d / "meta.json").write_text(json.dumps(meta))
    with warnings.catch_warnings():
        warnings.simplefilter("error", tx.OpenDataDownscaleWarning)
        s = read_tifxyz(d)
    np.testing.assert_array_equal(s._z[z > 0], z[z > 0])
    with pytest.raises(ValueError, match="vc_open_data_coordinates_scaled_to_original_volume"):
        read_tifxyz(d, downscale=4)
    with pytest.raises(ValueError, match="already"):
        tx.read_open_data_normalized(d, CATALOG)


@pytest.mark.parametrize("bad", [0, -4, float("nan"), float("inf")])
def test_invalid_downscale_rejected(ds4, bad) -> None:
    d, _ = ds4
    with pytest.raises(ValueError):
        read_tifxyz(d, downscale=bad)


def test_tifxyz_info_load_passes_downscale(ds4, tmp_path) -> None:
    d, (_, _, z) = ds4
    (info,) = list_tifxyz(tmp_path)
    s = info.load(downscale=4)
    np.testing.assert_array_equal(s._z[z > 0], 4 * z[z > 0])


def test_catalog_aware_warning_names_factor_and_skips_factor_1(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VESUVIUS_OPEN_DATA_CATALOG", raising=False)
    d4 = _normalized_dir(tmp_path, "PHercTest", "20250108000004-w029_test")
    d1 = _normalized_dir(tmp_path, "PHercTest", "20250101000000-w001_test")
    x, _, _ = _write_mesh(d4)
    _write_mesh(d1)
    with pytest.warns(tx.OpenDataDownscaleWarning, match="original_volume_downscale=4"):
        s = read_tifxyz(d4, open_data_catalog=CATALOG)
    np.testing.assert_array_equal(s._x[1:], x[1:])  # warns, applies nothing
    with warnings.catch_warnings():
        warnings.simplefilter("error", tx.OpenDataDownscaleWarning)
        read_tifxyz(d1, open_data_catalog=CATALOG)
        s4 = read_tifxyz(d4, open_data_catalog=CATALOG, downscale="catalog")
    np.testing.assert_array_equal(s4._x[1:][s4._valid_mask[1:]], 4 * x[1:][s4._valid_mask[1:]])


def test_catalog_from_environment(tmp_path, monkeypatch) -> None:
    cat = tmp_path / "metadata.json.gz"
    cat.write_bytes(gzip.compress(json.dumps(CATALOG).encode()))
    monkeypatch.setenv("VESUVIUS_OPEN_DATA_CATALOG", str(cat))
    d4 = _normalized_dir(tmp_path, "PHercTest", "20250108000004-w029_test")
    _write_mesh(d4)
    with pytest.warns(tx.OpenDataDownscaleWarning, match="original_volume_downscale=4"):
        read_tifxyz(d4)


def test_optional_meta_json_field(tmp_path, monkeypatch) -> None:
    # Proposed optional meta.json field: original_volume_downscale. No catalog needed.
    monkeypatch.delenv("VESUVIUS_OPEN_DATA_CATALOG", raising=False)
    d = tmp_path / "any" / "mesh"
    x, _, _ = _write_mesh(d)
    meta = json.loads((d / "meta.json").read_text())
    meta["original_volume_downscale"] = 2
    (d / "meta.json").write_text(json.dumps(meta))
    with pytest.warns(tx.OpenDataDownscaleWarning, match="meta.json gives original_volume_downscale=2"):
        read_tifxyz(d)
    s = read_tifxyz(d, downscale="catalog")
    np.testing.assert_array_equal(s._x[1:][s._valid_mask[1:]], 2 * x[1:][s._valid_mask[1:]])


def test_downscale_catalog_without_source_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VESUVIUS_OPEN_DATA_CATALOG", raising=False)
    d = tmp_path / "plain"
    _write_mesh(d)
    with pytest.raises(ValueError, match="downscale='catalog'"):
        read_tifxyz(d, downscale="catalog")
