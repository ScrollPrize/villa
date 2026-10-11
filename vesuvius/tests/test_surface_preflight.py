from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import tifffile

from vesuvius import surface_preflight
from vesuvius.tifxyz_label_transfer.io import load_surface


class FakeVolume:
    def __init__(self, data: np.ndarray, chunks: tuple[int, int, int] = (2, 2, 2)):
        self.data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.chunks = chunks

    def __getitem__(self, key):
        return self.data[key]


def write_surface(
    root: Path,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    z: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    scale: list[float] | None = None,
    bbox: list[list[float]] | None = None,
    meta_extra: dict[str, object] | None = None,
) -> Path:
    root.mkdir()
    x = np.asarray(x if x is not None else [[1, 2], [1, 2]], dtype=np.float32)
    y = np.asarray(y if y is not None else [[1, 1], [2, 2]], dtype=np.float32)
    z = np.asarray(z if z is not None else [[1, 1], [1, 1]], dtype=np.float32)
    tifffile.imwrite(root / "x.tif", x)
    tifffile.imwrite(root / "y.tif", y)
    tifffile.imwrite(root / "z.tif", z)
    if mask is not None:
        tifffile.imwrite(root / "mask.tif", np.asarray(mask, dtype=np.uint8))
    metadata: dict[str, object] = {
        "uuid": "fixture",
        "scale": scale if scale is not None else [1.0, 1.0],
    }
    if bbox is not None:
        metadata["bbox"] = bbox
    if meta_extra is not None:
        metadata.update(meta_extra)
    (root / "meta.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def test_inspect_pair_passes_valid_pair(tmp_path, monkeypatch) -> None:
    surface = write_surface(tmp_path / "surface")
    volume = FakeVolume(np.ones((4, 4, 4), dtype=np.uint16))
    monkeypatch.setattr(surface_preflight, "_open_volume", lambda *_: (volume, "0"))

    report = surface_preflight.inspect_pair(surface, "volume.zarr", max_samples=4)

    assert report["status"] == "PASS"
    assert report["summary"] == {"passed_required_gates": 10, "required_gate_count": 10}
    assert report["surface"]["valid_quad_count"] == 1
    assert report["volume"]["sampled_signal_support"]["support_fraction"] == 1.0


def test_inspect_pair_surface_only_passes_without_volume(tmp_path, monkeypatch) -> None:
    surface = write_surface(tmp_path / "surface")
    monkeypatch.setattr(
        surface_preflight,
        "_open_volume",
        lambda *_: (_ for _ in ()).throw(AssertionError("volume must stay closed")),
    )

    report = surface_preflight.inspect_pair(surface)

    assert report["status"] == "PASS"
    assert report["volume"] is None
    assert not {
        "volume_is_3d",
        "coordinates_within_volume",
        "sampled_volume_signal_support",
    } & {gate["name"] for gate in report["gates"]}


def test_spacing_median_handles_single_column_pair(tmp_path) -> None:
    surface = write_surface(
        tmp_path / "surface",
        x=np.asarray([[1000, 1020]], dtype=np.float32),
        y=np.asarray([[2000, 2000]], dtype=np.float32),
        z=np.asarray([[3000, 3000]], dtype=np.float32),
        scale=[0.05, 0.05],
    )

    report = surface_preflight.inspect_pair(surface)
    spacing = report["surface"]["grid_spacing_voxels"]["columns"]

    assert spacing["pair_count"] == 1
    assert spacing["median_spacing_voxels"] == pytest.approx(20, rel=0.02)


def test_spacing_accumulates_single_row_blocks(tmp_path) -> None:
    rows, columns = np.indices((64, 48), dtype=np.float32)
    surface = write_surface(
        tmp_path / "surface",
        x=1000 + columns * 20,
        y=2000 + rows * 20,
        z=np.full((64, 48), 3000, dtype=np.float32),
        scale=[0.05, 0.05],
    )

    report = surface_preflight.inspect_pair(surface, block_rows=1)
    spacing = report["surface"]["grid_spacing_voxels"]

    assert spacing["columns"]["pair_count"] == 64 * 47
    assert spacing["rows"]["pair_count"] == 63 * 48
    assert spacing["columns"]["median_spacing_voxels"] == pytest.approx(20, rel=0.02)
    assert spacing["rows"]["median_spacing_voxels"] == pytest.approx(20, rel=0.02)


def test_surface_only_fails_all_sentinel_grid(tmp_path) -> None:
    sentinel = np.full((6, 8), -1.0, dtype=np.float32)
    surface = write_surface(
        tmp_path / "surface",
        x=sentinel,
        y=sentinel,
        z=sentinel,
        scale=[0.05, 0.05],
    )

    report = surface_preflight.inspect_pair(surface)
    gates = {gate["name"]: gate for gate in report["gates"]}

    assert report["status"] == "FAIL"
    assert gates["valid_surface_vertices"]["observed"] == 0
    assert gates["valid_surface_vertices"]["passed"] is False
    assert gates["tifxyz_scale_consistency"]["passed"] is False
    assert gates["tifxyz_scale_consistency"]["observed"]["columns"]["pair_count"] == 0


def test_scale_consistency_fails_when_scale_disagrees_with_grid(tmp_path) -> None:
    rows, columns = np.indices((6, 8), dtype=np.float32)
    surface = write_surface(
        tmp_path / "surface",
        x=1000 + columns * 20,
        y=2000 + rows * 20,
        z=np.full((6, 8), 3000, dtype=np.float32),
        scale=[0.0002886, 0.0002886],
    )

    report = surface_preflight.inspect_pair(surface)
    gate = next(gate for gate in report["gates"] if gate["name"] == "tifxyz_scale_consistency")

    assert report["status"] == "FAIL"
    assert gate["passed"] is False
    assert gate["observed"]["columns"]["ratio"] == pytest.approx(20 * 0.0002886, rel=0.02)
    assert "scale disagrees" in gate["message"]


def test_scale_consistency_passes_anisotropic_within_tolerance(tmp_path) -> None:
    rows, columns = np.indices((6, 8), dtype=np.float32)
    surface = write_surface(
        tmp_path / "surface",
        x=1000 + columns * 20,
        y=2000 + rows * 30,
        z=np.full((6, 8), 3000, dtype=np.float32),
        scale=[0.05, 1 / 30],
    )

    report = surface_preflight.inspect_pair(surface)

    assert report["status"] == "PASS"
    assert next(
        gate for gate in report["gates"] if gate["name"] == "tifxyz_scale_consistency"
    )["passed"] is True


def test_scale_consistency_tolerance_is_configurable(tmp_path) -> None:
    rows, columns = np.indices((6, 8), dtype=np.float32)
    surface = write_surface(
        tmp_path / "surface",
        x=1000 + columns * 20,
        y=2000 + rows * 20,
        z=np.full((6, 8), 3000, dtype=np.float32),
        scale=[0.1, 0.1],
    )

    strict = surface_preflight.inspect_pair(surface, scale_tolerance=1.5)
    permissive = surface_preflight.inspect_pair(surface, scale_tolerance=3.0)

    assert strict["status"] == "FAIL"
    assert permissive["status"] == "PASS"


def test_scale_consistency_exact_match_passes_with_unit_tolerance(tmp_path) -> None:
    rows, columns = np.indices((6, 8), dtype=np.float32)
    surface = write_surface(
        tmp_path / "surface",
        x=1000 + columns * 32,
        y=2000 + rows * 32,
        z=np.full((6, 8), 3000, dtype=np.float32),
        scale=[1 / 32, 1 / 32],
    )

    report = surface_preflight.inspect_pair(surface, scale_tolerance=1.0)
    gate = next(gate for gate in report["gates"] if gate["name"] == "tifxyz_scale_consistency")

    assert report["status"] == "PASS"
    bounds = gate["observed"]["columns"]["median_spacing_bounds_voxels"]
    assert bounds[0] <= 32.0 <= bounds[1]
    assert gate["observed"]["columns"]["ratio_range"][0] <= 1.0 <= gate["observed"]["columns"]["ratio_range"][1]


def test_scale_consistency_passes_ratio_exactly_at_tolerance(tmp_path) -> None:
    rows, columns = np.indices((6, 8), dtype=np.float32)
    surface = write_surface(
        tmp_path / "surface",
        x=1000 + columns * 40,
        y=2000 + rows * 10,
        z=np.full((6, 8), 3000, dtype=np.float32),
        scale=[0.05, 0.05],
    )

    at_tolerance = surface_preflight.inspect_pair(surface, scale_tolerance=2.0)
    just_inside = surface_preflight.inspect_pair(surface, scale_tolerance=1.95)

    assert at_tolerance["status"] == "PASS"
    assert just_inside["status"] == "FAIL"


def test_bbox_consistency_fails_stale_bbox(tmp_path) -> None:
    rows, columns = np.indices((6, 8), dtype=np.float32)
    surface = write_surface(
        tmp_path / "surface",
        x=1000 + columns * 20,
        y=2000 + rows * 20,
        z=np.full((6, 8), 3000, dtype=np.float32),
        bbox=[[1000, 2000, 3000], [1020, 2040, 3000]],
    )

    report = surface_preflight.inspect_pair(surface)
    gate = next(gate for gate in report["gates"] if gate["name"] == "tifxyz_bbox_consistency")

    assert report["status"] == "FAIL"
    assert gate["passed"] is False
    assert gate["observed"]["max_excess_voxels"] > 0
    assert "bbox is stale" in gate["message"]


def test_bbox_consistency_passes_covering_bbox(tmp_path) -> None:
    rows, columns = np.indices((6, 8), dtype=np.float32)
    surface = write_surface(
        tmp_path / "surface",
        x=1000 + columns * 20,
        y=2000 + rows * 20,
        z=np.full((6, 8), 3000, dtype=np.float32),
        scale=[0.05, 0.05],
        bbox=[[1000, 2000, 3000], [1140, 2100, 3000]],
    )

    report = surface_preflight.inspect_pair(surface)
    gate = next(gate for gate in report["gates"] if gate["name"] == "tifxyz_bbox_consistency")

    assert report["status"] == "PASS"
    assert gate["passed"] is True


def test_bbox_consistency_rejects_malformed_bbox(tmp_path) -> None:
    surface = write_surface(tmp_path / "surface", meta_extra={"bbox": [1, 2, 3]})

    report = surface_preflight.inspect_pair(surface)
    gate = next(gate for gate in report["gates"] if gate["name"] == "tifxyz_bbox_consistency")

    assert report["status"] == "FAIL"
    assert gate["passed"] is False
    assert "2x3" in gate["observed"]["error"]


def test_bbox_gate_absent_when_metadata_has_no_bbox(tmp_path) -> None:
    report = surface_preflight.inspect_pair(write_surface(tmp_path / "surface"))

    assert "tifxyz_bbox_consistency" not in {gate["name"] for gate in report["gates"]}


def test_inspect_pair_fails_out_of_bounds_and_zero_support(tmp_path, monkeypatch) -> None:
    surface = write_surface(
        tmp_path / "surface",
        x=np.asarray([[1, 9], [1, 9]], dtype=np.float32),
    )
    volume = FakeVolume(np.zeros((4, 4, 4), dtype=np.uint16))
    monkeypatch.setattr(surface_preflight, "_open_volume", lambda *_: (volume, "0"))

    report = surface_preflight.inspect_pair(surface, "volume.zarr", max_samples=4)
    gates = {gate["name"]: gate for gate in report["gates"]}

    assert report["status"] == "FAIL"
    assert gates["coordinates_within_volume"]["passed"] is False
    assert gates["coordinates_within_volume"]["observed"] == {
        "out_of_bounds_count": 2
    }
    assert gates["sampled_volume_signal_support"]["passed"] is False


def test_inspect_pair_fails_nonfinite_selected_coordinate(tmp_path, monkeypatch) -> None:
    surface = write_surface(
        tmp_path / "surface",
        x=np.asarray([[1, np.nan], [1, 2]], dtype=np.float32),
        mask=np.full((2, 2), 255, dtype=np.uint8),
    )
    volume = FakeVolume(np.ones((4, 4, 4), dtype=np.uint16))
    monkeypatch.setattr(surface_preflight, "_open_volume", lambda *_: (volume, "0"))

    report = surface_preflight.inspect_pair(surface, "volume.zarr", max_samples=4)
    gates = {gate["name"]: gate for gate in report["gates"]}

    assert report["status"] == "FAIL"
    assert gates["finite_selected_coordinates"]["observed"] == 1


def test_inspect_pair_accepts_integer_scaled_tifxyz_mask(tmp_path, monkeypatch) -> None:
    mask = np.full((4, 4), 255, dtype=np.uint8)
    surface = write_surface(tmp_path / "surface", mask=mask)
    volume = FakeVolume(np.ones((4, 4, 4), dtype=np.uint16))
    monkeypatch.setattr(surface_preflight, "_open_volume", lambda *_: (volume, "0"))

    report = surface_preflight.inspect_pair(surface, "volume.zarr", max_samples=4)
    gates = {gate["name"]: gate for gate in report["gates"]}
    loaded = load_surface(surface)

    assert report["status"] == "PASS"
    np.testing.assert_array_equal(loaded.valid, np.ones((2, 2), dtype=bool))
    assert gates["tifxyz_mask_shape"]["passed"] is True
    assert gates["tifxyz_mask_shape"]["observed"] == [2, 2]


def test_inspect_pair_accepts_multipage_tifxyz_mask(tmp_path, monkeypatch) -> None:
    mask = np.full((2, 2, 2), 255, dtype=np.uint8)
    surface = write_surface(tmp_path / "surface", mask=mask)
    volume = FakeVolume(np.ones((4, 4, 4), dtype=np.uint16))
    monkeypatch.setattr(surface_preflight, "_open_volume", lambda *_: (volume, "0"))

    report = surface_preflight.inspect_pair(surface, "volume.zarr", max_samples=4)
    gates = {gate["name"]: gate for gate in report["gates"]}
    loaded = load_surface(surface)

    assert report["status"] == "PASS"
    np.testing.assert_array_equal(loaded.valid, np.ones((2, 2), dtype=bool))
    assert gates["tifxyz_mask_shape"]["passed"] is True
    assert gates["tifxyz_mask_shape"]["observed"] == [2, 2]


def test_inspect_pair_rejects_incompatible_tifxyz_mask(tmp_path, monkeypatch) -> None:
    surface = write_surface(
        tmp_path / "surface", mask=np.full((3, 4), 255, dtype=np.uint8)
    )
    monkeypatch.setattr(
        surface_preflight,
        "_open_volume",
        lambda *_: (_ for _ in ()).throw(AssertionError("volume must stay closed")),
    )

    report = surface_preflight.inspect_pair(surface, "volume.zarr", max_samples=4)
    gates = {gate["name"]: gate for gate in report["gates"]}

    assert report["status"] == "FAIL"
    assert gates["tifxyz_mask_shape"]["passed"] is False
    assert "incompatible with XYZ shape" in gates["tifxyz_mask_shape"]["observed"]["error"]


def test_inspect_pair_fails_nonfinite_implicit_valid_coordinate(
    tmp_path, monkeypatch
) -> None:
    surface = write_surface(
        tmp_path / "surface",
        x=np.asarray([[1, np.nan], [1, 2]], dtype=np.float32),
    )
    volume = FakeVolume(np.ones((4, 4, 4), dtype=np.uint16))
    monkeypatch.setattr(surface_preflight, "_open_volume", lambda *_: (volume, "0"))

    report = surface_preflight.inspect_pair(surface, "volume.zarr", max_samples=4)
    gates = {gate["name"]: gate for gate in report["gates"]}

    assert report["status"] == "FAIL"
    assert gates["finite_selected_coordinates"]["observed"] == 1


def test_margin_is_enforced_in_coordinate_space(tmp_path, monkeypatch) -> None:
    surface = write_surface(tmp_path / "surface")
    volume = FakeVolume(np.ones((4, 4, 4), dtype=np.uint16))
    monkeypatch.setattr(surface_preflight, "_open_volume", lambda *_: (volume, "0"))

    report = surface_preflight.inspect_pair(
        surface, "volume.zarr", margin=1.5, max_samples=4
    )
    gates = {gate["name"]: gate for gate in report["gates"]}

    assert report["status"] == "FAIL"
    assert gates["coordinates_within_volume"]["observed"] == {
        "out_of_bounds_count": 4
    }


def test_inspect_pair_reports_missing_files_without_throwing(tmp_path) -> None:
    surface = tmp_path / "surface"
    surface.mkdir()

    report = surface_preflight.inspect_pair(surface, "volume.zarr")

    assert report["status"] == "FAIL"
    assert report["gates"][0]["name"] == "tifxyz_required_files"
    assert sorted(report["gates"][0]["observed"]["missing"]) == [
        "meta.json",
        "x.tif",
        "y.tif",
        "z.tif",
    ]


def test_resolve_volume_array_prefers_multiscale_level_zero() -> None:
    expected = FakeVolume(np.ones((2, 2, 2), dtype=np.uint8))

    class FakeGroup(dict):
        attrs = {"multiscales": [{"datasets": [{"path": "level0"}]}]}

        def array_keys(self):
            return self.keys()

    array, key = surface_preflight._resolve_volume_array(
        FakeGroup(level0=expected), None
    )

    assert array is expected
    assert key == "level0"


def test_open_volume_accepts_direct_base_array_path(monkeypatch) -> None:
    expected = FakeVolume(np.ones((2, 2, 2), dtype=np.uint8))

    class FakeGroup(dict):
        attrs = {"multiscales": [{"datasets": [{"path": "0"}, {"path": "1"}]}]}

        def array_keys(self):
            return self.keys()

    class FakeZarr:
        @staticmethod
        def open(store, mode):
            assert store == "/volume.zarr"
            assert mode == "r"
            return FakeGroup({"0": expected, "1": expected})

    monkeypatch.setitem(sys.modules, "zarr", FakeZarr)

    array, key = surface_preflight._open_volume("/volume.zarr/0", None)

    assert array is expected
    assert key == "0"


def test_open_volume_rejects_direct_non_base_array_path(monkeypatch) -> None:
    expected = FakeVolume(np.ones((2, 2, 2), dtype=np.uint8))

    class FakeGroup(dict):
        attrs = {"multiscales": [{"datasets": [{"path": "0"}, {"path": "1"}]}]}

        def array_keys(self):
            return self.keys()

    class FakeZarr:
        @staticmethod
        def open(store, mode):
            assert store == "/volume.zarr"
            assert mode == "r"
            return FakeGroup({"0": expected, "1": expected})

    monkeypatch.setitem(sys.modules, "zarr", FakeZarr)

    with np.testing.assert_raises_regex(
        ValueError,
        "must select the base-resolution array.*expected '0', got '1'",
    ):
        surface_preflight._open_volume("/volume.zarr/1", None)


def test_open_volume_preserves_standalone_array_store(monkeypatch) -> None:
    expected = FakeVolume(np.ones((2, 2, 2), dtype=np.uint8))

    class FakeZarr:
        @staticmethod
        def open(store, mode):
            assert store == "/standalone.zarr"
            assert mode == "r"
            return expected

    monkeypatch.setitem(sys.modules, "zarr", FakeZarr)

    array, key = surface_preflight._open_volume("/standalone.zarr", None)

    assert array is expected
    assert key == ""


def test_resolve_volume_array_accepts_explicit_base_key() -> None:
    expected = FakeVolume(np.ones((2, 2, 2), dtype=np.uint8))

    class FakeGroup(dict):
        attrs = {
            "multiscales": [{"datasets": [{"path": "level0"}, {"path": "level1"}]}]
        }

        def array_keys(self):
            return self.keys()

    array, key = surface_preflight._resolve_volume_array(
        FakeGroup(level0=expected, level1=expected), "level0"
    )

    assert array is expected
    assert key == "level0"


def test_resolve_volume_array_rejects_non_base_key() -> None:
    expected = FakeVolume(np.ones((2, 2, 2), dtype=np.uint8))

    class FakeGroup(dict):
        attrs = {
            "multiscales": [{"datasets": [{"path": "level0"}, {"path": "level1"}]}]
        }

        def array_keys(self):
            return self.keys()

    with np.testing.assert_raises_regex(
        ValueError,
        "must select the base-resolution array.*expected 'level0', got 'level1'",
    ):
        surface_preflight._resolve_volume_array(
            FakeGroup(level0=expected, level1=expected), "level1"
        )


def test_main_writes_report_and_returns_fail_closed(tmp_path, monkeypatch) -> None:
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        surface_preflight,
        "inspect_pair",
        lambda *_args, **_kwargs: {"schema_version": 2, "status": "FAIL", "gates": []},
    )

    returncode = surface_preflight.main(
        [
            "--surface",
            str(tmp_path / "surface"),
            "--volume",
            "volume.zarr",
            "--output",
            str(output),
        ]
    )

    assert returncode == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "FAIL"


def test_main_surface_only_returns_fail_exit_code_and_prints_gate_summary(
    tmp_path, capsys
) -> None:
    sentinel = np.full((6, 8), -1.0, dtype=np.float32)
    surface = write_surface(
        tmp_path / "surface",
        x=sentinel,
        y=sentinel,
        z=sentinel,
        scale=[0.05, 0.05],
    )

    returncode = surface_preflight.main(["--surface", str(surface)])

    assert returncode == 2
    assert "valid_surface_vertices" in capsys.readouterr().err
