from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from vesuvius import surface_preflight


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
    (root / "meta.json").write_text(
        json.dumps({"uuid": "fixture", "scale": [1.0, 1.0]}),
        encoding="utf-8",
    )
    return root


def test_inspect_pair_passes_valid_pair(tmp_path, monkeypatch) -> None:
    surface = write_surface(tmp_path / "surface")
    volume = FakeVolume(np.ones((4, 4, 4), dtype=np.uint16))
    monkeypatch.setattr(surface_preflight, "_open_volume", lambda *_: (volume, "0"))

    report = surface_preflight.inspect_pair(surface, "volume.zarr", max_samples=4)

    assert report["status"] == "PASS"
    assert report["summary"] == {"passed_required_gates": 9, "required_gate_count": 9}
    assert report["surface"]["valid_quad_count"] == 1
    assert report["volume"]["sampled_signal_support"]["support_fraction"] == 1.0


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


def test_main_writes_report_and_returns_fail_closed(tmp_path, monkeypatch) -> None:
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        surface_preflight,
        "inspect_pair",
        lambda *_args, **_kwargs: {"schema_version": 1, "status": "FAIL"},
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
