"""vesuvius.crop (local OME-Zarr crops that resume) and vesuvius.doctor (environment checks)."""

import json

import numpy as np
import pytest
import zarr

from vesuvius.data.utils import open_zarr, open_zarr_group
from vesuvius.tools import crop as vc
from vesuvius.tools import doctor


def _ome(tmp_path, data, scale=(7.91, 7.91, 7.91)):
    path = str(tmp_path / "src.zarr")
    root = open_zarr_group(path, mode="w")
    arr = open_zarr(f"{path}/0", mode="w", shape=data.shape, chunks=(8, 8, 8), dtype=data.dtype)
    arr[...] = data
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"],
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": list(scale)}]}],
    }]
    return path


def _data(shape=(40, 30, 20)):
    return np.random.default_rng(1).integers(0, 256, size=shape, dtype=np.uint8)


def test_downsample2_means_and_keeps_odd_edges():
    block = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    out = vc.downsample2(block)
    assert out.shape == (2, 2, 2) and out.dtype == np.uint8
    assert out[0, 0, 0] == round(block[:2, :2, :2].mean())
    assert out[1, 1, 1] == block[2, 2, 2]


def test_crop_copies_the_region_with_its_place_and_voxel_size(tmp_path):
    data = _data()
    src = _ome(tmp_path, data)
    out = str(tmp_path / "crop.zarr")
    vc.main([src, out, "--crop", "4:36,2:28,1:19", "--levels", "3", "--chunks", "8,8,8"])
    root = zarr.open_group(out, mode="r")
    np.testing.assert_array_equal(root["0"][...], data[4:36, 2:28, 1:19])
    assert root["1"].shape == (16, 13, 9)
    assert root["2"].shape == (8, 7, 5)
    ms = root.attrs["multiscales"][0]
    assert [a.get("unit") for a in ms["axes"]] == ["micrometer"] * 3
    level1 = ms["datasets"][1]["coordinateTransformations"]
    assert level1[0]["scale"] == pytest.approx([15.82] * 3)
    assert level1[1]["translation"] == pytest.approx([4 * 7.91, 2 * 7.91, 1 * 7.91])
    record = root.attrs[vc.PROGRESS_KEY]
    assert record["complete"] and record["crop"] == [[4, 36], [2, 28], [1, 19]]


def test_interrupted_crop_resumes_where_it_stopped(tmp_path, monkeypatch):
    data = _data()
    src = _ome(tmp_path, data)
    out = str(tmp_path / "crop.zarr")
    real_read = vc.read_block
    calls = {"n": 0}

    def flaky(arr, box):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ConnectionError("network dropped")
        return real_read(arr, box)

    monkeypatch.setattr(vc, "read_block", flaky)
    with pytest.raises(ConnectionError):
        vc.main([src, out, "--crop", "0:40,:,:", "--chunks", "8,8,8", "--levels", "1"])
    assert zarr.open_group(out, mode="r").attrs[vc.PROGRESS_KEY]["level0_done_z"] == 16

    reads = []
    monkeypatch.setattr(vc, "read_block", lambda arr, box: reads.append(box[0]) or real_read(arr, box))
    vc.main([src, out, "--crop", "0:40,:,:", "--chunks", "8,8,8", "--levels", "1", "--resume"])
    assert reads[0] == (16, 24)  # the first two slabs were not read again
    np.testing.assert_array_equal(zarr.open_group(out, mode="r")["0"][...], data)


def test_resume_refuses_a_different_crop(tmp_path):
    src = _ome(tmp_path, _data())
    out = str(tmp_path / "crop.zarr")
    vc.main([src, out, "--crop", "0:8,:,:", "--levels", "1"])
    with pytest.raises(SystemExit, match="continues only the same crop"):
        vc.main([src, out, "--crop", "8:16,:,:", "--levels", "1", "--resume"])


class _Response:
    def __init__(self, status):
        self.status_code = status

    def close(self):
        pass


def test_server_check_reports_status_and_failures():
    assert doctor.check_server("s", "u", get=lambda *a, **k: _Response(200)).status == "ok"
    refused = doctor.check_server("s", "u", get=lambda *a, **k: _Response(503))
    assert refused.status == "fail" and "503" in refused.detail

    def down(*a, **k):
        raise ConnectionError("no route to host")

    failed = doctor.check_server("s", "u", get=down)
    assert failed.status == "fail" and "ConnectionError" in failed.detail and failed.fix


def test_display_check_explains_headless_use(monkeypatch):
    monkeypatch.setattr(doctor.sys, "platform", "linux")
    for var in ("DISPLAY", "WAYLAND_DISPLAY", "QT_QPA_PLATFORM"):
        monkeypatch.delenv(var, raising=False)
    check = doctor.check_display()
    assert check.status == "warn" and "xvfb-run" in check.fix
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert doctor.check_display().status == "ok"


def test_doctor_offline_json(capsys):
    code = doctor.main(["--offline", "--json"])
    checks = json.loads(capsys.readouterr().out)
    names = {c["name"] for c in checks}
    assert {"python", "vesuvius", "GPU", "display", "CPUs", "disk"} <= names
    assert not names & set(doctor.DATA_SERVERS)
    assert all(c["status"] in ("ok", "warn", "fail") for c in checks)
    assert code == (1 if any(c["status"] == "fail" for c in checks) else 0)
