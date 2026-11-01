"""labels_to_zarr runs end to end on every zarr the package accepts (pyproject: zarr>=2.18.7,<4).

main and generate_pyramid_levels called Group.create_dataset, which zarr 3 removed, so the script
crashed under zarr 3 before writing any label; with --expand, get_initialized_chunk_indices then
called store.keys(), which zarr 3 stores do not have. Levels are now created through
create_zarr_array and chunk keys are listed through the zarr 3 store API when keys() is absent.
"""
import json
import sys

import cv2
import numpy as np
import zarr
from scipy.ndimage import distance_transform_edt

from vesuvius.scripts import labels_to_zarr as l2z
from vesuvius.tifxyz import Tifxyz, write_tifxyz


def _write_surface(folder):
    yy, xx = np.mgrid[4:28, 4:28].astype(np.float32)
    surface = Tifxyz(_x=xx, _y=yy, _z=np.full_like(xx, 20), uuid="plane", _mask=np.ones(xx.shape, dtype=bool))
    write_tifxyz(folder / "plane", surface)
    label = np.zeros(xx.shape, dtype=np.uint8)
    label[6:10, 6:10] = 255
    cv2.imwrite(str(folder / "plane" / "plane_inklabels.png"), label)
    written = np.zeros((48, 48, 48), dtype=bool)
    written[20, 10:14, 10:14] = True
    return written


def _run(tmp_path, monkeypatch, *extra):
    out = tmp_path / "out.zarr"
    monkeypatch.setattr(sys, "argv", [
        "labels_to_zarr", str(tmp_path / "surfaces"), "--label-suffix", "_inklabels", "--label-value", "255",
        "--output", str(out), "--shape", "48", "48", "48", "--chunks", "16", "16", "16",
        "--num-workers", "1", "--num-levels", "2", "--padding", "0", *extra,
    ])
    l2z.main()
    return out


def test_writes_expanded_labels_and_a_pyramid_level(tmp_path, monkeypatch):
    written = _write_surface(tmp_path / "surfaces")
    out = _run(tmp_path, monkeypatch, "--expand", "1")

    expected = np.where(distance_transform_edt(~written) <= 1, 255, 0).astype(np.uint8)
    np.testing.assert_array_equal(zarr.open(str(out / "0"), mode="r")[...], expected)
    np.testing.assert_array_equal(zarr.open(str(out / "1"), mode="r")[...], expected[::2, ::2, ::2])
    assert json.loads((out / ".zgroup").read_text())["zarr_format"] == 2
    assert not (out / "zarr.json").exists()


def test_keeps_the_remapped_fill_value_on_every_level(tmp_path, monkeypatch):
    written = _write_surface(tmp_path / "surfaces")
    remap = tmp_path / "remap.json"
    remap.write_text(json.dumps({"0": 2, "255": 1}))
    out = _run(tmp_path, monkeypatch, "--remap-json", str(remap))

    expected = np.where(written, 1, 2).astype(np.uint8)
    np.testing.assert_array_equal(zarr.open(str(out / "0"), mode="r")[...], expected)
    np.testing.assert_array_equal(zarr.open(str(out / "1"), mode="r")[...], expected[::2, ::2, ::2])
    for level in ("0", "1"):
        assert json.loads((out / level / ".zarray").read_text())["fill_value"] == 2
