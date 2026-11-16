"""helirec::calibration::calibrate_intensity (helirec/calibration/intensity.hpp) through its bindings, on a pair of
small synthetic BLISS scans (tests/test_scan.py writes them): the reduction against nx2etf's arithmetic, the merged
frame, the caches (helirec's array files) and what invalidates them."""
import json

import numpy as np
import pytest

from helirec import _core, bliss, calibration

h5py = pytest.importorskip("h5py")
from test_scan import COLS, ENTRIES, ROWS, _expected_current, _write_scan  # noqa: E402

LAYOUT = dict(cor_main_detector=3.25, vector=(1.0, -2.5), measured=dict(axis_first_scan=3.3, axis_second_scan=5.8, second_scan_rows=-0.97, quantum=0.25))


def _pair(tmp_path):
    _write_scan(tmp_path, "s_HA", yrot_mm=0.00864)      # the axis a pixel right of the middle
    _write_scan(tmp_path, "s_TA", yrot_mm=0.03)
    p = bliss.pair_companions(bliss.discover_scans(str(tmp_path)))
    return p["main"], p["companion"]


def test_reduction(tmp_path):
    main, _ = _pair(tmp_path)
    dark, flats = calibration.reduce_darks_flats(main)
    k = {e[0]: i for i, e in enumerate(ENTRIES)}           # every frame of entry i holds the value i
    assert dark.shape == (ROWS, COLS) and flats.shape == (2, ROWS, COLS)
    assert (dark == k["2.1"]).all()
    current, kinds = _expected_current()
    for s, entry in enumerate(("3.1", "5.1")):
        ctl = current[kinds == 1][4 * s:4 * s + 4]
        f = np.float32(k[entry])
        expect = sum((dark + (f - dark) * (np.float32(0.2) / np.float32(c)) for c in ctl), np.zeros_like(dark)) / np.float32(4)
        np.testing.assert_array_equal(flats[s], expect)


def test_calibrate_and_its_caches(tmp_path):
    main, comp = _pair(tmp_path)
    cache = tmp_path / "cache"
    lines = []
    cal, arrays = calibration.calibrate(main, comp, LAYOUT, cache_dir=str(cache), log=lines.append)
    assert cal.main == main.name and cal.companion == comp.name and cal.version == calibration.CALIBRATION_VERSION
    pl, width, anchor = calibration.placement_from_vector(LAYOUT["vector"], COLS)
    assert cal.rot_center == 3.25 + anchor == 5.75 and cal.merged_width == width == COLS + 3
    assert (cal.placement[main.name].shift_x, cal.placement[main.name].x_readjustment) == (pl[0].shift_x, pl[0].x_readjustment) == (2, -0.5)
    assert cal.placement[comp.name].z_readjustment == 1.0
    assert cal.detector_blur_field == [] and cal.details["detector_blur"]["reason"] == "4 flats in the first series"
    assert any("detector blur: not measured" in line for line in lines)
    assert cal.details["layout"] == LAYOUT["measured"]
    np.testing.assert_array_equal(arrays["weight_map"], calibration.weight_map([arrays["companion_flats"], arrays["main_flats"]]))
    (aw_m, aw_c) = calibration.apodisation_widths(COLS, main.estimated_cor_from_motor_pix, LAYOUT["vector"][1])
    np.testing.assert_array_equal(arrays["main_weight"], arrays["weight_map"] * _core.apodisation_mask(COLS, aw_m[0], aw_m[1], 1e-8)[None, :])

    # the caches: an array file whose first line names its arrays, and calibration.json
    with open(cache / "calibration_arrays.bin", "rb") as fh:
        head = json.loads(fh.readline())
        data = fh.read()
    assert head["fingerprint"] == cal.fingerprint == calibration.acquisition_fingerprint(main, comp)
    names = [a["name"] for a in head["arrays"]]
    assert names == ["main_dark", "main_flats", "companion_dark", "companion_flats", "weight_map"]
    flats = head["arrays"][1]
    assert flats["shape"] == [2, ROWS, COLS]
    np.testing.assert_array_equal(np.frombuffer(data, "<f4", count=2 * ROWS * COLS, offset=flats["offset"]).reshape(2, ROWS, COLS), arrays["main_flats"])
    assert json.loads((cache / "calibration.json").read_text())["fingerprint"] == cal.fingerprint

    lines.clear()
    again, arrays2 = calibration.calibrate(main, comp, LAYOUT, cache_dir=str(cache), log=lines.append)
    assert any("calibration loaded from" in line for line in lines) and not any("reducing" in line for line in lines)
    for k in arrays:
        np.testing.assert_array_equal(arrays2[k], arrays[k])

    # another merged frame keeps the cached reduction and blur, and places the scans anew
    other = dict(LAYOUT, vector=(0.0, -3.0))
    moved, _ = calibration.calibrate(main, comp, other, cache_dir=str(cache), log=lines.append)
    assert moved.vector == (0.0, -3.0) and moved.rot_center == 6.25

    # replaced data under the same name: another fingerprint, the caches are ignored and made again
    f = main.referenced_files()[0]
    with h5py.File(f, "a") as h:
        h.attrs["touched"] = 1
    main = bliss.open_scan(main.master)
    assert calibration.acquisition_fingerprint(main, comp) != cal.fingerprint
    lines.clear()
    calibration.calibrate(main, comp, LAYOUT, cache_dir=str(cache), log=lines.append)
    assert any("other acquisition / version, ignored" in line for line in lines) and any("reducing" in line for line in lines)


def test_reduced_darks_flats_cache(tmp_path):
    main, _ = _pair(tmp_path)
    made = calibration.reduced_darks_flats(main, str(tmp_path / "c"), print)
    path = tmp_path / "c" / f"darks_flats_{main.name}.bin"
    assert path.exists()
    path.write_bytes(path.read_bytes().replace(b'"fingerprint": "', b'"fingerprint": "x', 1))   # another acquisition's
    lines = []
    again = calibration.reduced_darks_flats(main, str(tmp_path / "c"), lines.append)
    assert any("made again" in line for line in lines)
    for a, b in zip(made, again):
        np.testing.assert_array_equal(a, b)
