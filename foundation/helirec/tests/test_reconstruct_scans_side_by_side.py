"""helirec::Reconstruction of more scans than two: the scan with the rotation axis in its field and two moved
sideways beside it (HA, TA, TA2), merged; from the detector's files, and from a projection store at half the
resolution that holds a part of each helix. A small synthetic acquisition written the way BLISS / Lima write one
(tests/test_reconstruct_tool.py), of a sample of three Gaussians: two near the axis, one 150 px from it, which the
outer scans alone record. The geometry calibration is made up for it in the step's own form ("model": "joint")."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from helirec import _core
from helirec import calibration as C
from helirec import geometry as G
from helirec.recon import HelicalRecon

h5py = pytest.importorskip("h5py")
from test_reconstruct_tool import COLS, PIXEL_UM, ROWS, _write  # noqa: E402

_HERE = Path(_core.__file__).resolve().parent
ZARR_TOOL = next((str(d / "helirec-projections-zarr") for d in (_HERE, *_HERE.parents[:2]) if (d / "helirec-projections-zarr").exists()),
                 None) or shutil.which("helirec-projections-zarr")
DATASET = "/entry_0000/instrument/det3/data"
PER_TURN, TURNS, TRAVEL = 300, 3, 40.0          # frames per turn, turns, rows the helix travels per turn
OFFSETS = (20.5, 90.5, 160.5)                   # each scan's rotation axis from the detector's middle column
NAMES = ("s_B_HA", "s_B_TA", "s_C_TA2")         # (the last column under another sample name)
OUTER = (150.0, 40.0, 0.5, 7.0)                 # the third Gaussian: radius, direction, amplitude, sigma
PIXEL_CM = PIXEL_UM * 1e-4
PEAK_NEAR = 0.6 / (9.0 * np.sqrt(2 * np.pi)) / PIXEL_CM      # the density at the middle of a Gaussian of the sample [cm^-1]
PEAK_OUTER = OUTER[2] / (OUTER[3] * np.sqrt(2 * np.pi)) / PIXEL_CM


def _geometry(masters, n, first=0, end=None):
    """The step's file for the three scans: each one helix of n frames, of which [first, end) were on disk"""
    end = n if end is None else end
    heights = TRAVEL / PER_TURN * np.arange(n)
    stretch = dict(name="helix", index=0, helix=True, frames=n, first=first, end=end, offset=0)
    axes = [(COLS - 1) / 2 + off for off in OFFSETS]
    mid = float(heights[(first + end) // 2])
    return dict(step="geometry", model="joint", scans=masters,
                layout=dict(rows=ROWS, cols=COLS, frames_per_turn=PER_TURN, pixel_um=PIXEL_UM, row_sign=-1.0, anchor=0,
                            scans=[dict(name=name.split("_")[-1], mid_height=mid, coarse_axis=a, stretches=[stretch]) for name, a in zip(NAMES, axes)]),
                row_time=dict(seconds_per_row=0.0, frame_period_s=0.03, frames_per_row=0.0, measured=None), map=G.empty_map(ROWS, COLS),
                parameters=dict(roll=0.0, pitch=0.0, stage=dict(height_once=[0.0, 0.0], axis_twice=[0.0, 0.0]), about_height=0.0,
                                scans=[dict(name=name.split("_")[-1], axis=a, axis_slope=0.0, axial_scale=0.0, column_scale=0.0, d=[0.0, 0.0], height=0.0,
                                            stretch_height=dict(helix=0.0), mid_height=mid, height_once=[0.0, 0.0], rotation=[0.0, 0.0, 0.0]) for name, a in zip(NAMES, axes)]))


@pytest.fixture(scope="module")
def acquisition(tmp_path_factory):
    """(the projections folder, the masters, the frames per scan)"""
    root = tmp_path_factory.mktemp("side_by_side")
    n = PER_TURN * TURNS
    angles, heights = 360.0 / PER_TURN * np.arange(n), TRAVEL / PER_TURN * np.arange(n)
    masters = [_write(root / "projections", name, angles, heights, off, outer=OUTER) for name, off in zip(NAMES, OFFSETS)]
    return root, masters, n


def _peak(img, centre, r_lo, r_hi):
    """the largest value among the pixels r_lo..r_hi from the centre, and how far from the centre it lies"""
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    r = np.hypot(xx - centre, yy - centre)
    masked = np.where((r >= r_lo) & (r <= r_hi), img, -np.inf)
    k = np.unravel_index(np.argmax(masked), img.shape)
    return float(img[k]), float(r[k]), k


def _centroid(img, at, reach=14):
    """the centre of mass (y, x) of what lies above half the value at `at`, within `reach` pixels of it"""
    y0, x0 = at[0] - reach, at[1] - reach
    box = img[y0:at[0] + reach + 1, x0:at[1] + reach + 1].astype(np.float64)
    w = np.where(box > 0.5 * img[at], box, 0.0)
    yy, xx = np.mgrid[:box.shape[0], :box.shape[1]]
    return np.array([y0 + (w * yy).sum() / w.sum(), x0 + (w * xx).sum() / w.sum()])


def test_three_scans_from_the_detectors_files(tmp_path, acquisition):
    root, masters, n = acquisition
    cal = _geometry(masters, n)
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(cal, default=float))
    # the geometry reader: every scan's poses, and the merged frame they lay out
    angles = [360.0 / PER_TURN * np.arange(n)] * 3
    heights = [TRAVEL / PER_TURN * np.arange(n)] * 3
    poses, _ = G.scan_poses(cal, angles, heights)
    assert len(poses) == 3 and [float(np.median(p["axis"])) for p in poses] == [(COLS - 1) / 2 + off for off in OFFSETS]
    assert [G.frames_placed(cal, k) for k in range(3)] == [[(0, n)]] * 3
    layout = G.merged_layout(cal, poses, heights)
    assert layout["cor_main_detector"] == 140.0 and layout["vectors"] == [(0.0, -70.0), (0.0, -140.0)]
    placed, width, anchor = C.placement_from_vectors(layout["vectors"], COLS)
    assert [p.shift_x for p in placed] == [140, 70, 0] and width == COLS + 140 and anchor == 140.0

    r = HelicalRecon(root / "projections", cache_dir=str(tmp_path / "cache"), device="cpu", params=dict(geometry=str(path), sharpness_weighting="off"), log=lambda *a: None)
    assert [s.name for s in r.scans] == [name + "_0001" for name in NAMES] and r.cal.scans == [name + "_0001" for name in NAMES]
    assert r.cal.merged_width == COLS + 140 and r.cal.rot_center == 280.0 and r.info["grid_size"] == 561
    assert {k: r.cal.apodisation[k] for k in r.cal.scans} == {"s_B_HA_0001": (50.0, 100.0), "s_B_TA_0001": (50.0, 50.0), "s_C_TA2_0001": (0.0, 50.0)}
    h = (r.lo + r.hi) // 2
    img = r.reconstruct_heights(h, h + 1)[0]
    assert img.shape == (561, 561) and np.isfinite(img).all()
    value, radius, where = _peak(img, 280, 0, 60)
    assert value == pytest.approx(PEAK_NEAR, rel=0.04) and radius in (pytest.approx(30, abs=1.0), pytest.approx(20, abs=1.0))
    # the Gaussian that the outer scans alone record: where it is, and as dense as it is
    value, radius, outer_at = _peak(img, 280, 110, 190)
    assert value == pytest.approx(PEAK_OUTER, rel=0.04) and radius == pytest.approx(OUTER[0], abs=1.0)
    assert abs(img[np.hypot(*(np.mgrid[:561, :561] - 280)) > 200]).max() < 0.03 * PEAK_OUTER     # nothing beyond the sample
    # where it is: 150 px from the axis; and each outer scan alone holds it at the same place of the merged grid
    merged_at = _centroid(img, outer_at)
    assert np.hypot(*(merged_at - 280)) == pytest.approx(OUTER[0], abs=0.25)
    for only in ("companion", "ta2"):
        r.set_params(only_scan=only, grid_size=561)
        alone = r.reconstruct_heights(h, h + 1)[0]
        value, radius, at = _peak(alone, 280, 110, 190)
        # (a scan alone is a truncated sinogram: the ramp filter's tail is cut, and its values are off by a slow term)
        assert value == pytest.approx(PEAK_OUTER, rel=0.15) and np.abs(_centroid(alone, at) - merged_at).max() < 0.25
    with pytest.raises(ValueError, match="only_scan must be"):
        r.set_params(only_scan="ta3")


def test_the_calibrations_fan_reaches_the_backprojector(tmp_path, acquisition):
    """A calibration that states the beam's divergence (its layout's fan, frames of angle per column): the
    reconstruction hands it to the exact backprojector in radians per column, every merged column counted from the
    middle column of the scan that holds it; beam_divergence=false and the other backprojectors take a parallel beam."""
    root, masters, n = acquisition
    cal = _geometry(masters, n)
    cal["layout"]["fan"] = -2e-3
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(cal, default=float))
    r = HelicalRecon(root / "projections", cache_dir=str(tmp_path / "cache"), device="cpu", params=dict(geometry=str(path), sharpness_weighting="off"), log=lambda *a: None)
    k = -2e-3 * 2 * np.pi / PER_TURN
    assert r.params["beam_divergence"] is True and r.beam_divergence() == pytest.approx(k, rel=1e-12) and r.info["fan_per_column"] == pytest.approx(k, rel=1e-12)
    assert r.provenance()["resolved_beam_divergence"] == pytest.approx(k, rel=1e-12)
    fan = np.array(r._r.reconstructor.fan_of_columns)
    assert fan.shape == (COLS + 140,)
    # the scans' column 0 lies at the merged columns 140, 70 and 0 (the third scan alone holds the first columns, the first the last)
    np.testing.assert_allclose(fan[:40], k * (np.arange(40) - (COLS - 1) / 2), rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(fan[-40:], k * (np.arange(COLS + 100, COLS + 140) - 140 - (COLS - 1) / 2), rtol=1e-5, atol=1e-9)
    h = (r.lo + r.hi) // 2
    with_fan = r.reconstruct_heights(h, h + 1)[0]
    r.set_params(beam_divergence=False)
    assert r.beam_divergence() == 0 and r.info["fan_per_column"] == 0 and len(r._r.reconstructor.fan_of_columns) == 0
    parallel = r.reconstruct_heights(h, h + 1)[0]
    assert np.abs(with_fan - parallel).max() > 0.005 * np.abs(parallel).max()        # (smooth Gaussians, turned by under a pixel)
    # a calibration without a fan: the same slices whatever the parameter says
    del cal["layout"]["fan"]
    path.write_text(json.dumps(cal, default=float))
    none = HelicalRecon(root / "projections", cache_dir=str(tmp_path / "cache"), device="cpu", params=dict(geometry=str(path), sharpness_weighting="off"), log=lambda *a: None)
    assert none.beam_divergence() == 0 and none.info["fan_per_column"] == 0
    np.testing.assert_array_equal(none.reconstruct_heights(h, h + 1)[0], parallel)
    # another backprojector takes the beam as parallel
    cal["layout"]["fan"] = -2e-3
    path.write_text(json.dumps(cal, default=float))
    other = HelicalRecon(root / "projections", cache_dir=str(tmp_path / "cache"), device="cpu", params=dict(geometry=str(path), sharpness_weighting="off", backprojector="ghbp", angle_interp="linear"), log=lambda *a: None)
    assert other.beam_divergence() == pytest.approx(k, rel=1e-12) and other.info["fan_per_column"] == 0


@pytest.mark.skipif(ZARR_TOOL is None, reason="helirec-projections-zarr is not built")
def test_three_scans_from_a_projection_store_at_a_half(tmp_path, acquisition):
    """The store holds the middle of each helix, the darks and the flats at level 1; the masters lie in a folder of
    their own, without the detector's files. Every level keeps the stored pixels (j, k) 2^l, as a reduced read does."""
    root, masters, n = acquisition
    first, end, reduce = PER_TURN // 2, n - PER_TURN // 2, 1
    scale = tmp_path / "scale_1"
    jobs, tiles = [], []
    for name, master in zip(NAMES, masters):
        (tmp_path / "metadata" / name).mkdir(parents=True)
        shutil.copy(master, tmp_path / "metadata" / name / f"{name}_0001.h5")     # (the downloader's metadata folder)
        scan = _core.open_scan(master)
        for directory, role in (("scan0002", "dark"), ("scan0003", "flat"), ("scan0004", "projection"), ("scan0005", "flat")):
            with h5py.File(Path(master).parent / directory / "det3_0000.h5", "r") as f:
                frames = f[DATASET][:]
            lo, hi = (first, end) if role == "projection" else (0, len(frames))
            data = frames[lo:hi, ::2, ::2]
            output = f"{name}/{directory}/part.h5"
            (scale / output).parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(scale / output, "w") as f:
                f.create_dataset(DATASET, data=data if role == "projection" else data.astype(np.float32), chunks=(1, ROWS // 2, COLS // 2))
                f.attrs["complete"] = True
            jobs.append(dict(tile=name, output=output, relative_file=f"{directory}/det3_0000.h5", source_first=lo, role=role, rows=ROWS, cols=COLS, first=lo, count=hi - lo,
                             dtype="uint16", dataset=DATASET, reduce=reduce, metadata={}))
        tiles.append(dict(label=name, rows=ROWS, cols=COLS, chunk_rows=64))
        assert len(scan.unwrapped_angles_deg) == n
    (scale / "plan.json").write_text(json.dumps(dict(settings=dict(reduce=reduce), tiles=tiles, jobs=jobs)))
    store = tmp_path / "projections.zarr"
    run = subprocess.run([ZARR_TOOL, str(store), str(scale)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    in_metadata = [str(tmp_path / "metadata" / name / f"{name}_0001.h5") for name in NAMES]
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(_geometry(in_metadata, n, first, end), default=float))

    r = HelicalRecon(tmp_path / "metadata", device="cpu", params=dict(geometry=str(path), sharpness_weighting="off"), log=lambda *a: None, projection_store=store, reduce=reduce)
    assert [s.name for s in r.scans] == [name + "_0001" for name in NAMES]                 # found without their scan folders, the family joined
    assert [r._r.frames(k) for k in range(3)] == [(first, end)] * 3 and r._r.reduce == 1
    assert (r.cal.rows, r.cal.cols, r.cal.reduce) == (ROWS // 2, COLS // 2, 1) and r.cal.projection_store == str(store)
    assert r.cal.merged_width == COLS // 2 + 70 and r.cal.rot_center == 140.0 and r.info["grid_size"] == 281
    assert r.arrays["companion2_dark"].shape == (ROWS // 2, COLS // 2) and r.arrays["main_flats"].shape == (2, ROWS // 2, COLS // 2)
    provenance = r.provenance()
    assert provenance["frames"]["reduce"] == 1 and provenance["frames"]["scans"][2]["first"] == first and provenance["pixel_size_m"] == pytest.approx(2 * PIXEL_UM * 1e-6)
    h = (r.lo + r.hi) // 2
    img = r.reconstruct_heights(h, h + 1)[0]
    assert img.shape == (281, 281) and np.isfinite(img).all()
    # the same sample on pixels twice as large: the outer Gaussian at half the radius, as dense as it is
    value, radius, _ = _peak(img, 140, 55, 95)
    assert value == pytest.approx(PEAK_OUTER, rel=0.06) and radius == pytest.approx(OUTER[0] / 2, abs=1.0)
    value, radius, _ = _peak(img, 140, 0, 30)
    assert value == pytest.approx(PEAK_NEAR, rel=0.06)
    # refused: reduced frames without a store, and the reference's mode on a store
    with pytest.raises(ValueError, match="projection store"):
        HelicalRecon(tmp_path / "metadata", device="cpu", params=dict(geometry=str(path)), log=lambda *a: None, reduce=1)
    with pytest.raises(ValueError, match="legacy_reference"):
        HelicalRecon(tmp_path / "metadata", device="cpu", params=dict(geometry=str(path), mode="legacy_reference"), log=lambda *a: None, projection_store=store, reduce=reduce)
