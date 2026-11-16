"""A NeXus NXtomo master as GDA writes it at Diamond I12 (src/calibration/nxtomo.hpp) through open_scan, discover_scans
and the calibration session: a synthetic master written the way the beamline's are (one frame file linked from the
master, image_key runs, the Zebra's angles and capture times, the motors before the scan, the fly scan's command),
with the three things the real ones showed: a record one frame shorter than the frame file, a snapshot of the
translation stage taken while it moved, and no pixel size in the file."""
import json

import numpy as np
import pytest

from vc_tomography import _core, bliss

h5py = pytest.importorskip("h5py")

ROWS, COLS = 6, 8
N_DARK, N_FLAT, N_PROJ = 3, 4, 73          # 5 degree steps over a turn and a frame
STEP, EXPOSURE, PERIOD_MS = 5.0, 0.02, 40.0


def _write(root, number, title, in_beam=-8.0, snapshot_x=None, height=48.5, flats_at_end=0, short_record=False,
           supplement=dict(pixel_size_um=7.91, energy_kev=53, propagation_distance_m=None), lit=None):
    # lit: (first, end): the rows the beam lights; the flats' other rows hold 6 % of it above the dark
    root.mkdir(parents=True, exist_ok=True)
    key = [2] * N_DARK + [1] * N_FLAT + [0] * N_PROJ + [1] * flats_at_end
    n = len(key)
    angle = np.zeros(n)
    proj = np.flatnonzero(np.array(key) == 0)
    # the Zebra's capture: the angle at the trigger, stored in steps of 0.005 degrees
    angle[proj] = np.round(np.arange(N_PROJ) * STEP * 1.00003 / 0.005) * 0.005
    angle[proj[-1] + 1:] = angle[proj[-1]]
    capture = np.zeros(n)
    capture[proj] = 5170.5 + PERIOD_MS * np.arange(N_PROJ)
    frames = np.empty((n, ROWS, COLS), np.uint16)
    for i, k in enumerate(key):
        frames[i] = {2: 100, 1: 4000}.get(k, 1000 + i)
        if k == 1 and lit is not None:
            frames[i, :lit[0]] = frames[i, lit[1]:] = 334
    with h5py.File(root / f"pco1-{number}.hdf", "w") as f:
        d = f.create_dataset("entry/instrument/detector/data", data=frames, chunks=(1, ROWS, COLS))
        f["entry/data/data"] = d
    described = n - 1 if short_record else n      # the record one frame short of the frame file
    master = root / f"{number}.nxs"
    with h5py.File(master, "w") as f:
        e = f.create_group("entry1")
        e["title"] = title
        e["start_time"] = "2024-07-15T14:54:35.408+01"
        e["end_time"] = "2024-07-15T14:58:07.131+01"
        e["scan_dimensions"] = np.array([n], np.int32)
        inst = e.create_group("instrument")
        inst["image_key/image_key"] = np.array(key[:described], float)
        inst["zebraSM1/zebraSM1"] = angle[:described]
        inst["zebraContinuousMoveController/CaptureTime"] = capture[:described]
        inst["flyScanDetector/count_time"] = np.full(described, EXPOSURE)
        inst["flyScanDetector/data"] = h5py.ExternalLink(f"pco1-{number}.hdf", "/entry/instrument/detector/data")
        current = inst.create_dataset("source/current", data=250.1214)
        current.attrs["units"] = "mA"
        t = e.create_group("tomo_entry")
        t["definition"] = "NXtomo"
        t["instrument/detector/data"] = h5py.ExternalLink(f"pco1-{number}.hdf", "/entry/instrument/detector/data")
        t["instrument/detector/image_key"] = inst["image_key/image_key"]
        for name in ("x_pixel_size", "y_pixel_size", "distance"):
            t.create_dataset("instrument/detector/" + name, data=np.nan).attrs["units"] = "undefined"
        t["sample/rotation_angle"] = inst["zebraSM1/zebraSM1"]
        t["control/data"] = current
        b = e.create_group("before_scan")
        for name, value in dict(ss1_x=in_beam if snapshot_x is None else snapshot_x, ss1_y1=height, ss1_y2=10.4861, ss1_tx=-0.4, ss1_tz=0.2, ss1_theta=366.275).items():
            b["ss1/" + name] = value
        b["ring_current"] = np.array([251.5])
        b["flyscan_cfg/tomography_translation"] = "ss1_x"
        b["flyscan_cfg/tomography_flyscan_theta"] = "zebraSM1"
        b["flyscan_cfg/tomography_cmd"] = (f"i12tomoFlyScan(closeShutterAfterScan=False, description={title}, exposureTime={EXPOSURE}, helical_axis_stage=None, "
                                           f"imagesPerDark={N_DARK}, imagesPerFlat={N_FLAT}, inBeamPosition={in_beam:g}, outOfBeamPosition=-100, start=0, step={STEP}, stop=360)")
    if supplement is not None:
        (root / "vc_tomography_acquisition.json").write_text(json.dumps(supplement))
    return master, angle[proj], frames, proj


def test_open_scan(tmp_path):
    master, angle, frames, proj = _write(tmp_path, 146083, "PHerc.172_grid_01_53keV_r2c2", in_beam=-26.0, snapshot_x=-99.285, height=38.5)
    s = bliss.open_scan(str(master))
    assert s.master == str(master) and s.name == "146083" and s.sample_name == "PHerc.172_grid_01_53keV_r2c2"
    assert s.frame_shape == (ROWS, COLS) and s.dtype == "uint16" and s.detector == "flyScanDetector"
    assert s.n_proj == N_PROJ and len(s.darks) == N_DARK and [len(f) for f in s.flats] == [N_FLAT]
    assert list(s.entries) == ["darks", "flats", "projections"] and s.entries["projections"] == dict(role="proj", title=s.sample_name, n_frames=N_PROJ)

    # the frames: one file, the runs of image_key one after the other
    p = s.projections
    assert p.files == [str(tmp_path / "pco1-146083.hdf")] and p.dataset == "/entry/instrument/detector/data"
    np.testing.assert_array_equal(p.frame_index, proj)
    np.testing.assert_array_equal(s.darks.frame_index, np.arange(N_DARK))
    np.testing.assert_array_equal(s.flats[0].frame_index, N_DARK + np.arange(N_FLAT))
    got = _core.RawReader(list(p.files), p.dataset, ROWS, COLS).read_frames(p.file_index, p.frame_index)
    np.testing.assert_array_equal(got, frames[proj])
    s.check_files()

    # the angle at the middle of the exposure: the captured one plus half the arc swept (median step x exposure / period)
    arc = EXPOSURE / (PERIOD_MS * 1e-3)
    assert s.exposure_arc == arc and s.count_time_s == EXPOSURE
    np.testing.assert_array_equal(s.angles_deg, angle + 0.5 * arc * np.median(np.diff(angle)))
    np.testing.assert_array_equal(s.unwrapped_angles_deg, s.angles_deg)
    np.testing.assert_array_equal(s.flats_unwrapped_angles_deg, s.angles_deg[:1])

    # the stages: across the beam the command's position, not the snapshot's; the height stage's snapshot
    assert s.y_translation_m == -26.0 * 1e-3 and s.estimated_cor_from_motor_pix == s.y_translation_m / s.pixel_size_m
    np.testing.assert_array_equal(s.z_translation_m, np.full(N_PROJ, 38.5 * 1e-3))
    # one ring current for every frame of the scan; what the supplement gives and what nothing gives
    for c, n in ((s.control_A, N_PROJ), (s.darks_control_A, N_DARK), (s.flats_control_A[0], N_FLAT)):
        np.testing.assert_array_equal(c, np.full(n, 250.1214 * 1e-3))
    assert s.pixel_size_m == 7.91 * 1e-6 and s.energy_kev == 53.0 and np.isnan(s.distance_m)


def test_a_record_shorter_than_its_frame_file_and_flats_at_both_ends(tmp_path):
    master, angle, frames, proj = _write(tmp_path, 146080, "sample_r1c2", flats_at_end=3, short_record=True)
    s = bliss.open_scan(str(master))
    assert s.n_proj == N_PROJ and [len(f) for f in s.flats] == [N_FLAT, 2]        # the last flat frame is not described
    assert list(s.entries) == ["darks", "flats", "projections", "flats2"]
    np.testing.assert_array_equal(s.flats[1].frame_index, proj[-1] + 1 + np.arange(2))
    np.testing.assert_array_equal(s.flats_unwrapped_angles_deg, s.unwrapped_angles_deg[[0, -1]])
    assert s.unwrapped_angles_deg[-1] > 360                                        # a turn and a frame, not wrapped


def test_the_pixel_size_must_be_given(tmp_path):
    master, *_ = _write(tmp_path, 146079, "sample_r1c1", supplement=None)
    with pytest.raises(ValueError, match="states no pixel size.*vc_tomography_acquisition.json"):
        bliss.open_scan(str(master))
    assert bliss.discover_scans(str(tmp_path)) == []                               # left out, as any master that cannot be read
    (tmp_path / "vc_tomography_acquisition.json").write_text('{"pixel_size_um": 3.24, "source_sample_distance_m": 51, "propagation_distance_m": 0.5}')
    s = bliss.open_scan(str(master))
    assert s.pixel_size_m == 3.24 * 1e-6 and s.distance_m == 0.5 and np.isnan(s.energy_kev)


def test_discover_a_folder_of_masters(tmp_path):
    for number, title, x, y in ((146080, "s_r1c2", -26.0, 48.5), (146079, "s_r1c1", -8.0, 48.5), (146082, "s_r2c1", -8.0, 38.5)):
        _write(tmp_path / "visit", number, title, in_beam=x, height=y)
    scans = bliss.discover_scans(str(tmp_path))
    assert [s.name for s in scans] == ["146079", "146080", "146082"]
    assert [round(s.y_translation_m * 1e3) for s in scans] == [-8, -26, -8] and [s.z_translation_m[0] * 1e3 for s in scans] == [48.5, 48.5, 38.5]


def test_the_scans_of_a_grid_are_one_family_row_by_row(tmp_path):
    """Circular scans at two heights and three places across the beam, one of them a frame short: one family, row
    by row in the order of acquisition, each row from the scan with the axis in its field outwards."""
    places = {146079: (-44.0, 48.5), 146080: (-8.0, 48.5), 146081: (-26.0, 48.5), 146082: (-26.0, 38.5), 146083: (-44.0, 38.5), 146084: (-8.0, 38.5)}
    for number, (x, height) in places.items():
        _write(tmp_path, number, f"grid_{number}", in_beam=x, height=height, short_record=number == 146081)
    scans = bliss.discover_scans(str(tmp_path))
    assert [s.name for s in scans] == [str(n) for n in sorted(places)]
    assert len(scans[2].unwrapped_angles_deg) == N_PROJ - 1
    families = bliss.group_companions(scans)
    assert len(families) == 1
    assert [s.name for s in families[0]] == ["146080", "146081", "146079", "146084", "146082", "146083"]
    family = bliss.pair_companions(scans)
    assert family["main"].name == "146080" and [s.name for s in family["companions"]] == ["146081", "146079", "146084", "146082", "146083"]
    # scans at one place across the beam and two heights are a family too (a column of the grid)
    column = [s for s in scans if s.name in ("146080", "146084")]
    assert [s.name for s in bliss.group_companions(column)[0]] == ["146080", "146084"]
    # two scans at one place are not
    again = tmp_path / "again"
    _write(again, 146090, "grid_again", in_beam=-8.0, height=48.5)
    assert bliss.group_companions([scans[1], bliss.open_scan(str(again / "146090.nxs"))]) == []


def test_the_intensity_calibration_of_a_grid(tmp_path):
    """Six circular scans, the rotation axis in the left half of the first and the other columns on its right, a
    second row above the first; slits light the rows 1 to 4. The weights are cropped to the lit rows; a scan above
    the main scan is apodised as the main scan; the columns taper towards their neighbours, mirrored against the
    reference's layout (the axis in the right half, the companions on the left)."""
    from vc_tomography import calibration
    places = {146079: (-0.010, 48.5), 146080: (-0.060, 48.5), 146081: (-0.110, 48.5), 146082: (-0.010, 38.5), 146083: (-0.060, 38.5), 146084: (-0.110, 38.5)}
    for number, (x, height) in places.items():
        _write(tmp_path, number, f"grid_{number}", in_beam=x, height=height, lit=(1, 5))
    family = bliss.pair_companions(bliss.discover_scans(str(tmp_path)))
    scans = [family["main"], *family["companions"]]
    assert [s.name for s in scans] == [str(n) for n in places]
    layout = dict(cor_main_detector=2.25, vectors=[(0.0, 6.0), (0.5, 12.0), (3.0, 0.25), (3.0, 6.25), (3.5, 12.0)])
    lines = []
    cal, arrays = calibration.calibrate(scans, layout, log=lines.append)
    assert cal.lit_rows == (1, 5) and any("the beam lights rows 1..4 of 6" in line for line in lines)
    flat = arrays["main_flats"].mean(0) - arrays["main_dark"]
    assert calibration.lit_rows(flat) == (1, 5) and calibration.lit_rows(np.full((ROWS, COLS), 7.0, np.float32)) == (0, ROWS)
    stacks = [arrays[f"{tag}_flats"] for tag in ("companion5", "companion4", "companion3", "companion2", "companion", "main")]
    np.testing.assert_array_equal(arrays["weight_map"], calibration.weight_map(stacks, crop_top=1, crop_bottom=1))
    assert not np.array_equal(arrays["weight_map"], calibration.weight_map(stacks))
    # the merged frame: the first column at column 0, the others on its right
    assert cal.merged_width == COLS + 12 and cal.rot_center == 2.25
    assert [cal.placement[s.name].shift_x for s in scans] == [0, 6, 12, 0, 6, 12] and cal.placement["146082"].z_readjustment == 3.0
    # the apodisation: the main scan's slope beyond the axis on its left and a taper towards its neighbour; the same
    # for the scan above it; the others taper towards the main scan, and on their far side if a scan lies beyond
    main_left = (COLS - 1) / 2.0 + scans[0].estimated_cor_from_motor_pix + 1.0
    widths = cal.apodisation
    assert widths["146079"] == widths["146082"] == (main_left, 50.0) and 0.0 < main_left < COLS / 2
    assert widths["146080"] == widths["146083"] == (50.0, 50.0) and widths["146081"] == widths["146084"] == (50.0, 0.0)
    assert calibration.apodisation_widths(COLS, scans[0].estimated_cor_from_motor_pix, 6.0) == ((main_left, 50.0), (50.0, 0.0))
    np.testing.assert_array_equal(arrays["companion3_weight"], arrays["weight_map"] * _core.apodisation_mask(COLS, main_left, 50.0, 1e-8)[None, :])
    # the reference's layout mirrored: the axis must lie in the half away from the other scans
    with pytest.raises(NotImplementedError, match="half away from the companion"):
        calibration.apodisation_widths(COLS, scans[0].estimated_cor_from_motor_pix, -6.0)
    # scans above each other alone: no apodisation across the beam
    column, _ = calibration.calibrate([scans[0], scans[3]], dict(cor_main_detector=2.25, vectors=[(3.0, 0.25)]), log=lines.append)
    assert column.apodisation == {"146079": (0.0, 0.0), "146082": (0.0, 0.0)}


def test_the_record_of_a_calibration_session(tmp_path):
    master, angle, frames, proj = _write(tmp_path, 146083, "PHerc.172_grid_01_53keV_r2c2", in_beam=-26.0, snapshot_x=-99.285, height=38.5, short_record=True)
    (tmp_path / "vc_tomography_acquisition.json").write_text('{"pixel_size_um": 7.91, "source_sample_distance_m": 51, "fan_sense": 2}')
    with pytest.raises(ValueError, match="'fan_sense' is 1 or -1"):
        _core.CalibrationSession([str(master)], 2)
    (tmp_path / "vc_tomography_acquisition.json").write_text('{"pixel_size_um": 7.91, "source_sample_distance_m": 51, "fan_sense": -1}')
    sess = _core.CalibrationSession([str(master)], 2)
    info = sess.info(0)
    assert info["label"] == "r2c2" and (info["rows"], info["cols"]) == (ROWS, COLS) and info["pixel_um"] == 7.91 and info["frames_per_turn"] == 72
    assert (info["rotation"], info["translation_y"], info["translation_z"], info["sample_u"], info["sample_v"]) == ("zebraSM1", "ss1_x", "ss1_y1", "ss1_tx", "ss1_tz")
    assert sess.number(0, "scan/source_sample_distance") == 51e3                   # mm, as a BLISS record states it
    assert sess.number(0, "scan/fan_sense") == -1.0                                # the divergence in the other sense than BM18's
    st = sess.stretches(0)
    assert [(s["entry"], s["role"], s["n_frames"], s["latched"], s["helical"]) for s in st] == [
        ("darks", "dark", N_DARK, False, False), ("flats", "flat", N_FLAT, False, False), ("projections", "projection", N_PROJ - 1, True, False)]
    assert all(s["on_disk"] == [(0, s["n_frames"])] for s in st)
    arc = np.median(np.diff(angle)) * EXPOSURE / (PERIOD_MS * 1e-3)
    np.testing.assert_array_equal(sess.series(0, 2, "angle_deg"), angle[:-1] + 0.5 * arc)
    np.testing.assert_array_equal(sess.series(0, 2, "height_mm"), np.full(N_PROJ - 1, 38.5))
    np.testing.assert_array_equal(sess.series(0, 2, "timer_period"), np.full(N_PROJ - 1, PERIOD_MS * 1e-3))
    # the translation stage: in the beam for the projections, out of it for the flats, whatever the snapshot says
    assert sess.positioner(0, 2, "ss1_x") == -26.0 and sess.positioner(0, 1, "ss1_x") == -100.0 and sess.positioner(0, 2, "ss1_y1") == 38.5
    # frames through the record: the stretch's own numbering onto the one dataset
    np.testing.assert_array_equal(sess.read_rows(0, 2, np.array([0, 5], np.int64), 1, 2), frames[proj[[0, 5]], 1:3])
    np.testing.assert_array_equal(sess.dark(0), np.full((ROWS, COLS), 100, np.float32))
    np.testing.assert_array_equal(sess.gain(0), np.full((ROWS, COLS), 3900, np.float32))
