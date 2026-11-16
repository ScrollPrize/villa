"""Helpers shared by the test modules (tests/ is on pytest's pythonpath)."""
import pytest

import vc_tomography

cuda_only = pytest.mark.skipif(not vc_tomography.cuda_available(), reason="no CUDA")


def find_tool(name):
    """The path of one of vc_tomography's executables: beside the built _core module (or up to two folders above
    it), else on PATH; None when it is not built."""
    import shutil
    from pathlib import Path

    from vc_tomography import _core
    here = Path(_core.__file__).resolve().parent
    found = next((d / name for d in (here, *here.parents[:2]) if (d / name).exists()), None)
    return str(found) if found else shutil.which(name)


def raw_scan(path, raw, angles_deg, z_m=None, *, shift_x=0, x_readjustment=0.0, z_readjustment=0.0, control_A=None,
             dark=0.0, flats=50000.0, weight=None, pixel_size_m=0.01, **terms):
    """A ScanData (_core.make_scan_data) of the frames `raw` (n, rows, cols) written to the HDF5 file `path` (dataset
    "data", one chunk per frame), or of the frames already there when `raw` is None. Every frame is its own projection
    (file index 0, frame index k) at 0.2 A unless `control_A` says otherwise; the dark (a value or a frame) and one
    flat series `flats` (a value, a frame or (1, rows, cols)) at angle 0; weight: ones. `terms`: the rest of
    make_scan_data's keywords (registration terms, map, lit rows)."""
    import h5py
    import numpy as np

    from vc_tomography import _core
    if raw is not None:
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=np.ascontiguousarray(raw), chunks=(1, *raw.shape[1:]))
    with h5py.File(path, "r") as f:
        nr, nd = f["data"].shape[1:]
    angles = np.asarray(angles_deg, np.float64)
    n = len(angles)
    z = np.zeros(n) if z_m is None else np.asarray(z_m, np.float64)
    frame = lambda v: np.broadcast_to(np.asarray(v, np.float32), (nr, nd)).copy()
    flats = np.asarray(flats, np.float32)
    flats = np.broadcast_to(flats, (1, nr, nd)).copy() if flats.ndim < 3 else flats
    return _core.make_scan_data(angles, z, pixel_size_m, nr, nd, shift_x, x_readjustment, z_readjustment,
                                np.full(n, 0.2) if control_A is None else np.asarray(control_A, np.float64),
                                np.array([0], np.float32), frame(dark), flats, frame(1.0) if weight is None else weight,
                                _core.RawReader([str(path)], "data", nr, nd), np.zeros(n, np.int32), np.arange(n, dtype=np.int32),
                                **terms)


def joint_calibration(masters, axes, heights, *, rows, cols, frames_per_turn, pixel_um, first=0, end=None, frame_period_s=0.03):
    """A geometry calibration in the step's own form ("model": "joint"; vc_tomography_calibration geometry) of scans
    that each recorded one helix of len(heights) frames, of which [first, end) were on disk: scan k (its master
    masters[k], named by the last "_" part before the scan number) has its rotation axis at the column axes[k], and nothing else
    (no map, roll, stage motion, rows' timing or place)."""
    import numpy as np

    from vc_tomography import geometry as G
    n = len(heights)
    end = n if end is None else end
    stretch = dict(name="helix", index=0, helix=True, frames=n, first=first, end=end, offset=0)
    mid = float(np.asarray(heights)[(first + end) // 2])
    names = [str(m).rsplit("/", 1)[-1].removesuffix(".h5").rsplit("_", 1)[0].split("_")[-1] for m in masters]   # s_B_HA_0001.h5: HA
    return dict(step="geometry", model="joint", scans=[str(m) for m in masters],
                layout=dict(rows=rows, cols=cols, frames_per_turn=frames_per_turn, pixel_um=pixel_um, row_sign=-1.0, anchor=0,
                            scans=[dict(name=name, mid_height=mid, coarse_axis=a, stretches=[stretch]) for name, a in zip(names, axes)]),
                row_time=dict(seconds_per_row=0.0, frame_period_s=frame_period_s, frames_per_row=0.0, measured=None), map=G.empty_map(rows, cols),
                parameters=dict(roll=0.0, pitch=0.0, stage=dict(height_once=[0.0, 0.0], axis_twice=[0.0, 0.0]), about_height=0.0,
                                scans=[dict(name=name, axis=a, axis_slope=0.0, axial_scale=0.0, column_scale=0.0, d=[0.0, 0.0], height=0.0,
                                            stretch_height=dict(helix=0.0), mid_height=mid, height_once=[0.0, 0.0], rotation=[0.0, 0.0, 0.0])
                                        for name, a in zip(names, axes)]))
