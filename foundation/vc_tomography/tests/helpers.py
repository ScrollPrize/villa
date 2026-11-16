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
