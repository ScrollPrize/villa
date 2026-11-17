"""The device scratch the CUDA kernels keep between calls (thread_local buffers and FFT plans) does not outlive a
reconstructor: release_device_scratch() frees it, and a CUDA reconstructor's end does so. Measured as the device's
free memory (cudaMemGetInfo) around a median filter whose scratch is 256 MB, with wide margins: the device may be
shared."""
import ctypes
import gc
import json

import numpy as np
import pytest

import vc_tomography
from vc_tomography.recon import HelicalRecon

from helpers import cuda_only
from test_reconstruct_scans_side_by_side import _geometry, acquisition  # noqa: F401 (a fixture)

MB = 1 << 20
FRAMES = np.zeros((64, 1024, 1024), np.float32)       # the median's scratch: 256 MB


def free_bytes():
    rt = ctypes.CDLL("libcudart.so.13")                # (the runtime vc_tomography already loaded: the same context)
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    assert rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) == 0
    return free.value


def fill_scratch():
    """the median's scratch made 256 MB on this thread; the device's free memory before and after"""
    vc_tomography.apply_filters(FRAMES[:1, :8, :8], [{"type": "median", "radius": 1}], 0, device="cuda")  # (modules loaded)
    vc_tomography.release_device_scratch()
    before = free_bytes()
    vc_tomography.apply_filters(FRAMES, [{"type": "median", "radius": 1}], 0, device="cuda")
    after = free_bytes()
    assert before - after > 200 * MB, (before - after) / MB
    return before, after


@cuda_only
def test_release_device_scratch():
    before, after = fill_scratch()
    vc_tomography.release_device_scratch()
    assert free_bytes() - after > 200 * MB
    out = vc_tomography.apply_filters(FRAMES[:2, :64, :64] + 1, [{"type": "median", "radius": 1}], 0, device="cuda")
    assert (out == 1).all()                            # (the scratch is made again)


@cuda_only
def test_a_reconstructors_end_releases_it(tmp_path, acquisition):  # noqa: F811
    root, masters, n = acquisition
    path = tmp_path / "geometry.json"
    path.write_text(json.dumps(_geometry(masters, n), default=float))

    def reconstruct():
        r = HelicalRecon(root / "projections", cache_dir=str(tmp_path / "cache"), device="cuda",
                         params=dict(geometry=str(path), sharpness_weighting="off"), log=lambda *a: None)
        h = (r.lo + r.hi) // 2
        assert np.isfinite(r.reconstruct_heights(h, h + 1)).all()
        del r
        gc.collect()                                   # (HelicalRecon holds a reference cycle)

    reconstruct()          # (the first reconstruction of a process also loads its kernels' code: ~180 MB, kept)
    before, after = fill_scratch()
    reconstruct()
    assert free_bytes() - after > 200 * MB
