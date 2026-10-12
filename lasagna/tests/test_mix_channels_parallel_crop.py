"""mix_channels_parallel.py must not move voxels when the inputs' crop starts
inside a 32^3 chunk: the cropped part of each edge chunk used to be padded at
the chunk origin, shifting the data by (crop start mod 32) in the chunks at
the crop start."""
import os
import subprocess
import sys

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
if int(zarr.__version__.split(".")[0]) < 3:
    pytest.skip("mix_channels_parallel.py uses the zarr 3 API", allow_module_level=True)

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mix_channels_parallel.py")
SHAPE = (4, 64, 64, 64)
CHANNELS = ["cos", "grad_mag", "nx", "ny"]


def _inputs(tmp_path, crop):
    rng = np.random.default_rng(0)
    data = {}
    for name in ("base", "other"):
        arr = rng.integers(1, 256, SHAPE, dtype=np.uint8)
        z = zarr.open(str(tmp_path / f"{name}.zarr"), mode="w", shape=SHAPE, chunks=(1, 32, 32, 32),
                      dtype=np.uint8, zarr_format=2, fill_value=0)
        z[:] = arr
        z.attrs["preprocess_params"] = {"scaledown": 1, "channels": CHANNELS, "crop_xyzwhd": crop}
        data[name] = arr
    return data


def _mix(tmp_path):
    out = tmp_path / "out.zarr"
    subprocess.run([sys.executable, SCRIPT, "--base", str(tmp_path / "base.zarr"), "--other", str(tmp_path / "other.zarr"),
                    "--output", str(out), "--channels-other", "nx", "--workers", "1"],
                   check=True, stdout=subprocess.DEVNULL)
    return zarr.open(str(out), mode="r")[:]


@pytest.mark.parametrize("crop", [[10, 0, 0, 40, 64, 64], [5, 7, 3, 50, 40, 45], [0, 0, 0, 64, 64, 64]])
def test_output_matches_inputs_inside_the_crop(tmp_path, crop):
    data = _inputs(tmp_path, crop)
    out = _mix(tmp_path)
    x, y, z, w, h, d = crop
    sl = (slice(z, z + d), slice(y, y + h), slice(x, x + w))
    for ci, name in enumerate(CHANNELS):
        src = data["other"] if name == "nx" else data["base"]
        np.testing.assert_array_equal(out[ci][sl], src[ci][sl], err_msg=name)
        outside = np.ones(SHAPE[1:], bool)
        outside[sl] = False
        assert not out[ci][outside].any(), name
