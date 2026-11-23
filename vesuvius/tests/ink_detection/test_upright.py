import numpy as np
import tifffile
import zarr
from scipy import ndimage as ndi
from vc_reader import images

from vesuvius.ink_detection.inference import upright


def _tilted_segment(tmp_path, tilt_deg, mirror=False):
    """A plane whose z increases along a direction `tilt_deg` away from the grid rows."""
    h, w = 60, 80
    r, c = np.mgrid[0:h, 0:w].astype(np.float64)
    t = np.deg2rad(tilt_deg)
    z = 1000.0 + 20.0 * (r * np.cos(t) + c * np.sin(t))
    if mirror:
        z = z[:, ::-1]
    z[:3, :3] = -1.0                                  # a few invalid points, as in real grids
    seg = tmp_path / "seg.tifxyz"
    seg.mkdir()
    for name, arr in (("x", c * 20.0), ("y", r * 20.0), ("z", z)):
        tifffile.imwrite(seg / f"{name}.tif", arr.astype(np.float32))
    return seg


def test_angle_puts_z_down_the_rows(tmp_path):
    for tilt in (0.0, 40.0, -125.0):
        for mirror in (False, True):
            case = tmp_path / f"{tilt}_{mirror}"
            case.mkdir()
            seg = _tilted_segment(case, tilt, mirror)
            angle, down, side = upright.find_angle(*upright.read_z(seg))
            assert down > 15.0, (tilt, mirror, angle, down)
            assert side < 1.5, (tilt, mirror, angle, side)


def test_rotate_then_back_restores_the_canvas(tmp_path):
    seg = _tilted_segment(tmp_path, 40.0)
    rng = np.random.default_rng(0)
    layers = ndi.gaussian_filter(rng.random((4, 120, 160)), (0, 3, 3))
    layers = (255 * (layers - layers.min()) / np.ptp(layers)).astype(np.uint8)
    render = tmp_path / "render.zarr"
    zarr.open_array(str(render), mode="w", shape=layers.shape, chunks=(4, 64, 64), dtype=np.uint8)[:] = layers
    up = tmp_path / "up.zarr"
    meta = upright.rotate_volume(seg, render, up)
    assert meta["orig_shape"] == [120, 160]
    # stand-in for a prediction on the upright volume: its middle layer
    pred = tmp_path / "pred_up.tif"
    tifffile.imwrite(pred, np.asarray(zarr.open_array(str(up), mode="r")[2]))
    back = tmp_path / "pred.tif"
    assert upright.rotate_back(up, pred, back) == (120, 160)
    restored = np.squeeze(np.asarray(images.read(back))).astype(np.float64)
    inner = (slice(25, 95), slice(35, 125))           # away from the corners that rotate off the canvas
    r = np.corrcoef(restored[inner].ravel(), layers[2][inner].astype(np.float64).ravel())[0, 1]
    assert r > 0.98
