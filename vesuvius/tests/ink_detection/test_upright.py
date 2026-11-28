import numpy as np
import tifffile
from scipy import ndimage as ndi
from vc_reader import images
from vc_reader.reader import ZarrWriter, open_node

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
    (seg / "meta.json").write_text('{"scale": [0.05, 0.05]}')
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
    r, c = np.mgrid[0:120, 0:160].astype(np.float32)
    layers = np.stack([r, c, 0.7 * r + 1.3 * c, r - c])   # linear ramps survive bilinear resampling exactly
    render = tmp_path / "render.zarr"
    ZarrWriter.create(render, shape=layers.shape, chunks=(4, 64, 64), dtype=np.float32, auth=None)[:] = layers
    up = tmp_path / "up.zarr"
    meta = upright.rotate_volume(seg, render, up)
    assert meta["orig_shape"] == [120, 160]
    # stand-in for a prediction on the upright volume: its middle layer
    pred = tmp_path / "pred_up.tif"
    tifffile.imwrite(pred, np.asarray(open_node(str(up))[2]))
    back = tmp_path / "pred.tif"
    assert upright.rotate_back(up, pred, back) == (120, 160)
    restored = np.squeeze(np.asarray(images.read(back)))
    inner = (slice(25, 95), slice(35, 125))           # away from the corners that rotate off the canvas
    np.testing.assert_allclose(restored[inner], layers[2][inner], atol=1e-3)


def test_back_is_exact_at_quarter_turns(tmp_path):
    img = np.random.default_rng(0).integers(1, 255, (7, 9)).astype(np.uint8)
    for angle in (90.0, -90.0, 180.0):
        rot = ndi.rotate(img, angle, reshape=True, order=1)
        up = tmp_path / f"up{angle}.zarr"
        dst = ZarrWriter.create(up, shape=(1,) + rot.shape, chunks=(1, 16, 16), dtype=np.uint8, auth=None)
        dst.attrs[upright.ATTR] = {"angle_deg": angle, "orig_shape": list(img.shape), "rot_shape": list(rot.shape)}
        tifffile.imwrite(tmp_path / "pred_up.tif", rot)
        upright.rotate_back(up, tmp_path / "pred_up.tif", tmp_path / "pred.tif")
        np.testing.assert_array_equal(np.squeeze(np.asarray(images.read(tmp_path / "pred.tif"))), img)
