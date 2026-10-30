"""vesuvius.preview: slices, sweeps, projections, orbits, overlays and comparisons of a volume."""

import math

import numpy as np
import pytest
from PIL import Image

from vesuvius.data.utils import open_zarr, open_zarr_group
from vesuvius.tools import preview as pv


def _ome(tmp_path, data, name="vol.zarr", levels=2):
    """A small OME-Zarr group with a 2x pyramid, like the published volumes."""
    path = str(tmp_path / name)
    root = open_zarr_group(path, mode="w")
    level = data
    for i in range(levels):
        arr = open_zarr(f"{path}/{i}", mode="w", shape=level.shape, chunks=(8, 8, 8), dtype=level.dtype)
        arr[...] = level
        level = level[::2, ::2, ::2]
    root.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [{"name": a, "type": "space"} for a in "zyx"],
        "datasets": [{"path": str(i), "coordinateTransformations": [
            {"type": "scale", "scale": [2.0 ** i] * 3}]} for i in range(levels)],
    }]
    return path


def _ramp(shape=(16, 32, 24)):
    z, y, x = np.indices(shape)
    return (z * 8 + y * 2 + x).astype(np.uint8)


def test_parse_crop_fills_missing_bounds_and_checks_ranges():
    assert pv.parse_crop(None, (10, 20, 30)) == [(0, 10), (0, 20), (0, 30)]
    assert pv.parse_crop("2:4,:,5:", (10, 20, 30)) == [(2, 4), (0, 20), (5, 30)]
    with pytest.raises(SystemExit, match="three ranges"):
        pv.parse_crop("1:2,3:4", (10, 20, 30))
    with pytest.raises(SystemExit, match="outside"):
        pv.parse_crop("0:11,:,:", (10, 20, 30))


def test_pick_level_takes_the_finest_level_that_fits():
    source = pv.Source("s", [np.zeros((64, 512, 512)), np.zeros((32, 256, 256)), np.zeros((16, 128, 128))])
    crop = [(0, 64), (0, 512), (0, 512)]
    assert pv.pick_level(source, crop, 1024) == 0
    assert pv.pick_level(source, crop, 256) == 1
    assert pv.pick_level(source, crop, 10) == 2
    # A thin crop drawn along z only counts its y and x sides.
    assert pv.pick_level(source, [(0, 64), (0, 200), (0, 200)], 256, axis=0) == 0


def test_contrast_leaves_the_zero_background_out():
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 50:] = np.linspace(100, 200, 50, dtype=np.uint8)
    lo, hi = pv.contrast([img])
    assert lo >= 100 and hi <= 200
    assert pv.contrast([img], (0, 10)) == (0.0, 10.0)


def test_orbit_keeps_a_central_voxel_in_the_middle_at_every_angle():
    block = np.zeros((4, 21, 21), dtype=np.uint8)
    block[:, 10, 10] = 255
    for angle in np.linspace(0, math.pi, 7):
        out = pv.orbit_projection(block, angle, "max")
        assert out.shape[0] == 4
        assert abs(int(out[0].argmax()) - (out.shape[1] - 1) / 2) <= 1
        assert (pv.orbit_projection(block, angle, "mean") <= out).all()


def test_slice_png_matches_the_volume(tmp_path):
    data = _ramp()
    src = _ome(tmp_path, data)
    out = tmp_path / "z.png"
    assert pv.main([src, "-o", str(out), "--at", "5", "--range", "0", "255", "--no-labels"]) == 0
    img = np.asarray(Image.open(out))
    np.testing.assert_array_equal(img, pv.to_u8(data[5], 0, 255))


def test_views_sweep_and_compare(tmp_path):
    src = _ome(tmp_path, _ramp())
    three = tmp_path / "three.png"
    pv.main([src, "-o", str(three), "--views", "zyx", "--no-labels"])
    # z (32x24), y (16x24) and x (16x32) panels side by side with 6 px gaps.
    assert Image.open(three).size == (24 + 6 + 24 + 6 + 32, 32)

    sweep = tmp_path / "sweep.gif"
    pv.main([src, "-o", str(sweep), "--sweep", "5"])
    assert Image.open(sweep).n_frames == 5

    other = _ome(tmp_path, 255 - _ramp(), name="other.zarr")
    both = tmp_path / "both.png"
    pv.main([src, other, "-o", str(both), "--no-labels"])
    assert Image.open(both).size == (24 + 6 + 24, 32)


def test_mip_and_orbit(tmp_path):
    data = np.zeros((8, 16, 16), dtype=np.uint8)
    data[3, 4, 5] = 200
    src = _ome(tmp_path, data, levels=1)
    mip = tmp_path / "mip.png"
    pv.main([src, "-o", str(mip), "--mip", "--range", "0", "200", "--no-labels"])
    img = np.asarray(Image.open(mip))
    assert img[4, 5] == 255 and img.sum() == 255

    orbit = tmp_path / "orbit.gif"
    pv.main([src, "-o", str(orbit), "--orbit", "6", "--project", "mean"])
    assert Image.open(orbit).n_frames == 6


def test_overlay_draws_the_mask_in_red(tmp_path):
    data = np.full((8, 16, 16), 100, dtype=np.uint8)
    mask = np.zeros_like(data)
    mask[:, :8] = 1
    src = _ome(tmp_path, data, levels=1)
    msk = _ome(tmp_path, mask, name="mask.zarr", levels=1)
    out = tmp_path / "overlay.png"
    pv.main([src, "-o", str(out), "--overlay", msk, "--no-labels"])
    img = np.asarray(Image.open(out).convert("RGB")).astype(int)
    assert img[2, 2, 0] > img[2, 2, 1]          # masked: red
    assert img[12, 2, 0] == img[12, 2, 1]       # unmasked: gray


def test_overlay_of_another_shape_is_refused(tmp_path):
    src = _ome(tmp_path, _ramp(), levels=1)
    msk = _ome(tmp_path, np.zeros((4, 4, 4), np.uint8), name="mask.zarr", levels=1)
    with pytest.raises(SystemExit, match="unlike every input"):
        pv.main([src, "-o", str(tmp_path / "x.png"), "--overlay", msk])


def test_several_frames_need_a_gif(tmp_path):
    src = _ome(tmp_path, _ramp(), levels=1)
    with pytest.raises(SystemExit, match="gif"):
        pv.main([src, "-o", str(tmp_path / "x.png"), "--sweep", "3"])


def test_options_reject_unknown_names():
    with pytest.raises(TypeError, match="colour"):
        pv.options(colour="red")


def test_volume_preview_and_repr(tmp_path):
    from vesuvius import Volume

    vol = Volume(type="zarr", path=_ome(tmp_path, _ramp()))
    assert "2 level(s), level 0 (16, 32, 24) uint8" in repr(vol)
    img = vol.preview(at=3, no_labels=True, range=(0, 255))
    np.testing.assert_array_equal(np.asarray(img), pv.to_u8(_ramp()[3], 0, 255))
    frames = vol.preview(sweep=4)
    assert len(frames) == 4
