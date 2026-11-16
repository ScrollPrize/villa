import os

import numpy as np
import pytest

from vc_tomography.omezarr import OmeZarrWriter, to_uint8


@pytest.mark.parametrize("compression,dtype", [("none", np.float32), ("zlib", np.float32), ("none", np.uint8),
                                               ("vc-delta3d", np.uint8)])
def test_roundtrip_any_slab_order(tmp_path, compression, dtype):
    if compression == "vc-delta3d":
        from vc_tomography import _core
        if not _core.delta3d_available():
            pytest.skip("built without vc-delta3d")
    rng = np.random.default_rng(0)
    vol = (rng.random((37, 45, 61)) * 200).astype(dtype)
    vol[:, :, :32] = 0                                     # empty chunks: must not be written
    w = OmeZarrWriter(str(tmp_path / "v.zarr"), vol.shape, dtype, chunks=(8, 16, 32), voxel_size_um=2.0,
                      origin_zyx=(10, 20, 30), levels=3, compression=compression)
    for z0, z1 in [(16, 30), (0, 5), (30, 37), (5, 16)]:   # out of order, unaligned slabs
        w.write(z0, vol[z0:z1])
    w.close()
    np.testing.assert_array_equal(w.read(0, 0, 37), vol)
    assert w.read(1, 0, w.level_shapes[1][0]).shape == (19, 23, 31)
    assert w.n_skipped > 0 and not os.path.exists(tmp_path / "v.zarr" / "0" / "0" / "0" / "0")
    if compression == "vc-delta3d":
        return  # zarr-python has no vc-delta3d codec registered
    zarr = pytest.importorskip("zarr")
    g = zarr.open_group(str(tmp_path / "v.zarr"), mode="r")
    np.testing.assert_array_equal(g["0"][:], vol)
    ms = g.attrs["multiscales"][0]
    assert ms["datasets"][0]["coordinateTransformations"][1]["translation"] == [20.0, 40.0, 60.0]
    assert len(ms["datasets"]) == 3


def test_to_uint8_window():
    np.testing.assert_array_equal(to_uint8(np.array([-1, -0.03, 0.145, 1.0], np.float32)), [0, 0, 255, 255])


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_pyramid_level_is_the_mean_of_the_voxels_inside(tmp_path, dtype):
    """level 1 voxel = the mean of its 2 x 2 x 2 block's voxels that lie in level 0 (an odd edge averages fewer),
    uint8 rounded half to even (vcr::zarr::writePyramidLevel)"""
    rng = np.random.default_rng(4)
    vol = (rng.random((9, 11, 13)) * 250).astype(dtype)
    w = OmeZarrWriter(str(tmp_path / "p.zarr"), vol.shape, dtype, chunks=(4, 4, 8), levels=2)
    w.write(0, vol)
    w.close()
    v = vol.astype(np.float64)
    want = np.zeros((5, 6, 7))
    for z in range(5):
        for y in range(6):
            for x in range(7):
                want[z, y, x] = v[2 * z:2 * z + 2, 2 * y:2 * y + 2, 2 * x:2 * x + 2].mean()
    got = w.read(1, 0, 5)
    if dtype == np.uint8:
        np.testing.assert_array_equal(got, np.rint(want).astype(np.uint8))
    else:
        np.testing.assert_allclose(got, want, rtol=1e-6)


@pytest.mark.parametrize("preset", ["fast", "slow", "delta"])
def test_delta3d_presets_round_trip(tmp_path, preset):
    vol = np.random.default_rng(5).integers(0, 255, (64, 64, 64), dtype=np.uint8)
    w = OmeZarrWriter(str(tmp_path / f"{preset}.zarr"), vol.shape, np.uint8, chunks=(32, 32, 32), levels=2,
                      compression="vc-delta3d", delta3d_preset=preset)
    w.write(0, vol)
    w.close()
    np.testing.assert_array_equal(w.read(0, 0, 64), vol)
    with pytest.raises((ValueError, RuntimeError), match="predictor"):
        OmeZarrWriter(str(tmp_path / "bad.zarr"), vol.shape, np.uint8, chunks=(32, 32, 32), compression="vc-delta3d",
                      delta3d_preset="fastest")
