import os

import numpy as np
import pytest

from helirec.omezarr import OmeZarrWriter, to_uint8


@pytest.mark.parametrize("compression,dtype", [("none", np.float32), ("zlib", np.float32), ("none", np.uint8),
                                               ("vc-delta3d", np.uint8)])
def test_roundtrip_any_slab_order(tmp_path, compression, dtype):
    if compression == "vc-delta3d":
        from helirec import _core
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
