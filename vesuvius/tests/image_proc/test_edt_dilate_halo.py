import numpy as np
import pytest
import zarr

scipy_ndimage = pytest.importorskip("scipy.ndimage")

from vesuvius.image_proc.run.zarr_tasks.tasks import edt_dilate as m


def _create(group, name, **kw):
    # zarr 2 has Group.create_dataset; zarr 3.x replaced it with create_array.
    if hasattr(group, "create_array"):
        data = kw.pop("data", None)
        if data is not None:
            kw.setdefault("shape", data.shape)
            kw.setdefault("dtype", data.dtype)
        arr = group.create_array(name, **kw)
        if data is not None:
            arr[...] = data
        return arr
    return group.create_dataset(name, **kw)


@pytest.mark.parametrize("black_border", [False, True])
def test_dilation_crosses_processing_chunk_boundaries(tmp_path, black_border):
    shape, ch, dist = (32, 64, 64), (32, 32, 32), 5
    vol = np.zeros(shape, np.uint8)
    vol[16, 30, 30] = 1  # 2 voxels from the y/x boundary at 32
    _create(zarr.open_group(str(tmp_path / "in.zarr"), mode="w"), "0", data=vol, chunks=ch)
    out = zarr.open_group(str(tmp_path / "out.zarr"), mode="w")
    _create(out, "0", shape=shape, chunks=ch, dtype=np.uint8, fill_value=0)
    m._WORKER_STATE.update(
        input_path=str(tmp_path / "in.zarr"), output_path=str(tmp_path / "out.zarr"), resolution="0"
    )
    for cy in range(2):
        for cx in range(2):
            bounds = (0, 32, cy * 32, cy * 32 + 32, cx * 32, cx * 32 + 32)
            m._edt_dilate_worker(((0, cy, cx), bounds, dist, black_border))
    got = out["0"][:].astype(bool)
    assert got[16, 32, 32] and got[16, 30, 35] and got[16, 35, 30]
    expected = _oracle(vol, dist, black_border)
    np.testing.assert_array_equal(got, expected)


def _oracle(vol, dist, black_border):
    """Whole-volume reference (scipy): black_border=False makes the volume faces foreground."""
    inv = vol == 0
    if not black_border:
        inv = np.pad(inv, 1, constant_values=False)
    elif inv.all():
        return np.zeros(vol.shape, bool)  # no foreground, nothing to dilate
    d = scipy_ndimage.distance_transform_edt(inv)
    if not black_border:
        d = d[1:-1, 1:-1, 1:-1]
    return d <= dist


def _run_chunked(tmp_path, vol, ch, dist, black_border):
    shape = vol.shape
    _create(zarr.open_group(str(tmp_path / "in.zarr"), mode="w"), "0", data=vol, chunks=ch)
    out = zarr.open_group(str(tmp_path / "out.zarr"), mode="w")
    _create(out, "0", shape=shape, chunks=ch, dtype=np.uint8, fill_value=0)
    m._WORKER_STATE.update(
        input_path=str(tmp_path / "in.zarr"), output_path=str(tmp_path / "out.zarr"), resolution="0"
    )
    for cz in range(0, shape[0], ch[0]):
        for cy in range(0, shape[1], ch[1]):
            for cx in range(0, shape[2], ch[2]):
                b = (cz, min(cz + ch[0], shape[0]), cy, min(cy + ch[1], shape[1]),
                     cx, min(cx + ch[2], shape[2]))
                m._edt_dilate_worker(((cz, cy, cx), b, dist, black_border))
    return out["0"][:].astype(bool)


@pytest.mark.parametrize("black_border", [False, True])
@pytest.mark.parametrize("dist", [5, 6.0, 7])
def test_voxel_exactly_ceil_distance_from_seam(tmp_path, black_border, dist):
    # Foreground voxel exactly ceil(dist) voxels before the y-seam at 32: the first voxel
    # of the next chunk lies at distance exactly dist and must be set. A halo of
    # ceil(dist)-1 (or dist//2) does not see the source voxel and leaves it 0.
    shape, ch = (16, 64, 16), (16, 32, 16)
    k = int(np.ceil(dist))
    vol = np.zeros(shape, np.uint8)
    vol[8, 32 - k, 8] = 1
    got = _run_chunked(tmp_path, vol, ch, dist, black_border)
    assert got[8, 32, 8], "voxel just across the seam at distance == ceil(d) not dilated"
    expected = _oracle(vol, dist, black_border)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("bad", [-1.0, -0.5, float("nan"), float("-inf")])
def test_negative_or_nonfinite_distance_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="non-negative"):
        m.EdtDilateConfig(input_zarr="in", output_zarr="out", num_workers=1, distance=bad)
    with pytest.raises(ValueError, match="non-negative"):
        m._edt_dilate_worker(((0, 0, 0), (0, 1, 0, 1, 0, 1), bad, True))
