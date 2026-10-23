"""Both EDT backends must dilate the same voxels, including at boundary distances."""
import numpy as np
import pytest
import zarr

pytest.importorskip("scipy.ndimage")

from vesuvius.image_proc.run.zarr_tasks.tasks import edt_dilate as m

BACKENDS = ["scipy"] + (["edt"] if m.HAS_EDT else [])

# float32-rounded boundary distances: float32(sqrt(2)) and float32(sqrt(3)) are just
# *below* sqrt(2) and sqrt(3), so those shells must be excluded.
CASES = [float(np.float32(x)) for x in (1.0, np.sqrt(2), np.sqrt(3), 2.0)]


def _create(group, name, **kw):
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


def _exact_count(dist):
    r = np.arange(-3, 4)
    sq = r[:, None, None] ** 2 + r[None, :, None] ** 2 + r[None, None, :] ** 2
    return int((sq <= dist * dist).sum())


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dist", CASES)
def test_single_voxel_boundary_distances(tmp_path, monkeypatch, backend, dist):
    monkeypatch.setattr(m, "HAS_EDT", backend == "edt")
    n = 33
    vol = np.zeros((n, n, n), np.uint8)
    vol[16, 16, 16] = 1
    _create(zarr.open_group(str(tmp_path / "in.zarr"), mode="w"), "0", data=vol, chunks=vol.shape)
    out = zarr.open_group(str(tmp_path / "out.zarr"), mode="w")
    _create(out, "0", shape=vol.shape, chunks=vol.shape, dtype=np.uint8, fill_value=0)
    for k, v in dict(
        input_path=str(tmp_path / "in.zarr"), output_path=str(tmp_path / "out.zarr"), resolution="0"
    ).items():
        monkeypatch.setitem(m._WORKER_STATE, k, v)
    m._edt_dilate_worker(((0, 0, 0), (0, n, 0, n, 0, n), dist, True))
    assert int(out["0"][:].sum()) == _exact_count(dist)


def test_expected_counts():
    # 1 -> centre + 6 faces; float32(sqrt 2) < sqrt 2 so still 7 (edt's float32
    # sqrt used to round up to it and give 19); float32(sqrt 3) -> 19; 2 -> 33.
    assert [_exact_count(d) for d in CASES] == [7, 7, 19, 33]
