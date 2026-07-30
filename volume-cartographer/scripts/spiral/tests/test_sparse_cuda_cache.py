import numpy as np
import pytest
import torch
import zarr

from pack_resident_pools import pack_arrays, sidecar_path
from sdt_losses import sample_sdt_trilinear
from sparse_cuda_cache import ResidentBrickPool, SparseScalarStore


def write_array(path, data):
    array = zarr.open(
        str(path), mode='w', shape=data.shape, chunks=(16, 16, 16),
        dtype='|u1', compressor=None, fill_value=0, zarr_format=2,
        dimension_separator='/',
    )
    array[:] = data
    return str(path)


def make_pool(tmp_path, arrays, name, **kwargs):
    paths = [write_array(tmp_path / f'{name}_{i}', a)
             for i, a in enumerate(arrays)]
    out = pack_arrays(paths, sidecar_path(paths[0], '0'), label=name)
    return ResidentBrickPool(out, device='cpu', label=name, **kwargs)


def test_gather_matches_dense_multichannel(tmp_path):
    z, y, x = np.indices((40, 40, 70))
    first = ((z * 17 + y * 5 + x) % 251 + 1).astype(np.uint8)
    second = ((z * 3 + y * 11 + x * 7) % 251 + 1).astype(np.uint8)
    pool = make_pool(tmp_path, [first, second], 'pair')

    for indices in [
        torch.tensor([[0, 0, 0], [1, 2, 33], [5, 3, 63]]),
        torch.tensor([[35, 2, 2], [35, 35, 35]]),
        torch.zeros([0, 3], dtype=torch.long),
    ]:
        actual = pool.gather(indices)
        expected = torch.from_numpy(np.stack([
            first[tuple(indices.numpy().T)], second[tuple(indices.numpy().T)]
        ], axis=-1))
        torch.testing.assert_close(actual, expected)
    assert pool.stats()['gathers'] == 2  # the empty gather short-circuits


def test_absent_bricks_read_zero(tmp_path):
    data = np.zeros((48, 16, 16), dtype=np.uint8)
    data[:16] = 9  # only the first chunk row is occupied
    pool = make_pool(tmp_path, [data], 'sparse')
    assert pool.resident_bricks < pool.table.numel()
    values = pool.gather(torch.tensor([[2, 2, 2], [30, 5, 5], [47, 15, 15]]))
    assert values[:, 0].tolist() == [9, 0, 0]


def test_origin_and_z_roi_restriction(tmp_path):
    data = np.broadcast_to(
        (np.arange(64, dtype=np.uint16) % 251 + 1).astype(np.uint8)[:, None, None],
        (64, 4, 4),
    ).copy()
    paths = [write_array(tmp_path / 'large_z', data)]
    out = pack_arrays(paths, sidecar_path(paths[0], '0'), label='roi')
    pool = ResidentBrickPool(
        out, device='cpu', label='roi', origin_zyx=(32, 0, 0), z_roi=(32, 64))
    full = ResidentBrickPool(out, device='cpu', label='full')

    assert pool.resident_bricks < full.resident_bricks
    first = pool.gather(torch.tensor([[0, 0, 0]]))
    last = pool.gather(torch.tensor([[31, 3, 3]]))
    assert int(first[0, 0]) == int(data[32, 0, 0])
    assert int(last[0, 0]) == int(data[63, 3, 3])


def test_bounds_check_env(tmp_path, monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_RESIDENT_BOUNDS_CHECK', '1')
    data = np.ones((16, 16, 16), dtype=np.uint8)
    pool = make_pool(tmp_path, [data], 'bounds')
    with pytest.raises(IndexError):
        pool.gather(torch.tensor([[16, 0, 0]]))


def test_pack_ct_mask_zeroes_and_drops_bricks(tmp_path):
    from pack_resident_pools import CtMasker, verify_pool

    rng = np.random.default_rng(3)
    data = rng.integers(1, 255, size=(32, 32, 32), dtype=np.uint8)
    store = write_array(tmp_path / 'sdt', data)
    # CT at half resolution (ratio 2): zero except one occupied corner region,
    # so only target voxels [0:16, 0:16, 0:16] survive the mask.
    ct = np.zeros((16, 16, 16), dtype=np.uint8)
    ct[:8, :8, :8] = 7
    zarr.open(
        str(tmp_path / 'ct' / '2'), mode='w', shape=ct.shape, chunks=(8, 8, 8),
        dtype='|u1', compressor=None, fill_value=0, zarr_format=2,
        dimension_separator='.',
    )[:] = ct

    masker = CtMasker(tmp_path / 'ct', '2', data.shape)
    assert masker.ratio == (2, 2, 2)
    out = pack_arrays([store], sidecar_path(store, '0'), label='masked',
                      brick_shape=(8, 8, 8), ct_masker=masker)
    verify_pool(out, 500)  # mask-aware: pool zeros are accepted iff CT == 0

    pool = ResidentBrickPool(out, device='cpu', label='masked')
    assert pool.meta['ct_mask']['ratio'] == [2, 2, 2]
    # 64 bricks in the grid; only the 2x2x2 corner block survives
    assert pool.resident_bricks == 8 + 1
    inside = pool.gather(torch.tensor([[3, 3, 3]]))
    outside = pool.gather(torch.tensor([[3, 3, 20], [25, 25, 25]]))
    assert int(inside[0, 0]) == int(data[3, 3, 3])
    assert outside[:, 0].tolist() == [0, 0]


def test_sparse_sdt_sampling_matches_dense(tmp_path):
    x = np.arange(70, dtype=np.float32)
    encoded = (
        np.clip(np.rint(np.abs(x - 35.0) - 2.0), -127, 127) + 128
    ).astype(np.uint8)
    data = np.broadcast_to(encoded, (6, 6, 70)).copy()
    pool = make_pool(tmp_path, [data], 'sdt')
    dense = {
        "backend": "dense_test",
        "kind": "sdt",
        "volume": torch.from_numpy(data),
        "z_origin": 0,
        "scale_zyx": (1.0, 1.0, 1.0),
        "unit": 1.0,
        "offset": 128,
        "cap": 127.0,
        "shape": data.shape,
        "fingerprint": {},
    }
    sparse = {
        **dense,
        "backend": "sparse_cuda",
        "store": SparseScalarStore(pool),
    }
    sparse.pop("volume")
    points_dense = (
        torch.rand([256, 3]) * torch.tensor([4.0, 4.0, 68.0]) + 0.5
    ).requires_grad_(True)
    points_sparse = points_dense.detach().clone().requires_grad_(True)
    dense_value, dense_valid, dense_corners = sample_sdt_trilinear(
        dense, points_dense
    )
    sparse_value, sparse_valid, sparse_corners = sample_sdt_trilinear(
        sparse, points_sparse
    )
    torch.testing.assert_close(sparse_value, dense_value)
    torch.testing.assert_close(sparse_valid, dense_valid)
    torch.testing.assert_close(sparse_corners, dense_corners)

    dense_value.sum().backward()
    sparse_value.sum().backward()
    torch.testing.assert_close(points_sparse.grad, points_dense.grad)
