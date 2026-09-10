import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

import lasagna_data
from pack_resident_pools import _make_chunk_reader, pack_arrays, sidecar_path
from lasagna_data import ensure_fit_sparse_stores
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


def write_raw_pack_source(path, data):
    """Write the minimal uncompressed Zarr v2 array consumed by the packer."""
    path.mkdir(parents=True)
    (path / '.zarray').write_text(json.dumps({
        'zarr_format': 2,
        'shape': list(data.shape),
        'chunks': list(data.shape),
        'dtype': '|u1',
        'compressor': None,
        'fill_value': 0,
        'order': 'C',
        'filters': None,
        'dimension_separator': '.',
    }))
    (path / '0.0.0').write_bytes(data.tobytes())


def test_legacy_vcz1_reader_uses_standalone_delta3d(tmp_path, monkeypatch):
    payload = b'VCZ1 encoded chunk'
    chunk = bytes([1, 2, 3, 4])
    calls = []

    def decompress_with_magic(encoded, expected_size, magic):
        calls.append((encoded, expected_size, magic))
        return chunk

    monkeypatch.setitem(
        sys.modules,
        'vc_delta3d',
        SimpleNamespace(decompress_with_magic=decompress_with_magic),
    )
    original_import = builtins.__import__

    def import_without_volume_cartographer(name, *args, **kwargs):
        if name == 'vc' or name.startswith('vc.'):
            raise AssertionError(f'unexpected Volume Cartographer import: {name}')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', import_without_volume_cartographer)
    chunk_path = tmp_path / '0.0.0'
    chunk_path.write_bytes(payload)

    reader = _make_chunk_reader({'id': 'vcz1'}, len(chunk))

    np.testing.assert_array_equal(
        reader(chunk_path), np.frombuffer(chunk, dtype=np.uint8))
    assert calls == [(payload, len(chunk), b'VCZ1')]


class RecordingProgress:
    def __init__(self):
        self.begins = []
        self.updates = []
        self.finishes = []

    def begin(self, operation, stage_name, **kwargs):
        self.begins.append((operation, stage_name, kwargs))

    def update(self, step=None, **kwargs):
        self.updates.append((step, kwargs))

    def finish(self, detail=None):
        self.finishes.append(detail)


def test_fit_builds_missing_sparse_stores_and_reports_progress(tmp_path):
    shape = (16, 16, 16)
    data = np.ones(shape, dtype=np.uint8)
    nx = tmp_path / 'nx.ome.zarr'
    ny = tmp_path / 'ny.ome.zarr'
    grad = tmp_path / 'grad.ome.zarr'
    sdt = tmp_path / 'sdt.ome.zarr'
    for root in (nx, ny, grad):
        write_raw_pack_source(root / '4', data)
    write_raw_pack_source(sdt / '1', data)

    progress = RecordingProgress()
    kwargs = dict(
        use_normals=True,
        use_spacing=True,
        use_sdt=True,
        normal_nx_zarr_path=str(nx),
        normal_ny_zarr_path=str(ny),
        grad_mag_zarr_path=str(grad),
        normal_zarr_group='4',
        sdt_zarr_path=str(sdt),
        sdt_zarr_group='1',
        progress=progress,
    )
    ensure_fit_sparse_stores(**kwargs)

    expected = [
        sidecar_path(str(nx), '4', pair=True),
        sidecar_path(str(grad), '4'),
        sidecar_path(str(sdt), '1'),
    ]
    assert all((Path(path) / 'meta.json').is_file() for path in expected)
    assert [stage for _, stage, _ in progress.begins] == [
        'Building Lasagna normal sparse store',
        'Building Lasagna gradient sparse store',
        'Building surface-distance sparse store',
    ]
    assert len(progress.finishes) == 3
    assert progress.updates
    assert all(update[1]['total_steps'] > 0 for update in progress.updates)

    # Complete sidecars are reused on subsequent fits without another build.
    event_counts = (len(progress.begins), len(progress.updates),
                    len(progress.finishes))
    ensure_fit_sparse_stores(**kwargs)
    assert event_counts == (len(progress.begins), len(progress.updates),
                            len(progress.finishes))


def test_missing_sparse_store_uses_windows_file_lock(tmp_path, monkeypatch):
    sidecar = tmp_path / 'pool'
    lock_calls = []
    fake_msvcrt = SimpleNamespace(
        LK_LOCK=1,
        LK_UNLCK=2,
        locking=lambda _fd, mode, length: lock_calls.append((mode, length)),
    )
    monkeypatch.setitem(sys.modules, 'msvcrt', fake_msvcrt)
    monkeypatch.setattr(lasagna_data, 'fcntl', None)

    def fake_pack_arrays(_array_dirs, out_dir, **_kwargs):
        Path(out_dir).mkdir()
        (Path(out_dir) / 'meta.json').write_text('{}')

    monkeypatch.setattr(lasagna_data, 'pack_arrays', fake_pack_arrays)

    result = lasagna_data._ensure_sidecar(
        [], str(sidecar), label='test', stage_name='test')

    assert result == str(sidecar)
    assert lock_calls == [(fake_msvcrt.LK_LOCK, 1),
                          (fake_msvcrt.LK_UNLCK, 1)]
    assert Path(f'{sidecar}.lock').read_bytes() == b'\0'


def test_pack_progress_callback_uses_console_report_cadence(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / '.zarray').write_text(json.dumps({
        'zarr_format': 2,
        'shape': [32, 16, 16],
        'chunks': [16, 16, 16],
        'dtype': '|u1',
        'compressor': None,
        'fill_value': 0,
        'order': 'C',
        'filters': None,
        'dimension_separator': '.',
    }))
    chunk = np.ones((16, 16, 16), dtype=np.uint8).tobytes()
    (source / '0.0.0').write_bytes(chunk)
    (source / '1.0.0').write_bytes(chunk)
    updates = []

    pack_arrays(
        [str(source)], str(tmp_path / 'pool'), label='test',
        progress_callback=lambda current, total, detail: updates.append(
            (current, total, detail)),
    )

    assert [current for current, _, _ in updates] == [0, 2]


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
