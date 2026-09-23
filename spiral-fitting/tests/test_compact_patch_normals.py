"""CPU equivalence checks; never allocate on a GPU."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from compact_patch_normals import CompactPatchNormalPool, _popcount
from sparse_cuda_cache import ResidentBrickPool
from pack_resident_pools import open_pool


def make_pool(path, edge):
    rng = np.random.default_rng(42)
    cells = edge**3
    values = np.zeros((4, cells, 3), dtype=np.uint8)
    values[1] = rng.integers(1, 256, size=(cells, 3), dtype=np.uint8)
    values[2] = values[1]
    values[2, rng.random(cells) < .85] = 0
    # Partial-zero vectors must remain present, including at word boundaries.
    for i in [0, 7, 8, 62, 63, 64, cells - 1]:
        if i < cells:
            values[2, i] = [0, 128, 0]
    table = np.array([[[1, 0]], [[2, 3]]], dtype=np.int32)
    coords = np.array([[-1, -1, -1], [0, 0, 0], [1, 0, 0], [1, 0, 1]], dtype=np.int32)
    meta = dict(format='respool', version=2, rows=4, channels=['nx', 'ny', 'nz'],
                array_shape=[edge*2, edge, edge*2], brick_shape=[edge]*3)
    (path/'meta.json').write_text(json.dumps(meta))
    np.save(path/'table.npy', table)
    np.save(path/'brick_coords.npy', coords)
    for ch in range(3):
        values[..., ch].tofile(path/f'channel_{ch}.u8')
    return meta


@pytest.mark.parametrize('edge', [3, 4, 16])
@pytest.mark.parametrize('roi', [None, (0, 1), (17, 31), (1000, 1001)])
def test_every_cell_matches_resident_pool(tmp_path, edge, roi):
    meta = make_pool(tmp_path, edge)
    compact = CompactPatchNormalPool(tmp_path, z_roi=roi, device='cpu', batch_rows=2)
    original = ResidentBrickPool(tmp_path, z_roi=roi, device='cpu', label='reference')
    indices = torch.from_numpy(np.indices(meta['array_shape']).reshape(3, -1).T.copy())
    assert torch.equal(compact.gather(indices), original.gather(indices))
    assert compact.gather(indices[:0]).shape == (0, 3)
    assert torch.equal(compact.gather(indices[:12].reshape(2, 6, 3)),
                       original.gather(indices[:12].reshape(2, 6, 3)))
    actual_bytes = sum(t.numel()*t.element_size() for t in
                       [compact.bits, compact.prefix, compact.offsets, compact.values, compact.table])
    assert actual_bytes == compact.pool_bytes
    compact.close()
    original.close()
    assert compact.values is None and compact.bits is None


def test_popcount_signed_and_random_words():
    rng = np.random.default_rng(13)
    words = rng.integers(-(2**63), 2**63-1, size=10000, dtype=np.int64)
    words[:4] = [-1, -(2**63), 0, 2**63-1]
    expected = torch.tensor([(int(w) & ((1 << 64)-1)).bit_count() for w in words])
    assert torch.equal(_popcount(torch.from_numpy(words)), expected)


def test_rejects_nonempty_reserved_row(tmp_path):
    make_pool(tmp_path, 4)
    with (tmp_path/'channel_0.u8').open('r+b') as f:
        f.write(b'\x01')
    with pytest.raises(ValueError, match='reserved row'):
        CompactPatchNormalPool(tmp_path, device='cpu')


@pytest.mark.skipif(not os.environ.get('PATCH_NORMALS_TEST_EXPORT'),
                    reason='Set PATCH_NORMALS_TEST_EXPORT to a real patch export')
def test_real_compact_export():
    path = Path(os.environ['PATCH_NORMALS_TEST_EXPORT']) / 'signed_normals_u8.respool'
    meta, table, coords, pools = open_pool(path)
    brick = np.array(meta['brick_shape'])
    z = int(coords[len(coords) // 2, 0] * brick[0])
    rng = np.random.default_rng(42)
    points = (rng.random((60000, 3)) * [brick[0], *meta['array_shape'][1:]]).astype(np.int64)
    points[:, 0] += z
    b, local = points // brick, points % brick
    rows = table[b[:, 0], b[:, 1], b[:, 2]]
    linear = (local[:, 0] * brick[1] + local[:, 1]) * brick[2] + local[:, 2]
    expected = np.stack([p[rows, linear] for p in pools], axis=-1)
    compact = CompactPatchNormalPool(path, z_roi=(z, z + int(brick[0])), device='cpu')
    try:
        np.testing.assert_array_equal(compact.gather(torch.from_numpy(points)).numpy(), expected)
        assert compact.pool_bytes < compact.dense_pool_bytes
    finally:
        compact.close()
