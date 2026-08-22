"""The min3 pool must return exactly what the uncompressed pool returns.

The claim min3 makes is not "close enough", it is "the same bytes at half the
resident size", so every check here is an equality against the pool it stands
in for, on a sidecar built in the test rather than on a fixture.

The cases are the ones a wrong layout survives at random:

  * the code for voxel 7 of a block sits at bits 29..31, so it owns the sign
    bit of the int32 word the pool holds.  A decode that shifts without
    masking the sign extension reads that block wrong and nothing else.
  * a block spanning the full 7 that min3 allows, next to a flat block.
  * values at the top of the uint8 range, where minimum plus code is 255.
  * brick and block boundaries, where a linear index is easy to build with
    the wrong stride.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

from min3_pool import Min3BrickPool             # noqa: E402
from pack_min3 import encode_bricks             # noqa: E402
from sparse_cuda_cache import ResidentBrickPool  # noqa: E402

EDGE = 4
SIDE = 2
SHAPE = (8, 8, 8)


def build_volume():
    """A volume whose 2^3 blocks exercise the cases above."""
    rng = np.random.default_rng(7)
    base = rng.integers(0, 249, size=SHAPE, dtype=np.uint16)
    # Make every 2^3 block flat first, then add a controlled span, so no block
    # can accidentally exceed 7 and make the packer refuse for the wrong reason.
    base = base.reshape(4, SIDE, 4, SIDE, 4, SIDE)
    base[:] = base[:, :1, :, :1, :, :1]
    volume = base.reshape(SHAPE).astype(np.uint16)

    offsets = np.zeros(SHAPE, dtype=np.uint16)
    # Voxel 7 of a block is (z,y,x) odd in all three: give it the high codes so
    # bit 31 of the word is set.
    offsets[1::SIDE, 1::SIDE, 1::SIDE] = 7
    offsets[0, 0, 0] = 0
    volume = volume + offsets
    # One block sitting hard against the top of the range.
    volume[2:4, 2:4, 2:4] = 248
    volume[3, 3, 3] = 255
    return np.ascontiguousarray(volume.astype(np.uint8))


def bricks_of(volume):
    """(rows, edge^3) with row 0 reserved all-zero, plus the lookup table."""
    gz, gy, gx = (s // EDGE for s in SHAPE)
    table = np.zeros((gz, gy, gx), dtype=np.int32)
    rows = [np.zeros(EDGE ** 3, dtype=np.uint8)]
    for z in range(gz):
        for y in range(gy):
            for x in range(gx):
                table[z, y, x] = len(rows)
                rows.append(volume[z * EDGE:(z + 1) * EDGE,
                                   y * EDGE:(y + 1) * EDGE,
                                   x * EDGE:(x + 1) * EDGE].reshape(-1).copy())
    coords = np.zeros((len(rows), 3), dtype=np.int32)
    i = 1
    for z in range(gz):
        for y in range(gy):
            for x in range(gx):
                coords[i] = (z, y, x)
                i += 1
    return np.stack(rows), table, coords


def write_sidecars(root, volume):
    """Write a raw sidecar and its min3 re-encoding; return both paths."""
    rows, table, coords = bricks_of(volume)
    meta = {
        'format': 'respool',
        'version': 2,
        'array_shape': list(SHAPE),
        'brick_shape': [EDGE, EDGE, EDGE],
        'rows': int(rows.shape[0]),
        'channels': ['synthetic'],
    }
    raw = root / 'raw'
    raw.mkdir()
    (raw / 'meta.json').write_text(json.dumps(meta))
    np.save(raw / 'table.npy', table)
    np.save(raw / 'brick_coords.npy', coords)
    (raw / 'channel_0.u8').write_bytes(rows.tobytes())

    words, worst = encode_bricks(
        rows.reshape(-1, EDGE, EDGE, EDGE), SIDE)
    packed = root / 'min3'
    packed.mkdir()
    np.save(packed / 'table.npy', table)
    np.save(packed / 'brick_coords.npy', coords)
    (packed / 'channel_0.u32').write_bytes(
        words.astype('<u4', copy=False).tobytes())
    packed_meta = dict(meta)
    packed_meta.update(encoding='min3', encode_block=SIDE,
                       worst_block_span=int(worst))
    (packed / 'meta.json').write_text(json.dumps(packed_meta))
    return raw, packed, worst


class Min3PoolTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.volume = build_volume()
        cls.raw_dir, cls.min3_dir, cls.worst = write_sidecars(root, cls.volume)
        cls.raw = ResidentBrickPool(str(cls.raw_dir), label='raw',
                                    device='cpu')
        cls.min3 = Min3BrickPool(str(cls.min3_dir), label='min3',
                                 device='cpu')

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def all_indices(self):
        grid = np.indices(SHAPE).reshape(3, -1).T
        return torch.from_numpy(np.ascontiguousarray(grid)).to(torch.long)

    def test_the_fixture_exercises_the_sign_bit(self):
        """A test that never sets bit 31 would pass against a broken decode."""
        self.assertEqual(self.worst, 7)
        words = np.fromfile(self.min3_dir / 'channel_0.u32', dtype='<u4')
        self.assertTrue((words >> 31).any(),
                        'no word has its top code bit set; the sign-extension '
                        'case is untested')

    def test_pool_is_exactly_half(self):
        self.assertEqual(self.raw.pool_bytes, 2 * self.min3.pool_bytes)

    def test_every_voxel_decodes_to_the_source(self):
        idx = self.all_indices()
        got = self.min3.gather(idx).reshape(SHAPE)
        np.testing.assert_array_equal(got.numpy(), self.volume)

    def test_matches_the_uncompressed_pool_bitwise(self):
        idx = self.all_indices()
        self.assertTrue(torch.equal(self.raw.gather(idx), self.min3.gather(idx)))

    def test_gather_does_not_modify_its_input(self):
        idx = self.all_indices()
        before = idx.clone()
        self.min3.gather(idx)
        self.assertTrue(torch.equal(idx, before))

    def test_gather_does_not_consume_the_pool(self):
        """Decoding in place must not decode the resident words in place."""
        idx = self.all_indices()
        first = self.min3.gather(idx).clone()
        second = self.min3.gather(idx)
        self.assertTrue(torch.equal(first, second))

    def test_shape_is_preserved(self):
        idx = self.all_indices().reshape(8, 8, 8, 3)
        self.assertEqual(tuple(self.min3.gather(idx).shape), (8, 8, 8, 1))

    def test_empty_gather(self):
        idx = torch.zeros((0, 3), dtype=torch.long)
        self.assertEqual(tuple(self.min3.gather(idx).shape), (0, 1))

    def test_packer_refuses_a_block_it_cannot_encode(self):
        bad = np.zeros((1, EDGE, EDGE, EDGE), dtype=np.uint8)
        bad[0, 0, 0, 0] = 0
        bad[0, 1, 1, 1] = 8
        with self.assertRaises(ValueError):
            encode_bricks(bad, SIDE)


if __name__ == '__main__':
    unittest.main()
