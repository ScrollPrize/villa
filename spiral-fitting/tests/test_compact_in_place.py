"""In-place compaction must not corrupt, and must refuse what it cannot own.

Dropping the unkept rows with ``rows[keep]`` allocates a second array the size
of the survivors, so both exist at once: 2.16 GB beside 2.16 GB on the 2 um
store at three thousand slices, inside a peak a 16 GB machine cannot clear.
Moving the survivors down inside the original removes the duplicate.

The reason this is safe is narrow, so it is worth pinning: the write position
trails the read position only because compaction preserves order and produces a
prefix. Anything that breaks either property silently corrupts data, and the
two guards -- a read-only array, and a view into someone else's buffer -- both
exist because the packed track store is a read-only memmap that reaches this
code as a copy by accident of dtype.
"""
import unittest

import numpy as np

from tracks import _compact_rows_in_place


class CompactInPlaceTests(unittest.TestCase):

    def reference(self, rows, keep):
        return rows[keep].copy()

    def test_matches_fancy_indexing(self):
        rng = np.random.default_rng(0)
        for n in (1, 2, 17, 1000, 5000):
            rows = rng.integers(-2**20, 2**20, size=(n, 3)).astype(np.float32)
            keep = rng.random(n) > 0.35
            want = self.reference(rows, keep)
            got = _compact_rows_in_place(rows.copy(), keep)
            np.testing.assert_array_equal(got, want, err_msg=f'n={n}')

    def test_crosses_block_boundaries(self):
        # The default block is millions of rows, so nothing in a unit test
        # would ever span one. A small block exercises the case where the write
        # position has fallen behind the read position by more than a block.
        rng = np.random.default_rng(1)
        rows = rng.integers(0, 1000, size=(997, 3)).astype(np.float32)
        keep = rng.random(997) > 0.5
        want = self.reference(rows, keep)
        for block in (1, 2, 7, 64, 512):
            got = _compact_rows_in_place(rows.copy(), keep, block=block)
            np.testing.assert_array_equal(got, want, err_msg=f'block={block}')

    def test_keeps_order(self):
        rows = np.arange(300, dtype=np.float32).reshape(100, 3)
        keep = np.zeros(100, dtype=bool)
        keep[[0, 1, 50, 98, 99]] = True
        got = _compact_rows_in_place(rows.copy(), keep, block=8)
        np.testing.assert_array_equal(got[:, 0], [0, 3, 150, 294, 297])

    def test_all_kept_returns_the_input(self):
        rows = np.ones((10, 3), dtype=np.float32)
        got = _compact_rows_in_place(rows, np.ones(10, dtype=bool))
        self.assertIs(got.base if got.base is not None else got, rows)

    def test_none_kept(self):
        rows = np.ones((10, 3), dtype=np.float32)
        got = _compact_rows_in_place(rows, np.zeros(10, dtype=bool))
        self.assertEqual(got.shape, (0, 3))

    def test_refuses_a_read_only_array(self):
        """The packed store's coordinates are a read-only memmap.

        They arrive as a writable copy only because the store is int32 and the
        caller asks for float32. If that ever lines up, this must not write
        through to the file.
        """
        rows = np.ones((100, 3), dtype=np.float32)
        rows.flags.writeable = False
        keep = np.zeros(100, dtype=bool)
        keep[:60] = True
        got = _compact_rows_in_place(rows, keep)
        self.assertEqual(got.shape, (60, 3))
        self.assertTrue(rows.flags.writeable is False)

    def test_refuses_a_view(self):
        """Writing through a view rewrites whatever it is a window onto."""
        backing = np.arange(600, dtype=np.float32).reshape(200, 3)
        rows = backing[50:150]
        untouched = backing.copy()
        keep = np.zeros(100, dtype=bool)
        keep[::2] = True
        got = _compact_rows_in_place(rows, keep)
        np.testing.assert_array_equal(got, untouched[50:150][keep])
        np.testing.assert_array_equal(backing, untouched)

    def test_prefers_the_copy_when_most_rows_are_dropped(self):
        """Below half survival the stranded tail costs more than the copy."""
        rows = np.arange(3000, dtype=np.float32).reshape(1000, 3)
        keep = np.zeros(1000, dtype=bool)
        keep[:100] = True
        original = rows.copy()
        got = _compact_rows_in_place(rows, keep)
        np.testing.assert_array_equal(got, original[keep])
        np.testing.assert_array_equal(rows, original, 'input was mutated')

    def test_mask_length_must_match(self):
        rows = np.ones((10, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            _compact_rows_in_place(rows, np.ones(9, dtype=bool))


if __name__ == '__main__':
    unittest.main()
