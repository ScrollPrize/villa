"""Streaming the exclusion must produce exactly what materialising produced.

The streamed path exists to keep a second and third copy of the selected
points from existing at once.  That is only worth anything if the bytes it
produces are the same bytes, so every check here is an equality against the
path it replaces -- including against the native compaction, which is what
runs in production when the extension is installed.

The cases are the ones a blocked gather gets wrong while still looking right
on a uniform fixture:

  * tracks of unequal length, so a block boundary lands mid-selection
  * a selection that is not the identity and not sorted contiguously
  * block sizes smaller than, equal to and larger than a single track
  * a mask that removes the whole of some tracks and part of others
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parent.parent))

from tracks import (                                  # noqa: E402
    PackedTrackCollection,
    _compact_blocked,
    _compact_rows_in_place,
)

RNG = np.random.default_rng(20260822)


def make_collection(track_count=97, rows=None):
    lengths = RNG.integers(1, 40, size=track_count).astype(np.int64)
    offsets = np.empty(track_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    total = int(offsets[-1])
    coordinates = RNG.integers(0, 5000, size=(total, 3)).astype(np.float32)
    return PackedTrackCollection(
        coordinates, offsets,
        source_ids=np.arange(track_count, dtype=np.uint64),
        family_codes=np.zeros(track_count, dtype=np.int8),
        arclengths=np.ones(track_count, dtype=np.float64),
        tortuosities=np.ones(track_count, dtype=np.float64),
        rows=rows)


class StreamedExclusionTests(unittest.TestCase):

    def gather(self, collection, block_points):
        pieces = []
        expected_begin = 0
        for begin, block in collection.iter_selected_blocks(block_points):
            self.assertEqual(begin, expected_begin,
                             'blocks must arrive in materialised order')
            pieces.append(np.array(block, copy=True))
            expected_begin += len(block)
        if not pieces:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(pieces, axis=0)

    def test_blocks_reproduce_materialize_for_the_identity(self):
        collection = make_collection()
        flat, offsets = collection.materialize()
        flat = np.asarray(flat, dtype=np.float32)
        np.testing.assert_array_equal(collection.selected_offsets,
                                      np.asarray(offsets, dtype=np.int64))
        for block_points in (1, 7, 64, 10_000):
            with self.subTest(block_points=block_points):
                np.testing.assert_array_equal(
                    self.gather(collection, block_points), flat)

    def test_blocks_reproduce_materialize_for_a_scattered_selection(self):
        base = make_collection()
        rows = RNG.permutation(len(base.source_ids))[:53].astype(np.int64)
        collection = base.subset(np.arange(len(rows)))  # keeps the shape valid
        collection = PackedTrackCollection(
            base.coordinates, base.offsets, base.source_ids,
            base.family_codes, base.arclengths, base.tortuosities, rows=rows)
        flat, offsets = collection.materialize()
        flat = np.asarray(flat, dtype=np.float32)
        np.testing.assert_array_equal(collection.selected_offsets,
                                      np.asarray(offsets, dtype=np.int64))
        for block_points in (1, 3, 50, 1 << 20):
            with self.subTest(block_points=block_points):
                np.testing.assert_array_equal(
                    self.gather(collection, block_points), flat)

    def test_blocked_compaction_equals_boolean_indexing(self):
        collection = make_collection()
        flat = np.asarray(collection.materialize()[0], dtype=np.float32)
        for fraction in (0.0, 0.13, 0.5, 0.87, 1.0):
            keep = RNG.random(len(flat)) < fraction
            with self.subTest(fraction=fraction):
                for block_points in (1, 11, 4096):
                    got = _compact_blocked(
                        collection.iter_selected_blocks(block_points),
                        keep, int(keep.sum()))
                    np.testing.assert_array_equal(got, flat[keep])

    def test_blocked_compaction_equals_the_in_place_path(self):
        collection = make_collection()
        flat = np.asarray(collection.materialize()[0], dtype=np.float32)
        keep = RNG.random(len(flat)) < 0.8
        reference = _compact_rows_in_place(np.array(flat, copy=True), keep)
        got = _compact_blocked(collection.iter_selected_blocks(1024),
                               keep, int(keep.sum()))
        np.testing.assert_array_equal(got, reference)

    def test_a_short_count_is_refused_rather_than_returned(self):
        collection = make_collection()
        keep = np.ones(int(collection.selected_offsets[-1]), dtype=bool)
        with self.assertRaises(ValueError):
            _compact_blocked(collection.iter_selected_blocks(64), keep,
                             int(keep.sum()) - 1)

    def test_empty_selection(self):
        base = make_collection()
        collection = PackedTrackCollection(
            base.coordinates, base.offsets, base.source_ids,
            base.family_codes, base.arclengths, base.tortuosities,
            rows=np.zeros(0, dtype=np.int64))
        self.assertEqual(list(collection.iter_selected_blocks()), [])
        self.assertEqual(int(collection.selected_offsets[-1]), 0)


if __name__ == '__main__':
    unittest.main()
