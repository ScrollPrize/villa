"""The mmap crossing cache must load the same CSR the NPZ loads.

The point of the directory form is that its arrays stay mmap-backed, so the
kernel can reclaim them under pressure.  Two things therefore have to hold and
neither is visible in the returned values alone:

  * the arrays really are memory maps, not copies the loader made silently
  * a conversion that was interrupted is not loadable, so a short cache can
    never be mistaken for a complete one

Both are checked here, along with equality against the NPZ the directory was
converted from.
"""
import dbm
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_track_crossings import build_cache
from tracks import (
    load_track_crossing_cache,
    track_crossing_cache_path,
    track_crossing_mmap_cache_path,
    write_track_crossing_mmap_cache,
)


def line_track(length, *, z=10, y=10, axis=2):
    points = np.zeros((int(length) + 1, 3), dtype=np.int32)
    points[:, 0] = z
    points[:, 1] = y
    points[:, axis] = np.arange(int(length) + 1, dtype=np.int32)
    return points


class TrackCrossingMmapCacheTests(unittest.TestCase):

    def make_db(self, root):
        path = Path(root) / 'tracks.dbm'
        horizontal = line_track(20, z=10, y=10, axis=2)
        vertical = line_track(20, z=10, y=0, axis=1)
        vertical[:, 2] = 10
        with dbm.open(str(path), 'c') as database:
            database[b'h:0'] = pickle.dumps([horizontal])
            database[b'vy:0'] = pickle.dumps([vertical])
        return path

    def test_directory_form_loads_the_same_csr(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            build_cache(path, show_progress=False)
            from_npz = load_track_crossing_cache(path)
            self.assertIsNotNone(from_npz)

            destination = write_track_crossing_mmap_cache(path)
            self.assertEqual(destination, track_crossing_mmap_cache_path(path))
            from_dir = load_track_crossing_cache(path)
            self.assertIsNotNone(from_dir)
            self.assertEqual(set(from_npz), set(from_dir))
            for name in from_npz:
                np.testing.assert_array_equal(
                    from_npz[name], from_dir[name], err_msg=name)
                self.assertEqual(from_npz[name].dtype, from_dir[name].dtype,
                                 msg=name)

    def test_the_arrays_are_actually_mapped(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            build_cache(path, show_progress=False)
            write_track_crossing_mmap_cache(path)
            csr = load_track_crossing_cache(path)
            for name, array in csr.items():
                self.assertIsInstance(
                    array, np.memmap,
                    msg=f'{name} came back as a copy, so nothing was saved')

    def test_a_partial_conversion_is_not_loadable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            build_cache(path, show_progress=False)
            write_track_crossing_mmap_cache(path)
            (track_crossing_mmap_cache_path(path) / 'metadata.json').unlink()
            self.assertIsNone(load_track_crossing_cache(path, warn=False))

    def test_a_stale_directory_is_rejected_like_a_stale_npz(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            build_cache(path, show_progress=False)
            destination = write_track_crossing_mmap_cache(path)
            metadata_path = destination / 'metadata.json'
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            # Corrupt every signature the metadata carries, not just the one
            # this branch happens to write.  A tree that also records a content
            # signature would otherwise accept the directory on the field the
            # test left intact, and the test would pass by not testing.
            signatures = [k for k in metadata if 'signature' in k]
            self.assertTrue(signatures, 'no signature field to invalidate')
            for key in signatures:
                metadata[key] = [['tracks.dbm', 1, 1]]
            metadata_path.write_text(json.dumps(metadata), encoding='utf-8')
            self.assertIsNone(load_track_crossing_cache(path, warn=False))

    def test_conversion_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            build_cache(path, show_progress=False)
            write_track_crossing_mmap_cache(path)
            with self.assertRaises(FileExistsError):
                write_track_crossing_mmap_cache(path)

    def test_the_npz_still_loads_when_no_directory_exists(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            build_cache(path, show_progress=False)
            self.assertTrue(track_crossing_cache_path(path).is_file())
            self.assertFalse(track_crossing_mmap_cache_path(path).exists())
            self.assertIsNotNone(load_track_crossing_cache(path))


if __name__ == '__main__':
    unittest.main()
