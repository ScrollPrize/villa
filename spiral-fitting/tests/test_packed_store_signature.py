"""A published sidecar must survive being copied to another machine.

The packed track store and the crossing cache both fingerprint their source
DBM. That fingerprint used to include ``st_mtime_ns``, which is not a property
of the bytes and is not preserved by copying or downloading, so every sidecar
shipped with a dataset was refused on every machine except the one that wrote
it. These tests pin the portable behaviour and the change detection it has to
keep.

Nothing here needs the native track-store extension: the writer and the
signature check are pure Python, so the regression stays covered on machines
that have not built the C++ side.
"""
import dbm
import json
import os
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from build_track_crossings import build_cache
from tracks import (
    _packed_store_if_current,
    _tracks_db_signature,
    _tracks_db_content_signature,
    load_track_crossing_cache,
    track_store_path,
    write_packed_track_store,
)


def line_track(length, *, z=10, y=10, axis=2, offset=0):
    points = np.zeros((int(length) + 1, 3), dtype=np.int32)
    points[:, 0] = z
    points[:, 1] = y
    points[:, axis] = np.arange(int(length) + 1, dtype=np.int32) + offset
    return points


class PackedStoreSignatureTests(unittest.TestCase):
    def make_db(self, root, *, offset=0):
        path = Path(root) / 'tracks.dbm'
        with dbm.open(str(path), 'c') as database:
            database[b'h:0'] = pickle.dumps(
                [line_track(20, z=10, y=10, axis=2, offset=offset)])
            database[b'vy:0'] = pickle.dumps(
                [line_track(20, z=10, y=0, axis=1, offset=offset)])
        return path

    def backing_files(self, path):
        logical = Path(str(path))
        return [candidate
                for candidate in (logical, *(Path(str(logical) + suffix)
                                             for suffix in ('.db', '.dat',
                                                            '.dir', '.pag')))
                if candidate.is_file()]

    def retouch(self, path):
        """Give the DBM the modification time a fresh download would give it."""
        for candidate in self.backing_files(path):
            stat = candidate.stat()
            os.utime(candidate, ns=(stat.st_atime_ns,
                                    stat.st_mtime_ns + 5_000_000_000))

    def test_store_survives_a_changed_modification_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            store = write_packed_track_store(path, show_progress=False)
            self.assertEqual(store, track_store_path(path))

            self.retouch(path)
            self.assertIsNotNone(
                _packed_store_if_current(path),
                'a store must stay usable after the DBM is copied or '
                'downloaded, which is the only thing a new mtime proves')

    def test_store_is_rejected_when_the_contents_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            write_packed_track_store(path, show_progress=False)

            # Same key count and same pickle lengths, different coordinates:
            # the file size is unchanged, so only a content-derived signature
            # can tell that this is no longer the packed DBM.
            before = [candidate.stat().st_size
                      for candidate in self.backing_files(path)]
            self.make_db(temporary, offset=7)
            after = [candidate.stat().st_size
                     for candidate in self.backing_files(path)]
            self.assertEqual(before, after, 'test needs an equal-size rewrite')

            self.assertIsNone(_packed_store_if_current(path))

    def test_legacy_store_is_accepted_on_name_and_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            store = write_packed_track_store(path, show_progress=False)

            # A store written before the portable signature existed.
            metadata_path = store / 'metadata.json'
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            metadata.pop('source_db_signature_content')
            metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

            self.retouch(path)
            self.assertIsNotNone(
                _packed_store_if_current(path),
                'the shipped 12 GiB store carries only the legacy signature, '
                'so it has to be honoured on the fields that travel with it')

    def test_legacy_store_still_rejected_when_the_size_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            store = write_packed_track_store(path, show_progress=False)
            metadata_path = store / 'metadata.json'
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            metadata.pop('source_db_signature_content')
            metadata_path.write_text(json.dumps(metadata), encoding='utf-8')

            with dbm.open(str(path), 'c') as database:
                database[b'h:1'] = pickle.dumps([line_track(400)])

            self.assertIsNone(_packed_store_if_current(path))

    def test_content_signature_ignores_only_the_timestamp(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            legacy_before = _tracks_db_signature(path)
            content_before = _tracks_db_content_signature(path)

            self.retouch(path)
            self.assertNotEqual(legacy_before, _tracks_db_signature(path))
            self.assertEqual(content_before, _tracks_db_content_signature(path))


    def test_crossing_cache_survives_a_changed_modification_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            build_cache(path, show_progress=False)
            self.assertIsNotNone(load_track_crossing_cache(path))

            self.retouch(path)
            self.assertIsNotNone(
                load_track_crossing_cache(path),
                'the crossing cache carries the same defect as the packed '
                'store and has to be fixed with it')


if __name__ == '__main__':
    unittest.main()
