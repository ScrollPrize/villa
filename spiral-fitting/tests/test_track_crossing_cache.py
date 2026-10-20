import dbm
import json
import os
import pickle
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from build_track_crossings import build_cache
from tracks import (
    PackedTrackCollection,
    _build_crossing_partner_csr,
    _db_signatures_match,
    _load_native_track_crossings,
    _pack_track_points,
    _packed_store_if_current,
    _restrict_crossing_partner_csr,
    load_track_crossing_cache,
    load_tracks_from_dbm,
    prepare_main_phase_tracks,
    _sample_prepared_track_points,
    _tracks_db_signature,
    track_crossing_cache_path,
    track_store_path,
    write_packed_track_store,
)


def line_track(length, *, z=10, y=10, axis=2):
    points = np.zeros((int(length) + 1, 3), dtype=np.int32)
    points[:, 0] = z
    points[:, 1] = y
    points[:, axis] = np.arange(int(length) + 1, dtype=np.int32)
    return points


class TrackCrossingCacheTests(unittest.TestCase):
    def make_db(self, root):
        path = Path(root) / 'tracks.dbm'
        horizontal = line_track(20, z=10, y=10, axis=2)
        vertical = line_track(20, z=10, y=0, axis=1)
        vertical[:, 2] = 10
        outside = line_track(20, z=30, y=10, axis=2)
        with dbm.open(str(path), 'c') as database:
            database[b'h:0'] = pickle.dumps([horizontal, outside])
            database[b'vy:0'] = pickle.dumps([vertical])
        return path

    def test_builder_writes_adjacent_versioned_csr(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            destination = build_cache(path, show_progress=False)

            self.assertEqual(destination, track_crossing_cache_path(path))
            self.assertTrue(destination.is_file())
            cache = load_track_crossing_cache(path)
            self.assertIsNotNone(cache)
            self.assertEqual(len(cache['source_ids']), 3)
            # One accepted pair is represented in both adjacency directions.
            self.assertEqual(len(cache['partners']), 2)
            np.testing.assert_array_equal(
                cache['offsets'][1:] - cache['offsets'][:-1], [1, 0, 1])

    def test_fit_remaps_whole_db_cache_after_z_filtering(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            build_cache(path, show_progress=False)
            cache = load_track_crossing_cache(path)
            tracks, families, source_ids = load_tracks_from_dbm(
                path, 0, 20, return_families=True,
                return_source_ids=True, show_progress=False)

            prepared = prepare_main_phase_tracks(
                tracks, None, 0.0, 'cpu',
                sampling_config={
                    'track_crossing_precompute_max': 1,
                    'track_max_track_crossing_per_step': 1,
                },
                track_families=families,
                track_source_ids=source_ids,
                crossing_cache=cache,
            )

            horizontal = families.index('horizontal')
            vertical = families.index('vertical')
            self.assertIn('crossing_index', prepared)
            self.assertEqual(
                int(prepared['crossing_index_stats']['directed_crossings']), 2)
            probabilities = np.zeros(len(families), dtype=np.float32)
            probabilities[horizontal] = 1
            prepared['sampling_probabilities'] = torch.from_numpy(probabilities)
            sample = _sample_prepared_track_points(prepared, 1, 4)
            self.assertEqual(sample['track_idx'].tolist(), [horizontal, vertical])
            np.testing.assert_array_equal(
                sample['sampled_scroll'][sample['primary_cross_flat'][0]].numpy(),
                sample['sampled_scroll'][sample['partner_cross_flat'][0]].numpy())

    def test_packed_store_loads_and_prepares_without_track_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            build_cache(path, show_progress=False)
            cache = load_track_crossing_cache(path)
            write_packed_track_store(path, show_progress=False)
            build_cache(
                path, force=True, z_lo=0, z_hi=20,
                show_progress=False)
            cache = load_track_crossing_cache(
                path, expected_z_range=(0, 20))

            tracks, family_codes, source_ids = load_tracks_from_dbm(
                path, 0, 20, return_families=True,
                return_source_ids=True, show_progress=False)

            self.assertIsInstance(tracks, PackedTrackCollection)
            self.assertEqual(len(tracks), 2)
            np.testing.assert_array_equal(family_codes, [0, 1])
            np.testing.assert_array_equal(source_ids, [0, 1 << 32])
            prepared = prepare_main_phase_tracks(
                tracks, None, 0.0, 'cpu',
                sampling_config={
                    'track_crossing_precompute_max': 1,
                    'track_max_track_crossing_per_step': 1,
                },
                track_families=family_codes,
                track_source_ids=source_ids,
                crossing_cache=cache,
            )
            self.assertEqual(prepared['lengths'].tolist(), [21, 21])
            self.assertIn('crossing_index', prepared)
            self.assertEqual(
                int(prepared['crossing_index_stats']['directed_crossings']), 2)

    def test_builder_limits_cache_to_half_open_z_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            build_cache(
                path, z_lo=0, z_hi=20, show_progress=False)

            cache = load_track_crossing_cache(
                path, expected_z_range=(0, 20))
            self.assertIsNotNone(cache)
            self.assertEqual(len(cache['source_ids']), 2)
            self.assertEqual(len(cache['partners']), 2)
            self.assertIsNone(load_track_crossing_cache(
                path, warn=False, expected_z_range=(20, 40)))

            tracks, families, source_ids = load_tracks_from_dbm(
                path, 0, 20, return_families=True,
                return_source_ids=True, show_progress=False,
                low_memory=True)
            self.assertTrue(all(track.dtype == np.int32 for track in tracks))
            prepared = prepare_main_phase_tracks(
                tracks, None, 0.0, 'cpu',
                sampling_config={
                    'track_crossing_precompute_max': 1,
                    'track_max_track_crossing_per_step': 1,
                },
                track_families=families,
                track_source_ids=source_ids,
                crossing_cache=cache,
            )
            self.assertEqual(
                int(prepared['crossing_index_stats']['connected_tracks']), 2)

    def test_hybrid_builder_uses_first_local_index_for_repeated_voxel(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'tracks.dbm'
            horizontal = np.array([
                [10, 10, 0],
                [10, 10, 1],
                [10, 10, 2],
                [10, 10, 1],
                [10, 10, 2],
                [10, 10, 3],
            ], dtype=np.int32)
            vertical = np.array([
                [10, 8, 2],
                [10, 9, 2],
                [10, 10, 2],
                [10, 11, 2],
                [10, 12, 2],
            ], dtype=np.int32)
            with dbm.open(str(path), 'c') as database:
                database[b'h:0'] = pickle.dumps([horizontal])
                database[b'vy:0'] = pickle.dumps([vertical])

            build_cache(path, show_progress=False)
            cache = load_track_crossing_cache(path)

            self.assertIsNotNone(cache)
            self.assertEqual(len(cache['partners']), 2)
            self.assertEqual(int(cache['self_local'][0]), 2)
            self.assertEqual(int(cache['partner_local'][0]), 2)

    def test_hybrid_parallel_builder_matches_serial(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            build_cache(path, workers=1, show_progress=False)
            serial = {
                name: values.copy()
                for name, values in load_track_crossing_cache(path).items()
            }

            build_cache(path, force=True, workers=2, show_progress=False)
            parallel = load_track_crossing_cache(path)

            self.assertIsNotNone(parallel)
            self.assertEqual(serial.keys(), parallel.keys())
            for name in serial:
                np.testing.assert_array_equal(serial[name], parallel[name])

    def test_builder_rejects_empty_z_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            with self.assertRaisesRegex(ValueError, 'less than'):
                build_cache(
                    path, z_lo=20, z_hi=20,
                    show_progress=False)

    def test_changed_dbm_invalidates_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db(temporary)
            build_cache(path, show_progress=False)
            with dbm.open(str(path), 'w') as database:
                database[b'h:new'] = pickle.dumps([line_track(4)])

            self.assertIsNone(load_track_crossing_cache(path, warn=False))

    def make_db_with_sidecars(self, root):
        path = self.make_db(root)
        build_cache(path, show_progress=False)
        write_packed_track_store(path, show_progress=False)
        return path

    def edit_sidecar_metadata(self, path, edit):
        """Apply ``edit`` to the metadata of both sidecars beside ``path``."""
        cache_path = track_crossing_cache_path(path)
        with np.load(cache_path, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in stored.files}
        metadata = json.loads(str(arrays['metadata'].item()))
        edit(metadata, 'db_signature', 'db_digest')
        arrays['metadata'] = np.asarray(json.dumps(metadata, sort_keys=True))
        with cache_path.open('wb') as stream:
            np.savez(stream, **arrays)
        store_metadata = track_store_path(path) / 'metadata.json'
        metadata = json.loads(store_metadata.read_text(encoding='utf-8'))
        edit(metadata, 'source_db_signature', 'source_db_digest')
        store_metadata.write_text(json.dumps(metadata), encoding='utf-8')

    def sidecars_current(self, path):
        crossings = load_track_crossing_cache(path, warn=False) is not None
        packed = _packed_store_if_current(path) == track_store_path(path)
        self.assertEqual(crossings, packed)
        return crossings

    def test_unchanged_mtimes_validate_without_hashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db_with_sidecars(temporary)
            metadata = json.loads(
                (track_store_path(path) / 'metadata.json').read_text(encoding='utf-8'))
            for digest in metadata['source_db_digest'].values():
                self.assertRegex(digest, '^sha256:[0-9a-f]{64}$')
            with mock.patch('tracks._tracks_db_digest',
                            side_effect=AssertionError('hashed')):
                self.assertTrue(self.sidecars_current(path))

    def test_same_size_rewrite_invalidates_sidecars_with_digest(self):
        # SQLite-backed DBMs (dbm.sqlite3, Python >= 3.14) are usually edited
        # without changing size, so names and sizes alone cannot catch this.
        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db_with_sidecars(temporary)
            name, size, _mtime = max(
                _tracks_db_signature(path), key=lambda item: item[1])
            backing = Path(temporary) / name
            with backing.open('r+b') as stream:
                stream.seek(size // 2)
                byte = stream.read(1)
                stream.seek(size // 2)
                stream.write(bytes([byte[0] ^ 0xFF]))
            os.utime(backing, ns=(10**18, 10**18))
            self.assertEqual(backing.stat().st_size, size)

            self.assertFalse(self.sidecars_current(path))

    def test_copied_dbm_with_new_mtimes_keeps_sidecars_with_digest(self):
        with (tempfile.TemporaryDirectory() as source,
              tempfile.TemporaryDirectory() as target):
            path = self.make_db_with_sidecars(source)
            copied = Path(target) / path.name
            for name, _size, _mtime in _tracks_db_signature(path):
                shutil.copyfile(Path(source) / name, Path(target) / name)
                os.utime(Path(target) / name, ns=(10**18 + 123_456_700,) * 2)
            shutil.copyfile(
                track_crossing_cache_path(path),
                track_crossing_cache_path(copied))
            shutil.copytree(track_store_path(path), track_store_path(copied))

            self.assertTrue(self.sidecars_current(copied))

    def test_sidecars_without_digest_keep_the_mtime_rule(self):
        stored_ns = 1785295357776482751

        def as_old_sidecar(metadata, signature_key, digest_key):
            metadata.pop(digest_key, None)
            metadata[signature_key] = [
                [name, size, stored_ns]
                for name, size, _mtime in metadata[signature_key]]

        with tempfile.TemporaryDirectory() as temporary:
            path = self.make_db_with_sidecars(temporary)
            self.edit_sidecar_metadata(path, as_old_sidecar)
            whole_second = stored_ns // 10**9 * 10**9
            for mtime_ns, current in (
                    (whole_second, True),             # HTTP Last-Modified
                    (stored_ns // 100 * 100, False),  # NTFS keeps 100 ns
                    (whole_second + 10**9, False),    # FAT 2 s, time zones
            ):
                with self.subTest(mtime_ns=mtime_ns):
                    for name, _size, _mtime in _tracks_db_signature(path):
                        os.utime(Path(temporary) / name, ns=(mtime_ns,) * 2)
                    self.assertEqual(self.sidecars_current(path), current)

    def test_parallel_crossing_scan_matches_single_worker(self):
        horizontal = line_track(20, z=10, y=10, axis=2)
        verticals = []
        for x in (4, 10, 16):
            track = line_track(20, z=10, y=0, axis=1)
            track[:, 2] = x
            verticals.append(track)
        tracks = [horizontal, *verticals]
        families = ['horizontal', 'vertical', 'vertical', 'vertical']
        source_ids = np.arange(len(tracks), dtype=np.uint64)

        serial = _build_crossing_partner_csr(
            tracks, families, source_ids=source_ids,
            workers=1, worker_chunk_groups=1)
        parallel = _build_crossing_partner_csr(
            tracks, families, source_ids=source_ids,
            workers=2, worker_chunk_groups=1)

        self.assertEqual(serial.keys(), parallel.keys())
        for name in serial:
            np.testing.assert_array_equal(serial[name], parallel[name])

    def test_native_crossing_kernel_matches_python_reference(self):
        native = _load_native_track_crossings()
        if native is None:
            self.skipTest('VC native crossing extension is not built')
        horizontal = line_track(20, z=10, y=10, axis=2)
        verticals = []
        for x in (4, 10, 16):
            track = line_track(20, z=10, y=0, axis=1)
            track[:, 2] = x
            verticals.append(track)
        tracks = [horizontal, *verticals]
        families = ['horizontal', 'vertical', 'vertical', 'vertical']
        family_codes = np.asarray([0, 1, 1, 1], dtype=np.int8)
        source_ids = np.arange(len(tracks), dtype=np.uint64)
        lengths = np.asarray([len(track) for track in tracks], dtype=np.int64)
        offsets = np.empty(len(tracks) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        coordinates = np.concatenate(tracks).astype(np.int32, copy=False)
        packed = _pack_track_points(coordinates)
        order = native.parallel_argsort(packed, workers=2)
        events = native.scan_crossing_events(
            coordinates, offsets, family_codes, packed, order, workers=2)
        actual = native.consolidate_crossing_events(
            events, coordinates, offsets, source_ids, workers=2)
        actual.pop('accepted_events')
        actual.pop('paired_tracks')

        expected = _build_crossing_partner_csr(
            tracks, families, source_ids=source_ids, workers=1)
        self.assertEqual(actual.keys(), expected.keys())
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])

    def test_native_resampler_returns_empty_crossing_record_map(self):
        native = _load_native_track_crossings()
        if native is None:
            self.skipTest('VC native crossing extension is not built')
        tracks = [
            line_track(2, z=10, y=10),
            line_track(2, z=20, y=20),
        ]
        crossing_index = native.prepare_crossing_index(
            np.zeros(3, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.asarray([len(track) for track in tracks], dtype=np.int32),
        )
        result = native.resample_tracks(
            np.concatenate(tracks).astype(np.float32),
            np.asarray([0, len(tracks[0]), sum(map(len, tracks))],
                       dtype=np.int64),
            minimum_spacing=1.0, maximum_spacing=2.0,
            crossing_index=crossing_index,
        )

        self.assertIn('crossing_record_sample', result)
        np.testing.assert_array_equal(
            result['crossing_record_sample'],
            np.empty(0, dtype=np.int32))

    def test_native_radix_argsort_is_stable(self):
        native = _load_native_track_crossings()
        if native is None:
            self.skipTest('VC native crossing extension is not built')
        packed = np.asarray(
            [9, 1, 9, 3, 1, (1 << 60) - 1, 0], dtype=np.uint64)
        order = native.parallel_argsort(packed, workers=3)
        expected = np.argsort(packed, kind='stable').astype(np.uint32)
        np.testing.assert_array_equal(order, expected)


if __name__ == '__main__':
    unittest.main()


class DbSignatureMatchTests(unittest.TestCase):
    NS = 1785295357776482751                 # a real DBM mtime_ns; mirrors keep only 1785295357000000000

    def test_exact_match(self):
        self.assertTrue(_db_signatures_match([['tracks.dbm', 10, self.NS]], [['tracks.dbm', 10, self.NS]]))

    def test_mtime_truncated_to_whole_seconds_matches(self):
        truncated = self.NS // 1_000_000_000 * 1_000_000_000
        self.assertTrue(_db_signatures_match([['tracks.dbm', 10, self.NS]], [['tracks.dbm', 10, truncated]]))
        self.assertTrue(_db_signatures_match([['tracks.dbm', 10, truncated]], [['tracks.dbm', 10, self.NS]]))

    def test_local_edit_within_the_same_second_does_not_match(self):
        self.assertFalse(_db_signatures_match([['tracks.dbm', 10, self.NS]], [['tracks.dbm', 10, self.NS + 1000]]))

    def test_different_second_does_not_match(self):
        self.assertFalse(_db_signatures_match([['tracks.dbm', 10, self.NS]], [['tracks.dbm', 10, self.NS + 1_000_000_000]]))
        truncated = self.NS // 1_000_000_000 * 1_000_000_000
        self.assertFalse(_db_signatures_match([['tracks.dbm', 10, self.NS]], [['tracks.dbm', 10, truncated + 1_000_000_000]]))

    def test_name_or_size_mismatch(self):
        self.assertFalse(_db_signatures_match([['tracks.dbm', 10, self.NS]], [['other.dbm', 10, self.NS]]))
        self.assertFalse(_db_signatures_match([['tracks.dbm', 10, self.NS]], [['tracks.dbm', 11, self.NS]]))

    def test_length_mismatch(self):
        stored = [['tracks.dbm.dat', 10, self.NS], ['tracks.dbm.dir', 5, self.NS]]
        self.assertFalse(_db_signatures_match(stored, stored[:1]))
        self.assertFalse(_db_signatures_match(stored[:1], stored))

    def test_malformed_entries_do_not_match(self):
        expected = [['tracks.dbm', 10, self.NS]]
        for stored in (None, {}, [None], [['tracks.dbm', 10]], [['tracks.dbm', None, self.NS]],
                       [['tracks.dbm', 10, None]], [['tracks.dbm', 'ten', self.NS]]):
            with self.subTest(stored=stored):
                self.assertFalse(_db_signatures_match(stored, expected))
