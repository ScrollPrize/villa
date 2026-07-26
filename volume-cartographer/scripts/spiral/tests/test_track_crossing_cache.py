import dbm
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from build_track_crossings import build_cache
from tracks import (
    PackedTrackCollection,
    _build_crossing_partner_csr,
    _load_native_track_crossings,
    _materialize_cached_crossing_partner_table,
    _materialize_crossing_partner_table,
    _pack_track_points,
    _restrict_crossing_partner_csr,
    load_track_crossing_cache,
    load_tracks_from_dbm,
    prepare_main_phase_tracks,
    track_crossing_cache_path,
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
                    'max_track_crossing_per_step': 1,
                },
                track_families=families,
                track_source_ids=source_ids,
                crossing_cache=cache,
            )

            horizontal = families.index('horizontal')
            vertical = families.index('vertical')
            self.assertEqual(
                int(prepared['crossing_partners'][horizontal, 0]), vertical)
            self.assertEqual(
                int(prepared['crossing_partners'][vertical, 0]), horizontal)
            self.assertEqual(
                int(prepared['crossing_self_local'][horizontal, 0]), 10)
            self.assertEqual(
                int(prepared['crossing_partner_local'][horizontal, 0]), 10)

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
                    'max_track_crossing_per_step': 1,
                },
                track_families=family_codes,
                track_source_ids=source_ids,
                crossing_cache=cache,
            )
            self.assertEqual(prepared['lengths'].tolist(), [21, 21])
            self.assertEqual(prepared['crossing_partners'].tolist(), [[1], [0]])

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
                    'max_track_crossing_per_step': 1,
                },
                track_families=families,
                track_source_ids=source_ids,
                crossing_cache=cache,
            )
            self.assertTrue(np.all(
                prepared['crossing_partners'].numpy() >= 0))

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

    def test_native_fused_table_matches_python_restrict_and_spacing(self):
        horizontal = line_track(20, z=10, y=10, axis=2)
        tracks = [horizontal]
        for x in (3, 7, 12, 18):
            vertical = line_track(20, z=10, y=0, axis=1)
            vertical[:, 2] = x
            tracks.append(vertical)
        source_ids = np.arange(10, 60, 10, dtype=np.uint64)
        csr = _build_crossing_partner_csr(
            tracks, ['horizontal'] + ['vertical'] * 4,
            source_ids=source_ids)
        selected_source_ids = source_ids[[0, 1, 3, 4]]
        expected = _materialize_crossing_partner_table(
            _restrict_crossing_partner_csr(csr, selected_source_ids),
            3, 'cpu')
        actual = _materialize_cached_crossing_partner_table(
            csr, selected_source_ids, 3, 'cpu', workers=2)
        for expected_array, actual_array in zip(expected, actual):
            np.testing.assert_array_equal(
                expected_array.numpy(), actual_array.numpy())

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
