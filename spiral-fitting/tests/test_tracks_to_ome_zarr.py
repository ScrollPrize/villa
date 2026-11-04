import dbm
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr
from click.testing import CliRunner

from tracks_to_ome_zarr import (
    ALL_LABEL_BITS,
    LocalLabeler,
    RoundRobinLabeler,
    SpatialLabeler,
    _rasterize_polyline,
    main,
)


class TrackRasterizationTests(unittest.TestCase):
    def test_rasterize_polyline_fills_intervertex_gaps(self):
        track = np.asarray([[1, 2, 3], [1, 2, 7], [3, 4, 7]], dtype=np.int32)
        points = _rasterize_polyline(track)
        np.testing.assert_array_equal(
            points,
            np.asarray([
                [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6], [1, 2, 7],
                [2, 3, 7], [3, 4, 7],
            ]),
        )

    def test_spatial_labeler_separates_nearby_tracks(self):
        labeler = SpatialLabeler(reuse_distance=16)
        first = labeler.assign(np.asarray([[10, 10, 10], [10, 10, 12]]))
        nearby = labeler.assign(np.asarray([[11, 10, 10], [11, 10, 12]]))
        self.assertNotEqual(first, nearby)

    def test_spatial_labeler_can_reuse_a_label_far_away(self):
        labeler = SpatialLabeler(reuse_distance=4)
        labels = [
            labeler.assign(np.asarray([[index * 20, 0, 0]]))
            for index in range(256)
        ]
        self.assertEqual(len(set(labels[:255])), 255)
        self.assertEqual(labels[255], labels[0])
        self.assertEqual(labeler.forced_reuses, 0)

    def test_spatial_labeler_forced_reuse_picks_least_local_label(self):
        labeler = SpatialLabeler(reuse_distance=4)
        labeler.occupancy[(0, 0, 0)] = ALL_LABEL_BITS
        labeler.occupancy[(1, 0, 0)] = ALL_LABEL_BITS & ~(1 << 255)

        label = labeler.assign(np.asarray([[0, 0, 0]]))

        self.assertEqual(label, 255)
        self.assertEqual(labeler.forced_reuses, 1)

    def test_integer_key_spatial_labeler_matches_tuple_key_labels(self):
        tracks = [
            np.asarray([[1, 1, 1], [1, 1, 7]]),
            np.asarray([[2, 1, 1], [2, 1, 7]]),
            np.asarray([[40, 40, 40], [40, 40, 48]]),
        ]
        tuple_labeler = SpatialLabeler(reuse_distance=4)
        integer_labeler = SpatialLabeler(reuse_distance=4, shape=(64, 64, 64))

        self.assertEqual(
            tuple_labeler.assign_many(tracks),
            integer_labeler.assign_many(tracks),
        )

    def test_local_labeler_avoids_reuse_in_crossed_cells(self):
        labeler = LocalLabeler(reuse_distance=4, shape=(16, 16, 16))
        crossing = labeler.assign(np.asarray([[1, 1, 1], [1, 1, 5]]))
        sharing = labeler.assign(np.asarray([[1, 1, 5], [1, 1, 6]]))

        self.assertNotEqual(crossing, sharing)

    def test_local_labeler_reports_unavoidable_cell_reuse(self):
        labeler = LocalLabeler(reuse_distance=4, shape=(16, 16, 16))
        track = np.asarray([[1, 1, 1]])
        labels = [labeler.assign(track) for _ in range(256)]

        self.assertEqual(len(set(labels[:255])), 255)
        self.assertEqual(labels[255], labels[0])
        self.assertEqual(labeler.forced_reuses, 1)

    def test_round_robin_labeler_cycles_all_nonzero_values(self):
        labels = RoundRobinLabeler().assign_many([None] * 256)

        np.testing.assert_array_equal(labels[:255], np.arange(1, 256, dtype=np.uint8))
        self.assertEqual(int(labels[255]), 1)


class TrackOmeZarrIntegrationTests(unittest.TestCase):
    def test_dbm_to_zstd_ome_zarr(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            database_path = temporary / 'tracks.dbm'
            output_path = temporary / 'tracks.ome.zarr'
            with dbm.open(str(database_path), 'c') as database:
                database[b'a'] = pickle.dumps([
                    np.asarray([[1, 1, 1], [1, 1, 5]], dtype=np.int32),
                ])
                database[b'b'] = pickle.dumps([
                    np.asarray([[1, 1, 5], [1, 5, 5]], dtype=np.int32),
                ])

            result = CliRunner().invoke(main, [
                str(database_path),
                '--out', str(output_path),
                '--shape', '16,16,16',
                '--chunk', '16',
                '--reuse-distance', '8',
                '--workers', '2',
                '--write-threads', '2',
                '--batch-points', '1',
                '--records-per-flush', '1',
            ])
            if result.exception:
                raise result.exception
            self.assertEqual(result.exit_code, 0, result.output)

            root = zarr.open_group(output_path, mode='r')
            array = root['0']
            self.assertEqual(array.dtype, np.dtype('uint8'))
            self.assertEqual(array.shape, (16, 16, 16))
            self.assertNotEqual(int(array[1, 1, 1]), 0)
            self.assertNotEqual(int(array[1, 5, 5]), 0)
            # Sorted record a owns the shared endpoint under first-track-wins.
            self.assertEqual(int(array[1, 1, 5]), int(array[1, 1, 1]))

            multiscales = root.attrs['multiscales']
            self.assertEqual(multiscales[0]['version'], '0.4')
            self.assertEqual(multiscales[0]['datasets'][0]['path'], '0')
            self.assertEqual(root.attrs['label_mode'], 'local')
            self.assertTrue(root.attrs['complete'])

            with open(output_path / '0' / '.zarray') as stream:
                metadata = json.load(stream)
            self.assertEqual(metadata['compressor']['id'], 'zstd')
            self.assertEqual(metadata['compressor']['level'], 3)
            self.assertEqual(metadata['dimension_separator'], '/')
            self.assertFalse(Path(f'{output_path}.tracks-progress.json').exists())

    def test_pipeline_schedule_does_not_change_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            database_path = temporary / 'tracks.dbm'
            serial_path = temporary / 'serial.ome.zarr'
            pipelined_path = temporary / 'pipelined.ome.zarr'
            with dbm.open(str(database_path), 'c') as database:
                database[b'a'] = pickle.dumps([
                    np.asarray([[1, 1, 1], [1, 1, 8]], dtype=np.int32),
                ])
                database[b'b'] = pickle.dumps([
                    np.asarray([[1, 1, 8], [1, 8, 8]], dtype=np.int32),
                ])
                database[b'c'] = pickle.dumps([
                    np.asarray([[1, 8, 8], [8, 8, 8]], dtype=np.int32),
                ])

            runner = CliRunner()
            common = [str(database_path), '--shape', '16,16,16', '--chunk', '16']
            serial = runner.invoke(main, [
                *common, '--out', str(serial_path), '--workers', '1',
                '--write-threads', '1', '--records-per-flush', '128',
            ])
            if serial.exception:
                raise serial.exception
            pipelined = runner.invoke(main, [
                *common, '--out', str(pipelined_path), '--workers', '2',
                '--write-threads', '2', '--records-per-flush', '1',
            ])
            if pipelined.exception:
                raise pipelined.exception

            np.testing.assert_array_equal(
                zarr.open_group(serial_path, mode='r')['0'][:],
                zarr.open_group(pipelined_path, mode='r')['0'][:],
            )


if __name__ == '__main__':
    unittest.main()
