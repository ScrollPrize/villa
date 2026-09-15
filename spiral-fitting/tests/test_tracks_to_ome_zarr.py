import dbm
import json
import pickle
import tempfile
import unittest
from pathlib import Path
import numpy as np
import zarr
from click.testing import CliRunner
from tracks_to_ome_zarr import SpatialLabeler, main


class TrackRasterizationTests(unittest.TestCase):
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


class TrackOmeZarrIntegrationTests(unittest.TestCase):
    def test_dbm_to_zstd_ome_zarr_is_independent_of_pipeline_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            database_path = temporary / 'tracks.dbm'
            output_path = temporary / 'tracks.ome.zarr'
            serial_path = temporary / 'serial.ome.zarr'
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

            common = [str(database_path), '--shape', '16,16,16',
                      '--chunk', '16', '--reuse-distance', '8']
            runner = CliRunner()
            serial = runner.invoke(main, [
                *common, '--out', str(serial_path), '--workers', '1',
                '--write-threads', '1', '--records-per-flush', '128',
            ])
            if serial.exception:
                raise serial.exception
            self.assertEqual(serial.exit_code, 0, serial.output)
            result = runner.invoke(main, [
                *common, '--out', str(output_path),
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
            self.assertNotEqual(int(array[8, 8, 8]), 0)
            # Rasterization fills the segments between DBM vertices.
            self.assertTrue((array[1, 1, 1:9] != 0).all())
            self.assertTrue((array[1, 1:9, 8] != 0).all())
            # Sorted record a owns the shared endpoint under first-track-wins.
            self.assertEqual(int(array[1, 1, 8]), int(array[1, 1, 1]))

            self.assertTrue((array[1:9, 8, 8] != 0).all())
            np.testing.assert_array_equal(
                zarr.open_group(serial_path, mode='r')['0'][:], array[:])

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
