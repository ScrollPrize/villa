import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
from vc_reader.reader.disk_cache import decoded_cache_path

from dinovol_2.dataset.ssl_zarr_dataset import SSLZarrDataset
from dinovol_2.dataset.volume_reader import open_volume


class VolumeCacheTests(unittest.TestCase):
    def test_existing_chunks_are_reused_and_misses_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source' / 'volume.zarr' / '0'
            source.mkdir(parents=True)
            metadata = dict(zarr_format=2, shape=[4, 2, 2], chunks=[2, 2, 2],
                            dtype='|u1', compressor=None, fill_value=0, order='C',
                            filters=None, dimension_separator='/')
            (source / '.zarray').write_text(json.dumps(metadata))
            requests = []

            class Handler(SimpleHTTPRequestHandler):
                def do_GET(self):
                    requests.append(self.path)
                    super().do_GET()

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(
                Handler, directory=str(root / 'source')))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f'http://127.0.0.1:{server.server_port}/volume.zarr'
                cache_root = root / 'cache'
                cached = Path(decoded_cache_path(cache_root, url + '/0'))
                cached.mkdir(parents=True)
                (cached / '.zarray').write_text(json.dumps(metadata))
                # The first chunk exists only in a pre-existing decoded cache.
                first = np.full((2, 2, 2), 17, dtype=np.uint8)
                second = np.full((2, 2, 2), 29, dtype=np.uint8)
                (cached / '0/0').mkdir(parents=True)
                (cached / '0/0/0').write_bytes(first.tobytes())
                (source / '1/0').mkdir(parents=True)
                (source / '1/0/0').write_bytes(second.tobytes())
                expected = np.concatenate([first, second])
                with open_volume(url, 0, cache_dir=cache_root) as reader:
                    np.testing.assert_array_equal(reader[:], expected)
                    reader.flush()
                self.assertNotIn('/volume.zarr/0/0/0/0', requests)
                self.assertIn('/volume.zarr/0/1/0/0', requests)
                self.assertEqual((cached / '1/0/0').read_bytes(), second.tobytes())
                (source / '1/0/0').unlink()
                requests.clear()
                with open_volume(url, 0, cache_dir=cache_root) as reader:
                    np.testing.assert_array_equal(reader[:], expected)
                self.assertFalse(any(path.endswith('/0/0') for path in requests))
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_dataset_passes_cache_to_initialization_masks_and_worker_reads(self):
        config = dict(global_crop_size=[2] * 3, source_sampling_size=[2] * 3,
                      volume_cache_dir='/tmp/volume-cache-test', occupancy_level=1,
                      object_mask={'gap_um': 18},
                      datasets=[dict(volume_path='s3://bucket/volume.zarr',
                                     volume_scale=0, pixel_size_um=9.0)])
        with patch('dinovol_2.dataset.ssl_zarr_dataset.open_volume') as opened:
            opened.return_value.shape = (8, 8, 8)
            opened.return_value.__getitem__.side_effect = np.ones((8, 8, 8), dtype=np.uint8).__getitem__
            dataset = SSLZarrDataset(config)
            try:
                dataset._get_volume_array(dataset.volumes[0])
                dataset._load_object_mask(dataset.volumes[0])
                self.assertEqual(opened.call_count, 4)
                for call in opened.call_args_list:
                    self.assertEqual(call.kwargs['cache_dir'], config['volume_cache_dir'])
            finally:
                dataset.close()

    def test_recipe_is_volume_only_and_cache_matches_existing_layout(self):
        path = Path(__file__).resolve().parents[1] / 'dinovol_2/configs/pretrain_8_9um_heads32768.json'
        config = json.loads(path.read_text())
        dataset = config['data']['dataset']
        self.assertFalse(dataset['object_points']['enabled'])
        self.assertFalse(config['method']['object_losses']['enabled'])
        self.assertFalse(config['method']['point_supervision']['enabled'])
        for volume in dataset['datasets']:
            self.assertFalse(any(key.startswith(('surface_', 'fiber_', 'mesh_')) for key in volume))
            relative = volume['volume_path'].split('s3://vesuvius-challenge-open-data/', 1)[1]
            expected = Path('/mnt/raid_nvme/volume_cache/s3/vesuvius-challenge-open-data') / relative
            self.assertEqual(Path(decoded_cache_path(dataset['volume_cache_dir'], volume['volume_path'])), expected)
