import contextlib
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
from vc_reader.reader.disk_cache import decoded_cache_path

from dinovol_2.dataset.volume_reader import open_volume
from dinovol_2.prefetch_volumes import (LogProgress, LevelPlan, configured_levels, inspect_level, main,
                                      prefetch_level, select_mask_chunks, staged_sources)


def config_for(url, cache):
    return {'data': {'dataset': {
        'volume_cache_dir': str(cache), 'occupancy_level': 5,
        'datasets': [{'sample_id': 'test', 'volume_path': url, 'volume_scale': 0}],
    }}}


class LoggingTests(unittest.TestCase):
    def test_periodic_progress_and_final_update_without_terminal_controls(self):
        output = io.StringIO()
        with patch('dinovol_2.prefetch_volumes.time.monotonic', return_value=0) as clock:
            with contextlib.redirect_stdout(output):
                with LogProgress(total=4, desc='sample L0', unit='chunk', interval=30) as progress:
                    progress.update(1)
                    self.assertEqual(len(output.getvalue().splitlines()), 1)
                    progress.set_postfix(GETs=2, fetched='1 MiB')
                    clock.return_value = 30
                    progress.update(1)
                    self.assertEqual(len(output.getvalue().splitlines()), 2)
                    progress.update(2)
        text = output.getvalue()
        self.assertIn('2/4 chunk (50.0%)', text)
        self.assertIn('4/4 chunk (100.0%)', text)
        self.assertIn('GETs=2 fetched=1 MiB', text)
        self.assertNotIn('\r', text)
        self.assertNotIn('\x1b', text)


class PlanningTests(unittest.TestCase):
    def test_all_coarse_levels_precede_training_and_training_scope_keeps_prerequisites(self):
        config = config_for('s3://bucket/a', '/tmp/cache')
        config['data']['dataset']['datasets'].append(
            dict(sample_id='b', volume_path='s3://bucket/b', volume_scale=2))
        coarse, fine = staged_sources(config, scope='training')
        self.assertEqual([v.level for v in coarse + fine], [5, 5, 0, 2])
        self.assertEqual(staged_sources(config, scope='training', mask_filter=False)[0], [])

    def test_mask_projection_matches_brute_force_with_padding_edges_and_slabs(self):
        rng = np.random.default_rng(10)
        for level in (0, 1, 3, 5):
            for padding in (0, 1, 3):
                with self.subTest(level=level, padding=padding):
                    factor = 2 ** level
                    shape, chunks = (73, 51, 35), (11, 13, 7)
                    mask_shape = tuple((d + factor - 1) // factor for d in shape)
                    values = (rng.random(mask_shape) < .01).astype(np.uint8)
                    source = Mock(level=level)
                    source.open.return_value = contextlib.nullcontext(values)
                    coarse = LevelPlan(source, mask_shape, (3, 4, 5), 1)
                    fine = LevelPlan(Mock(level=0), shape, chunks, 1)
                    expected = np.zeros(fine.grid, dtype=bool)
                    for index in np.ndindex(fine.grid):
                        lo = [max(0, i * c // factor - padding) for i, c in zip(index, chunks)]
                        hi = [min(m, (min((i + 1) * c, d) + factor - 1) // factor + padding)
                              for i, c, d, m in zip(index, chunks, shape, mask_shape)]
                        expected[index] = values[tuple(slice(a, b) for a, b in zip(lo, hi))].any()
                    select_mask_chunks(fine, coarse, padding=padding, slab_bytes=math.prod(mask_shape[1:]))
                    np.testing.assert_array_equal(fine.selected, expected)
                    self.assertEqual(len(list(fine.regions())), int(expected.sum()))
                    source.open.assert_called_once_with(cache_writes=False, offline=True)

    def test_levels_scopes_deduplication_and_filtering(self):
        config = config_for('s3://bucket/test', '/tmp/cache')
        entries = config['data']['dataset']['datasets']
        entries.extend([dict(entries[0]), dict(sample_id='local', volume_path='/tmp/local', volume_scale=0)])
        self.assertEqual([v.level for v in configured_levels(config)], [5, 0])
        self.assertEqual([v.level for v in configured_levels(config, scope='training')], [0])
        self.assertEqual([v.level for v in configured_levels(config, scope='occupancy')], [5])
        self.assertEqual(configured_levels(config, samples=['local']), [])
        with self.assertRaisesRegex(ValueError, 'Unknown sample'):
            configured_levels(config, samples=['typo'])
        self.assertEqual(configured_levels(config['data'], cache_dir='/tmp/override')[0].cache_dir,
                         '/tmp/override')


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source' / 'volume.zarr' / '0'
        self.source.mkdir(parents=True)
        self.metadata = dict(zarr_format=2, shape=[17, 8, 8], chunks=[8, 8, 8],
                             dtype='|u1', compressor=None, fill_value=0, order='C',
                             filters=None, dimension_separator='/')
        (self.source / '.zarray').write_text(json.dumps(self.metadata))
        self.expected = np.zeros((17, 8, 8), dtype=np.uint8)
        for index in (0, 1):
            chunk = np.full((8, 8, 8), 17 + index, dtype=np.uint8)
            path = self.source / str(index) / '0' / '0'
            path.parent.mkdir(parents=True)
            path.write_bytes(chunk.tobytes())
            self.expected[index * 8:(index + 1) * 8] = chunk
        # The final edge chunk is absent from the source and reads as fill_value.
        self.requests = []
        self.fail = set()
        fixture = self

        class Handler(SimpleHTTPRequestHandler):
            def do_GET(self):
                fixture.requests.append(self.path)
                if self.path in fixture.fail:
                    self.send_error(403)
                else:
                    super().do_GET()

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(
            Handler, directory=str(self.root / 'source')))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}/volume.zarr'
        self.cache = self.root / 'cache'
        self.config = config_for(self.url, self.cache)
        self.write_level('volume.zarr', 5, np.ones((1, 1, 1), dtype=np.uint8), (1, 1, 1))
        self.level = configured_levels(self.config, scope='training')[0]
        self.cached = Path(decoded_cache_path(self.cache, self.url + '/0'))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def chunk_requests(self):
        return [p for p in self.requests if p.rsplit('/', 1)[-1].isdigit()]

    def write_level(self, volume, level, values, chunks):
        root = self.root / 'source' / volume / str(level)
        root.mkdir(parents=True, exist_ok=True)
        metadata = dict(self.metadata, shape=list(values.shape), chunks=list(chunks))
        (root / '.zarray').write_text(json.dumps(metadata))
        grid = tuple((d + c - 1) // c for d, c in zip(values.shape, chunks))
        for index in np.ndindex(grid):
            block = values[tuple(slice(i * c, (i + 1) * c) for i, c in zip(index, chunks))]
            if not block.any():
                continue
            padded = np.zeros(chunks, dtype=values.dtype)
            padded[tuple(slice(0, d) for d in block.shape)] = block
            path = root.joinpath(*(str(i) for i in index))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(padded.tobytes())

    def test_coarse_first_then_filter_skips_empty_chunks_without_marking_them_absent(self):
        config = config_for(self.url.replace('volume.zarr', 'a.zarr'), self.cache)
        config['data']['dataset']['datasets'].append(
            dict(sample_id='b', volume_path=self.url.replace('volume.zarr', 'b.zarr'), volume_scale=0))
        for volume in ('a.zarr', 'b.zarr'):
            mask = np.zeros((8, 2, 2), dtype=np.uint8)
            mask[1, 0, 0] = 8
            fine = np.zeros((256, 64, 64), dtype=np.uint8)
            fine[32:64, :32, :32] = 8
            # Coarse masks may miss finer features: skipped chunks must stay fetchable.
            fine[224:256, :32, :32] = 7
            self.write_level(volume, 5, mask, (4, 2, 2))
            self.write_level(volume, 0, fine, (32, 32, 32))
        path = self.root / 'filtered.json'
        path.write_text(json.dumps(config))
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--config', str(path), '--connections', '3', '--compress-cache']), 0)
        chunks = self.chunk_requests()
        coarse_indices = [i for i, p in enumerate(chunks) if p.split('/')[2] == '5']
        fine_indices = [i for i, p in enumerate(chunks) if p.split('/')[2] == '0']
        self.assertLess(max(coarse_indices), min(fine_indices))
        self.assertEqual(len(coarse_indices), 4)
        self.assertEqual(len(fine_indices), 24)  # 12 selected per volume, instead of 32.
        self.assertIn('selected 12 / 32 chunks; skipped 20', output.getvalue())
        self.assertNotIn('/a.zarr/0/7/0/0', chunks)
        with open_volume(config['data']['dataset']['datasets'][0]['volume_path'], 0,
                         cache_dir=self.cache) as reader:
            np.testing.assert_array_equal(reader[224:256, :32, :32], 7)
        self.assertIn('/a.zarr/0/7/0/0', self.requests)

    def test_dry_run_metadata_only_and_edge_plan(self):
        config = self.root / 'config.json'
        config.write_text(json.dumps(self.config))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--config', str(config), '--scope', 'training', '--dry-run']), 0)
        self.assertFalse(self.cache.exists())
        self.assertEqual(self.chunk_requests(), [])
        plan = inspect_level(self.level)
        self.assertEqual(plan.count, 3)
        self.assertEqual(list(plan.regions())[-1], ((16, 0, 0), (1, 8, 8)))

    def test_direct_script_execution_uses_checkout_reader(self):
        config = self.root / 'config.json'
        config.write_text(json.dumps(self.config))
        package_dir = Path(__file__).resolve().parents[1] / 'dinovol_2'
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        result = subprocess.run(
            [sys.executable, 'prefetch_volumes.py', '--config', str(config),
             '--scope', 'training', '--connections', '2', '--compress-cache'],
            cwd=package_dir, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('Complete: 4 chunk positions', result.stdout)
        metadata = json.loads((self.cached / '.zarray').read_text())
        self.assertEqual(metadata['compressor'], {'id': 'vc-delta3d', 'quant': 1})

    def test_existing_cache_reused_missing_chunks_saved_and_resumable(self):
        self.cached.mkdir(parents=True)
        (self.cached / '.zarray').write_text(json.dumps(self.metadata))
        chunk = self.cached / '0/0/0'
        chunk.parent.mkdir(parents=True)
        chunk.write_bytes((self.source / '0/0/0').read_bytes())
        (self.source / '0/0/0').unlink()
        plan = inspect_level(self.level)
        stats = prefetch_level(plan, connections=3, batch_chunks=2, compress_cache=True)
        self.assertEqual(stats['chunks'], 3)
        self.assertNotIn('/volume.zarr/0/0/0/0', self.requests)
        self.assertTrue((self.cached / '1/0/0').exists())
        self.assertIsNone(json.loads((self.cached / '.zarray').read_text())['compressor'])
        self.requests.clear()
        self.assertEqual(prefetch_level(plan, connections=1, batch_chunks=1)['chunks'], 3)
        with open_volume(self.url, 0, cache_dir=self.cache) as reader:
            np.testing.assert_array_equal(reader[:], self.expected)
        self.assertEqual(self.chunk_requests(), [])

    def test_log_file_records_failure_resume_compression_and_appends(self):
        config = self.root / 'config.json'
        config.write_text(json.dumps(self.config))
        log = self.root / 'logs' / 'prefetch.log'
        args = ['--config', str(config), '--scope', 'training', '--no-mask-filter',
                '--connections', '64', '--compress-cache', '--log-file', str(log)]
        self.fail.add('/volume.zarr/0/1/0/0')
        self.assertEqual(main(args), 1)
        first = log.read_text()
        self.assertIn('Prefetch exited with status 1', first)
        self.fail.clear()
        self.assertEqual(main(args), 0)
        text = log.read_text()
        self.assertTrue(text.startswith(first))
        self.assertIn('connections=64 compress_cache=True', text)
        self.assertIn('test L0: 3/3 chunk (100.0%)', text)
        self.assertIn('GETs=', text)
        self.assertIn('Prefetch exited with status 0', text)
        self.assertNotIn('\r', text)
        self.assertNotIn('\x1b', text)
        self.assertEqual(json.loads((self.cached / '.zarray').read_text())['compressor'],
                         {'id': 'vc-delta3d', 'quant': 1})

    def test_training_reads_create_lossless_delta3d_cache_by_default(self):
        with open_volume(self.url, 0, cache_dir=self.cache) as reader:
            np.testing.assert_array_equal(reader[:], self.expected)
            reader.flush()
        metadata = json.loads((self.cached / '.zarray').read_text())
        self.assertEqual(metadata['compressor'], {'id': 'vc-delta3d', 'quant': 1})
        self.assertEqual((self.cached / '0/0/0').read_bytes()[:4], b'D3D1')
        self.requests.clear()
        with open_volume(self.url, 0, cache_dir=self.cache, offline=True) as reader:
            np.testing.assert_array_equal(reader[:], self.expected)
        self.assertEqual(self.requests, [])

    def test_lossless_delta3d_prefetch_is_readable_by_training(self):
        plan = inspect_level(self.level)
        stats = prefetch_level(plan, connections=3, batch_chunks=2, compress_cache=True)
        self.assertEqual(stats['chunks'], 3)
        metadata = json.loads((self.cached / '.zarray').read_text())
        self.assertEqual(metadata['compressor'], {'id': 'vc-delta3d', 'quant': 1})
        self.assertEqual((self.cached / '0/0/0').read_bytes()[:4], b'D3D1')
        self.requests.clear()
        with open_volume(self.url, 0, cache_dir=self.cache) as reader:
            np.testing.assert_array_equal(reader[:], self.expected)
        self.assertEqual(prefetch_level(plan, connections=2)['chunks'], 3)
        self.assertEqual(self.chunk_requests(), [])

    def test_download_failure_surfaces_and_retry_keeps_successful_chunks(self):
        self.fail.add('/volume.zarr/0/1/0/0')
        plan = inspect_level(self.level)
        with self.assertRaises(Exception):
            prefetch_level(plan, connections=1, batch_chunks=1)
        self.assertTrue((self.cached / '0/0/0').exists())
        self.fail.clear()
        self.requests.clear()
        self.assertEqual(prefetch_level(plan, connections=2)['chunks'], 3)
        self.assertNotIn('/volume.zarr/0/0/0/0', self.requests)


if __name__ == '__main__':
    unittest.main()
