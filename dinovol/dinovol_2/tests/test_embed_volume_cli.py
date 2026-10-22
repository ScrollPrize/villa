"""CPU regressions: python -m unittest discover -s dinovol_2/tests -p 'test_embed_volume_cli.py'."""
import copy
import json
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from dinovol_2.eval import embed_volume as ev


class ArrayReader:
    def __init__(self):
        self.shape, self.dtype, self.chunks = (17, 13, 15), np.dtype('uint8'), (4, 4, 4)
        self.data = np.random.default_rng(0).integers(1, 200, self.shape, dtype=np.uint8)
        self.data[:, :, :6] = 0

    def read_region(self, start, size):
        out = np.zeros(size, self.dtype)
        lo = [max(0, s) for s in start]
        hi = [min(s + n, d) for s, n, d in zip(start, size, self.shape)]
        if all(b > a for a, b in zip(lo, hi)):
            out[tuple(slice(a - s, b - s) for a, b, s in zip(lo, hi, start))] = self.data[
                tuple(slice(a, b) for a, b in zip(lo, hi))]
        return out


class CpuPipeline:
    def __init__(self, *args, **kwargs):
        pass

    def batch(self, data, **kwargs):
        return {'image': data['image'].float()}


class TinyBackbone(torch.nn.Module):
    embed_dim = 3

    def forward_features(self, tile, **kwargs):
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        pooled = F.avg_pool3d(tile.float(), 2).flatten(2).transpose(1, 2)
        # Context-dependent tokens exercise blending across tile and slab boundaries.
        return {'x_norm_patchtokens': torch.cat((pooled, pooled * 0.3 + tile.mean(), -pooled * 0.6), -1),
                'embed_skips': []}


class TinyHead(torch.nn.Module):
    out_dim, ink_dim = 4, 2

    def __init__(self):
        super().__init__()
        self.fpn_norm = torch.nn.LayerNorm(3)
        self.obj_proj = torch.nn.Linear(3, self.out_dim)
        self.ink_proj = torch.nn.Linear(3, self.ink_dim)

    def fpn_grid(self, tokens, skips):
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        return tokens * 0.5

    def readout(self, tokens, skips, fpn, *, dtype):
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        values = tokens.to(dtype) + fpn.to(dtype)
        return {'dense': F.normalize(self.obj_proj(values), dim=-1),
                'ink': F.normalize(self.ink_proj(values), dim=-1)}


class VolumeCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.reader = ArrayReader()
        self.loaded = SimpleNamespace(backbone=TinyBackbone(), dense_head=None, device=torch.device('cpu'),
                                      patch_size=(2, 2, 2), embedding_type='default', global_input_size=(12,) * 3,
                                      normalization_scheme='none', intensity_properties=None, source_branch='teacher')
        for target, replacement in [('A.Pipeline', CpuPipeline), ('_configure_backbone_for_spatial_shape', lambda *a: None)]:
            p = patch.object(ev.A, 'Pipeline', replacement) if target == 'A.Pipeline' else patch.object(ev, target, replacement)
            p.start()
            self.addCleanup(p.stop)

    def embedder(self):
        return ev.RegionEmbedder(self.reader, self.loaded, transforms=[], outputs=('tokens', 'tokens_unit'),
                                 dense_stride=1, lattice=[ev.AxisLattice(12, 2, 2)] * 3,
                                 codec='vc-delta3d')

    def metadata(self, embedder, z_range=(0, 17)):
        outputs = {}
        for name, spec, shape, chunks in embedder.arrays(z_range, z_cell=True):
            outputs[name] = dict(shape_zyxc=list(shape), chunks_zyxc=list(chunks), step_zyx=list(spec.step),
                                 dtype=embedder.dtypes[name], codec=embedder.codecs[name])
        return dict(z_range=list(z_range), y_range=None, x_range=None, outputs=outputs,
                    slabs=[list(s) for s in ev.plan_slabs(z_range, 12)], checkpoint_sha256='original')

    def new_sink(self, path, embedder, attrs):
        sink = ev.ZarrGroupWriter(path, attrs=attrs)
        embedder.create_arrays(sink, tuple(attrs['z_range']), z_cell=True)
        return sink

    def test_previous_column_accumulators_are_released_before_next_column(self):
        embedder = self.embedder()
        attrs = self.metadata(embedder)
        sink = self.new_sink(self.root / 'released', embedder, attrs)
        self.addCleanup(sink.close)
        original = embedder._embed_column
        previous = []

        def checked(*args, **kwargs):
            self.assertTrue(all(ref() is None for ref in previous))
            acc, valid = original(*args, **kwargs)
            previous[:] = [weakref.ref(tensor) for tensor in acc.values()] if acc else []
            return acc, valid

        with patch.object(embedder, '_embed_column', side_effect=checked):
            ev.run_slabs(embedder, sink, attrs, resume=False)
        self.assertTrue(all(ref() is None for ref in previous))

    def test_column_blocks_preserve_all_outputs_and_bound_row_strips(self):
        self.loaded.embedding_type = 'deeper'
        self.loaded.dense_head = TinyHead()
        self.reader.data[:, 4:7, 8:10] = 0
        embedder = ev.RegionEmbedder(
            self.reader, self.loaded, transforms=[], outputs=ev.OUTPUT_NAMES,
            dense_stride=1, lattice=[ev.AxisLattice(12, 2, 2)] * 3,
            codec='vc-delta3d', point_chunk_size=128)
        for read in ('band', 'column'):
            for width in (0, 1, 2):
                embedder.column_block_size = width
                attrs = self.metadata(embedder)
                attrs['x_range'], attrs['y_range'] = [1, 14], [1, 13]
                sink = self.new_sink(self.root / f'{read}{width}', embedder, attrs)
                original = embedder._finish_column

                def checked(*args):
                    original(*args)
                    next_top = args[10]
                    if width:
                        # One left halo column and one trailing corner slot.
                        self.assertLessEqual(len(next_top), width + 2)

                with patch.object(embedder, '_finish_column', side_effect=checked):
                    ev.run_slabs(embedder, sink, attrs, resume=False, slab_group_size=2)
                sink.close()
                if width:
                    for name in embedder.outputs:
                        np.testing.assert_array_equal(
                            ev.read_embeddings(self.root / f'{read}0', name),
                            ev.read_embeddings(sink.path, name))

    def test_column_block_failure_does_not_commit_group_and_can_resume_with_new_width(self):
        embedder = self.embedder()
        embedder.column_block_size = 1
        attrs = self.metadata(embedder)
        sink = self.new_sink(self.root / 'block_failure', embedder, attrs)
        original = embedder._run_columns
        calls = 0

        def fail_second(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('block failed')
            return original(*args, **kwargs)

        with patch.object(embedder, '_run_columns', side_effect=fail_second), \
             self.assertRaisesRegex(RuntimeError, 'block failed'):
            ev.run_slabs(embedder, sink, attrs, resume=False, slab_group_size=2)
        sink.close()
        saved = json.loads((sink.path / '.zattrs').read_text())
        self.assertFalse(saved.get('completed_slabs'))
        embedder.column_block_size = 2
        sink = ev.ZarrGroupWriter(sink.path, attrs=None)
        ev.run_slabs(embedder, sink, saved, resume=True, slab_group_size=2)
        sink.close()
        self.assertTrue(saved['complete'])
        reference = self.new_sink(self.root / 'block_reference', embedder, self.metadata(embedder))
        ev.run_slabs(embedder, reference, self.metadata(embedder), resume=False, slab_group_size=2)
        reference.close()
        for name in embedder.outputs:
            np.testing.assert_array_equal(ev.read_embeddings(sink.path, name),
                                          ev.read_embeddings(reference.path, name))

    def test_native_normalization_oom_reclaims_cache_once(self):
        embedder = self.embedder()
        embedder.device = torch.device('cuda')
        raw = torch.ones((2, 3, 4), dtype=torch.uint8)
        output = raw.float()[None, None]
        oom = RuntimeError('CUDA transform: out of memory')
        with patch.object(embedder.pipeline, 'batch', side_effect=[oom, {'image': output}]) as batch, \
             patch.object(torch.cuda, 'device'), patch.object(torch.cuda, 'empty_cache') as reclaim:
            self.assertTrue(torch.equal(embedder._normalize_column(raw), raw.float()))
            self.assertEqual(batch.call_count, 2)
            reclaim.assert_called_once_with()
        for device, error, attempts in [
            ('cuda', oom, 2), ('cpu', oom, 1),
            ('cuda', RuntimeError('CUDA transform: illegal memory access'), 1),
            ('cuda', torch.OutOfMemoryError('PyTorch allocation failed'), 1),
        ]:
            embedder.device = torch.device(device)
            with self.subTest(device=device, error=str(error)), \
                 patch.object(embedder.pipeline, 'batch', side_effect=error) as batch, \
                 patch.object(torch.cuda, 'device'), patch.object(torch.cuda, 'empty_cache') as reclaim:
                with self.assertRaises(type(error)):
                    embedder._normalize_column(raw)
                self.assertEqual(batch.call_count, attempts)
                self.assertEqual(reclaim.call_count, attempts - 1)

    def test_compiles_called_methods_and_disables_gradients(self):
        self.loaded.embedding_type = 'deeper'
        self.loaded.dense_head = TinyHead().train()
        self.loaded.backbone.train()
        graphs = []

        def backend(graph, inputs):
            graphs.append(graph)
            return graph.forward

        real_compile = torch.compile

        def compile_for_test(fn, **options):
            self.assertEqual(options, dict(backend='inductor', mode='default', dynamic=False))
            # Exercise Dynamo tracing and compiled dispatch without a platform-specific code generator.
            return real_compile(fn, backend=backend, dynamic=False)

        with patch.object(ev.torch, 'compile', side_effect=compile_for_test) as compiler:
            embedder = ev.RegionEmbedder(self.reader, self.loaded, transforms=[],
                                         outputs=('tokens', 'dense', 'ink'), dense_stride=2,
                                         lattice=[ev.AxisLattice(12, 2, 2)] * 3, point_chunk_size=128,
                                         compile_model=True)
        self.assertEqual([call.args[0].__name__ for call in compiler.call_args_list],
                         ['forward_features', 'fpn_grid', 'readout'])
        for model in (self.loaded.backbone, self.loaded.dense_head):
            self.assertFalse(model.training)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
        attrs = self.metadata(embedder)
        sink = self.new_sink(self.root / 'compiled', embedder, attrs)
        ev.run_slabs(embedder, sink, attrs, resume=False)
        sink.close()
        self.assertGreaterEqual(len(graphs), 3)
        self.assertGreater(np.abs(ev.read_embeddings(sink.path, 'dense')).sum(), 0)
        self.assertTrue(torch.is_grad_enabled())  # inference context does not leak to the caller
        self.assertTrue(ev.parse_args(['--checkpoint', 'c', '--volume', 'v', '--output', 'o']).compile)

    def test_streamed_matches_single_region_with_partial_edges(self):
        embedder = self.embedder()
        for z_range in [(0, 17), (2, 15)]:
            with self.subTest(z_range=z_range):
                attrs = self.metadata(embedder, z_range)
                full, streamed = self.root / f'full{z_range[0]}', self.root / f'streamed{z_range[0]}'
                sink = self.new_sink(full, embedder, attrs)
                embedder.run(sink, z_range=(z_range[0], z_range[0] + 24), array_z_range=z_range, progress=False)
                sink.close()
                sink = self.new_sink(streamed, embedder, attrs)
                ev.run_slabs(embedder, sink, attrs, resume=False, read='column')
                sink.close()
                for name in embedder.outputs:
                    np.testing.assert_array_equal(ev.read_embeddings(full, name), ev.read_embeddings(streamed, name))
                    meta = json.loads((streamed / name / '.zarray').read_text())
                    self.assertEqual(meta['compressor']['id'], 'vc-delta3d')
                    self.assertEqual(meta['shape'][0], (z_range[1] - z_range[0] + 1) // 2)

    def test_interrupted_resume_removes_partial_chunks_and_keeps_completed_slab(self):
        embedder = self.embedder()
        expected = self.metadata(embedder)
        attrs = copy.deepcopy(expected)
        path = self.root / 'resume'
        sink = self.new_sink(path, embedder, attrs)
        real_run = embedder.run

        def interrupted(sink, **kwargs):
            if kwargs['z_range'][0] == 12:
                # Simulate a partially written slab, including data in a masked chunk.
                sink.write('tokens', (6, 0, 0), np.ones((3, 3, 3, 3), np.int8))
                sink.write('tokens_scale', (6, 0, 0), np.ones((3, 3, 3), np.float16))
                raise RuntimeError('interrupted')
            return real_run(sink, **kwargs)

        with patch.object(embedder, 'run', interrupted), self.assertRaisesRegex(RuntimeError, 'interrupted'):
            ev.run_slabs(embedder, sink, attrs, resume=False)
        sink.close()
        saved = json.loads((path / '.zattrs').read_text())
        self.assertEqual(list(saved['completed_slabs']), ['0'])
        ev.validate_resume(saved, expected, path)
        committed = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (path / 'tokens/0').rglob('*') if p.is_file()}
        sink = ev.ZarrGroupWriter(path, attrs=None)
        with patch.object(embedder, 'run', wraps=real_run) as run:
            ev.run_slabs(embedder, sink, saved, resume=True, slab_group_size=2)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs['z_range'], (12, 24))
        sink.close()
        for p, content in committed.items():
            self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), content)
        self.assertFalse((path / 'tokens/2/0/0/0').exists())
        ref = self.root / 'reference'
        attrs = copy.deepcopy(expected)
        sink = self.new_sink(ref, embedder, attrs)
        ev.run_slabs(embedder, sink, attrs, resume=False)
        sink.close()
        for name in embedder.outputs:
            np.testing.assert_array_equal(ev.read_embeddings(path, name), ev.read_embeddings(ref, name))
        sink = ev.ZarrGroupWriter(path, attrs=None)
        with patch.object(embedder, 'run', side_effect=AssertionError('completed run must be skipped')):
            ev.run_slabs(embedder, sink, saved, resume=True)
        sink.close()

    def test_grouped_slabs_reuse_tiles_and_match_partial_output(self):
        embedder = self.embedder()
        totals = []
        for group_size in (1, 2):
            attrs = self.metadata(embedder)
            sink = self.new_sink(self.root / f'group{group_size}', embedder, attrs)
            totals.append(ev.run_slabs(embedder, sink, attrs, resume=False, slab_group_size=group_size))
            sink.close()
            self.assertEqual(set(attrs['completed_slabs']), {'0', '1'})
            ev.validate_resume(attrs, self.metadata(embedder), sink.path)
        self.assertLess(totals[1]['tiles'], totals[0]['tiles'])
        for name in embedder.outputs:
            np.testing.assert_array_equal(ev.read_embeddings(self.root / 'group1', name),
                                          ev.read_embeddings(self.root / 'group2', name))
        self.assertEqual(attrs['slab_runtime']['1']['group_slabs'], [0, 1])
        self.assertEqual(attrs['completed_slabs']['1'], {})

    def test_interrupted_group_is_recomputed_and_committed_together(self):
        embedder = self.embedder()
        attrs = self.metadata(embedder)
        sink = self.new_sink(self.root / 'group_failed', embedder, attrs)
        real_run = embedder.run

        def interrupted(sink, **kwargs):
            real_run(sink, **kwargs)
            raise RuntimeError('group interrupted')

        with patch.object(embedder, 'run', interrupted), self.assertRaisesRegex(RuntimeError, 'group interrupted'):
            ev.run_slabs(embedder, sink, attrs, resume=False, slab_group_size=2)
        sink.close()
        saved = json.loads((sink.path / '.zattrs').read_text())
        self.assertFalse(saved.get('completed_slabs'))
        sink = ev.ZarrGroupWriter(sink.path, attrs=None)
        with patch.object(embedder, 'run', wraps=real_run) as run:
            ev.run_slabs(embedder, sink, saved, resume=True, slab_group_size=2)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.kwargs['z_range'], (0, 24))
        sink.close()
        self.assertEqual(set(saved['completed_slabs']), {'0', '1'})
        self.assertTrue(saved['complete'])

    def test_group_planning_does_not_cross_completed_slabs_or_gaps(self):
        slabs = [(0, 12), (12, 24), (24, 36), (36, 48), (60, 72)]
        self.assertEqual(list(ev.pending_slab_groups(slabs, {'1': {}}, 2)), [[0], [2, 3], [4]])
        with self.assertRaisesRegex(ValueError, 'positive'):
            list(ev.pending_slab_groups(slabs, {}, 0))

    def test_async_write_failure_does_not_commit_slab(self):
        embedder = self.embedder()
        attrs = self.metadata(embedder)
        sink = self.new_sink(self.root / 'failed', embedder, attrs)
        with patch.object(sink, '_store', side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                ev.run_slabs(embedder, sink, attrs, resume=False, slab_group_size=2)
            with self.assertRaises(OSError):
                sink.close()
        self.assertFalse(json.loads((sink.path / '.zattrs').read_text()).get('completed_slabs'))

    def test_all_masked_slabs_are_committed_without_chunk_files(self):
        self.reader.data[:] = 0
        embedder = self.embedder()
        attrs = self.metadata(embedder)
        sink = self.new_sink(self.root / 'empty', embedder, attrs)
        ev.run_slabs(embedder, sink, attrs, resume=False)
        sink.close()
        self.assertEqual(set(attrs['completed_slabs']), {'0', '1'})
        self.assertTrue(attrs['complete'])
        self.assertFalse(list((sink.path / 'tokens').glob('*/*/*/0')))
        sink = ev.ZarrGroupWriter(sink.path, attrs=None)
        with patch.object(embedder, 'run', side_effect=AssertionError('empty slabs must be skipped')):
            ev.run_slabs(embedder, sink, attrs, resume=True)
        sink.close()

    def test_resume_validates_checkpoint_plan_and_array_schema(self):
        embedder = self.embedder()
        expected = self.metadata(embedder)
        saved = dict(expected, completed_slabs={})
        path = self.root / 'validate'
        sink = self.new_sink(path, embedder, saved)
        sink.close()
        ev.validate_resume(saved, expected, path)
        for key, value in [('checkpoint_sha256', 'different'), ('slabs', [[0, 24]]), ('z_range', [0, 16])]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                ev.validate_resume(saved, dict(expected, **{key: value}), path)
        meta_path = path / 'tokens/.zarray'
        meta = json.loads(meta_path.read_text())
        meta['chunks'][0] = 1
        meta_path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'storage metadata'):
            ev.validate_resume(saved, expected, path)

    def test_cache_path_resolves_remote_and_enables_fetching(self):
        source = 's3://bucket/scroll/volume.zarr'
        local = self.root / 'cache/s3/bucket/scroll/volume.zarr'
        (local / '2').mkdir(parents=True)
        (local / '2/.zattrs').write_text(json.dumps({'vc_reader': {'source_url': source + '/2'}}))
        self.assertEqual(ev.resolve_volume_source(str(local), 2, None), (source, str(self.root / 'cache')))
        args = ev.parse_args(['--volume', str(local), '--level', '2', '--checkpoint', 'checkpoint',
                              '--output', str(self.root / 'out')])
        with patch.object(ev, 'open_volume') as opener, patch.object(ev, '_embed_volume'):
            ev.embed_volume(args)
            self.assertEqual(opener.call_args.args, (source,))
            self.assertEqual(opener.call_args.kwargs['cache_dir'], str(self.root / 'cache'))
            self.assertFalse(opener.call_args.kwargs['offline'])
            self.assertTrue(opener.call_args.kwargs['cache_writes'])

    def test_main_reuses_statistics_and_rejects_changed_checkpoint(self):
        checkpoint = self.root / 'model.pt'
        torch.save({'step': 10}, checkpoint)
        stats = self.root / 'stats.json'
        stats.write_text(json.dumps({'mean': 20., 'std': 10.}))
        self.loaded.normalization_scheme = 'zscore'
        argv = ['--checkpoint', str(checkpoint), '--volume', 's3://bucket/volume.zarr', '--level', '2',
                '--output', str(self.root / 'main'), '--outputs', 'tokens,tokens_unit',
                '--codec', 'vc-delta3d', '--no-compile', '--normalization-stats', str(stats)]
        with patch.object(ev, 'load_backbone_from_checkpoint', return_value=self.loaded):
            ev._embed_volume(ev.parse_args(argv), self.reader, 's3://bucket/volume.zarr', None)
            attrs = json.loads((self.root / 'main/.zattrs').read_text())
            self.assertTrue(attrs['complete'])
            self.assertEqual(attrs['z_range'], [0, 17])
            self.assertFalse(attrs['execution']['compile'])
            changed = copy.deepcopy(attrs)
            changed['execution']['compile'] = True
            expected = {k: v for k, v in attrs.items() if k not in ('completed_slabs', 'complete', 'run')}
            with self.assertRaisesRegex(ValueError, 'execution'):
                ev.validate_resume(changed, expected, self.root / 'main')
            # No statistics file is needed on resume: the saved statistics are authoritative.
            argv = argv[:-2] + ['--resume']
            with patch.object(ev, 'sample_volume_histogram', side_effect=AssertionError('resampled')):
                ev._embed_volume(ev.parse_args(argv), self.reader, 's3://bucket/volume.zarr', attrs)
            torch.save({'step': 10, 'changed': True}, checkpoint)
            with self.assertRaisesRegex(ValueError, 'checkpoint_sha256'):
                ev._embed_volume(ev.parse_args(argv), self.reader, 's3://bucket/volume.zarr', attrs)

    def test_overwrite_removes_old_chunks_and_lock_prevents_concurrent_cli(self):
        path = self.root / 'overwrite'
        sink = ev.ZarrGroupWriter(path, attrs={})
        sink.create('old', (2, 2, 2, 1), (2, 2, 2, 1), np.int8)
        sink.write('old', (0, 0, 0), np.ones((2, 2, 2, 1), np.int8))
        sink.close()
        sink = ev.ZarrGroupWriter(path, attrs={}, overwrite=True)
        sink.close()
        self.assertFalse((path / 'old').exists())
        with ev.output_lock(path):
            with self.assertRaisesRegex(RuntimeError, 'another embedding process'):
                with ev.output_lock(path):
                    pass
        with ev.output_lock(path):
            pass


if __name__ == '__main__':
    unittest.main()
