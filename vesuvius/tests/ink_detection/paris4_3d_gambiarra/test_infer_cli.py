"""infer_volume end to end on small synthetic level-0 zarrs: with the real student network (CUDA + vesuvius), and on
CPU with the exact stand-in network of test_volume: done-marker verification, interrupted units, the RUN.json guard."""
import contextlib
import functools
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from vesuvius.ink_detection.paris4_3d_gambiarra import infer_volume as IV
from vesuvius.ink_detection.paris4_3d_gambiarra import volume as V

from .test_student import has_vesuvius
from .test_volume import cpu_codes, read_codes, stand_in, write_level0


@unittest.skipUnless(torch.cuda.is_available() and has_vesuvius(), 'needs CUDA and vesuvius')
class InferVolume(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'vesuvius.ink_detection.paris4_3d_gambiarra.infer_volume', *args], capture_output=True, text=True,
                              env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)))

    def test_run_and_resume(self):
        from vesuvius.ink_detection.paris4_3d_gambiarra import student as S
        rng = np.random.default_rng(3)
        vol = np.clip(rng.normal(90, 25, (384,) * 3), 0, 255).astype(np.uint8)
        present = np.ones((3, 3, 3), bool)
        present[0, 0, 0] = present[2, 1, 2] = False
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_level0(tmp / 'l0', vol, present)
            S.save_weights(S.build_network(27), tmp / 'w.safetensors')
            args = ['--weights', str(tmp / 'w.safetensors'), '--out', str(tmp / 'run'), '--ct', str(tmp / 'l0'),
                    '--unit', '128', '--chunk', '64', '--compressor', 'none', '--no-fsync']
            first = self.run_cli(*args)
            self.assertEqual(first.returncode, 0, first.stderr)
            array = tmp / 'run' / 'code.zarr' / '0'
            meta = json.loads((array / '.zarray').read_text())
            self.assertEqual((meta['shape'], meta['chunks'], meta['compressor']), ([192] * 3, [64] * 3, None))
            files = sorted(tuple(int(v) for v in f.relative_to(array).parts) for f in array.glob('*/*/*'))
            self.assertEqual(files, sorted(tuple(int(v) for v in k) for k in zip(*np.nonzero(present))))
            self.assertEqual(len(list((tmp / 'run' / 'state' / 'done').glob('*.json'))), 8)
            codes = read_codes(array, (192,) * 3, 64)
            inside = codes[32:160, 32:160, 32:160]
            mask = np.repeat(np.repeat(np.repeat(present, 64, 0), 64, 1), 64, 2)[32:160, 32:160, 32:160]
            self.assertTrue(np.all(inside[mask] > 0) and np.all(inside[~mask] == 0))
            self.assertEqual(int(codes[:32].max()), 0)  # outside crop coverage

            again = self.run_cli(*args)  # every unit is marked done: nothing recomputed
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertNotIn('unit ', again.stdout)
            np.testing.assert_array_equal(read_codes(array, (192,) * 3, 64), codes)

            changed = self.run_cli(*args, '--batch', '8')  # a different run in the same directory is refused
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn('different', changed.stderr)


class StandInStudent(nn.Module):
    """infer_volume's Student on CPU: test_volume's exact stand-in; non-finite logits while `broken` is set."""
    broken = False

    def __init__(self, seed=27):
        super().__init__()
        self.network = nn.Identity()

    def low(self, x):
        y = stand_in(x)
        return y * float('nan') if StandInStudent.broken else y


class InferVolumeOnCPU(unittest.TestCase):
    """main() with the stand-in network: 512 x 256 x 256 level-0 voxels (the last 128-voxel z slab absent), 64^3 units
    and chunks: 16 units, the four of the absent slab empty."""
    def setUp(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        self.addCleanup(torch.set_rng_state, torch.get_rng_state())
        self.addCleanup(torch.use_deterministic_algorithms, torch.are_deterministic_algorithms_enabled(),
                        warn_only=torch.is_deterministic_algorithms_warn_only_enabled())
        for module, name in ((torch.backends.cuda.matmul, 'allow_tf32'), (torch.backends.cudnn, 'benchmark'),
                             (torch.backends.cudnn, 'deterministic')):
            self.addCleanup(setattr, module, name, getattr(module, name))
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('torch.cuda.manual_seed_all'))
        stack.enter_context(patch.object(IV, 'Student', StandInStudent))
        stack.enter_context(patch.object(IV, 'load_weights', lambda network, path: network))
        stack.enter_context(patch.object(V, 'Engine', functools.partial(V.Engine, device='cpu')))
        stack.enter_context(patch.object(StandInStudent, 'broken', False))
        self.tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        rng = np.random.default_rng(4)
        shape0 = (512, 256, 256)
        vol = np.clip(rng.normal(90, 25, shape0), 0, 255).astype(np.uint8)
        present = np.ones((4, 2, 2), bool)
        present[3] = False
        vol[384:] = 0
        write_level0(self.tmp / 'l0', vol, present)
        self.shape1 = V.output_shape(shape0)
        self.expected = cpu_codes(vol, present)
        self.assertGreater(len(np.unique(self.expected)), 20)
        (self.tmp / 'w.safetensors').write_bytes(b'stand-in weights')
        self.array = self.tmp / 'run' / 'code.zarr' / '0'

    def cli(self, *extra, out='run'):
        argv = ['--weights', str(self.tmp / 'w.safetensors'), '--out', str(self.tmp / out), '--ct', str(self.tmp / 'l0'),
                '--unit', '64', '--chunk', '64', '--batch', '2', '--compressor', 'none', '--no-fsync', '--no-check',
                '--write-threads', '2', '--fetch-threads', '2', *extra]
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            IV.main(argv)
        return text.getvalue()

    def codes(self):
        return read_codes(self.array, self.shape1, 64)

    def chunk(self, key):
        return self.array.joinpath(*map(str, key))

    def test_interrupted_unit_then_marker_verification(self):
        # An interrupted first attempt: the first unit writes its chunks, then fails (no marker); leftovers of
        # another kind are planted: a temporary chunk file and a chunk file in a tile without present CT.
        with patch.object(StandInStudent, 'broken', True), self.assertRaisesRegex(RuntimeError, 'non-finite'):
            self.cli()
        self.assertTrue(self.chunk((0, 0, 0)).exists())
        self.assertFalse((self.tmp / 'run' / 'state' / 'done' / '0_0_0.json').exists())
        temporary = self.array / '0' / '0' / '.0.4242.part'
        temporary.write_bytes(b'partial')
        self.chunk((3, 1, 1)).parent.mkdir(parents=True)
        self.chunk((3, 1, 1)).write_bytes(bytes([200]) * 64 ** 3)

        log = self.cli()
        self.assertIn('done markers verified: 0; units computed: 16', log)
        np.testing.assert_array_equal(self.codes(), self.expected)
        self.assertFalse(temporary.exists() or self.chunk((3, 1, 1)).exists())
        markers = {p.stem: json.loads(p.read_text()) for p in (self.tmp / 'run' / 'state' / 'done').glob('*.json')}
        self.assertEqual(sorted(k for k, v in markers.items() if v['state'] == 'empty'), ['3_0_0', '3_0_1', '3_1_0', '3_1_1'])

        log = self.cli()  # everything verifies: nothing recomputed
        self.assertNotIn('unit ', log)
        self.assertIn('done markers verified: 16; units computed: 0', log)

        # Missing, same-size corrupted and unrecorded chunk files and an unreadable marker: only those units rerun.
        self.chunk((1, 0, 0)).unlink()
        data = bytearray(self.chunk((2, 1, 1)).read_bytes())
        data[1000] ^= 1
        self.chunk((2, 1, 1)).write_bytes(bytes(data))
        self.chunk((3, 0, 1)).parent.mkdir(parents=True, exist_ok=True)
        self.chunk((3, 0, 1)).write_bytes(bytes([7]) * 64 ** 3)
        (self.tmp / 'run' / 'state' / 'done' / '0_1_0.json').write_text('{"unit": "0_1')
        log = self.cli()
        self.assertIn('unit 1_0_0: done marker rejected (chunk 1 0 0 missing)', log)
        self.assertIn('unit 2_1_1: done marker rejected (chunk 2 1 1 differs from the record)', log)
        self.assertIn('unit 3_0_1: done marker rejected (chunk 3 0 1 not recorded)', log)
        self.assertIn('unit 0_1_0: done marker rejected (unreadable marker', log)
        self.assertIn('done markers verified: 12; units computed: 4', log)
        np.testing.assert_array_equal(self.codes(), self.expected)
        self.assertFalse(self.chunk((3, 0, 1)).exists())

        # A marker recording a different chunk set than the unit plan is not trusted either.
        marker = self.tmp / 'run' / 'state' / 'done' / '1_1_1.json'
        record = json.loads(marker.read_text())
        record['files'].pop('1 1 1')
        marker.write_text(json.dumps(record))
        (self.tmp / 'run' / 'state' / 'done' / '3_0_0.json').write_text('[]')
        log = self.cli()
        self.assertIn('unit 1_1_1: done marker rejected (the recorded chunks differ from the plan)', log)
        self.assertIn('unit 3_0_0: done marker rejected (malformed marker)', log)
        np.testing.assert_array_equal(self.codes(), self.expected)

    def test_changed_presence_reruns_affected_units(self):
        # A saved presence map, later replaced at the same path: level-0 chunk (2, 1, 1) is now treated as absent.
        presence = np.ones((4, 2, 2), bool)
        presence[3] = False
        np.save(self.tmp / 'presence.npy', presence)
        self.cli('--presence', str(self.tmp / 'presence.npy'))
        np.testing.assert_array_equal(self.codes(), self.expected)
        presence[2, 1, 1] = False
        np.save(self.tmp / 'presence.npy', presence)
        log = self.cli('--presence', str(self.tmp / 'presence.npy'))
        rejected = sorted(line.split(':')[0] for line in log.splitlines() if 'the CT presence of the unit changed' in line)
        self.assertTrue(rejected and len(rejected) < 16, rejected)
        vol = np.clip(np.random.default_rng(4).normal(90, 25, (512, 256, 256)), 0, 255).astype(np.uint8)
        vol[384:] = 0
        vol[256:384, 128:, 128:] = 0
        np.testing.assert_array_equal(self.codes(), cpu_codes(vol, presence))
        # Markers of earlier versions (no presence record) are not trusted.
        marker = self.tmp / 'run' / 'state' / 'done' / '0_0_0.json'
        record = json.loads(marker.read_text())
        del record['presence_sha256']
        marker.write_text(json.dumps(record))
        self.assertIn('unit 0_0_0: done marker rejected (the marker records no CT presence',
                      self.cli('--presence', str(self.tmp / 'presence.npy')))

    def test_shards_share_chunk_directories(self):
        # Shard 0 takes the x = 0 units, shard 1 the x = 1 units: their chunks share every z/y directory.
        self.assertIn('units computed: 8', self.cli('--shard', '1/2'))
        self.assertIn('units computed: 8', self.cli('--shard', '0/2'))
        np.testing.assert_array_equal(self.codes(), self.expected)
        self.assertIn('done markers verified: 8; units computed: 0', self.cli('--shard', '1/2'))
        self.assertIn('done markers verified: 16; units computed: 0', self.cli())

    def test_changed_settings_are_refused(self):
        self.cli('--max-units', '1')
        before = {p: p.read_bytes() for p in (self.tmp / 'run').rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ValueError, r"different \['batch'\]"):
            self.cli('--batch', '1')
        (self.tmp / 'w.safetensors').write_bytes(b'other weights')
        with self.assertRaisesRegex(ValueError, r"different \['weights_sha256'\]"):
            self.cli()
        self.assertEqual({p: p.read_bytes() for p in (self.tmp / 'run').rglob('*') if p.is_file()}, before)

    def test_concurrent_start_with_other_settings_writes_nothing(self):
        # Another process published its RUN.json first (between our start and our metadata writes).
        run = self.tmp / 'race'
        run.mkdir()
        (run / 'RUN.json').write_text(json.dumps(dict(unit=128)))
        with self.assertRaisesRegex(ValueError, 'different'):
            self.cli(out='race')
        self.assertEqual(sorted(p.relative_to(run).as_posix() for p in run.rglob('*')),
                         ['RUN.json', 'state', 'state/done'])


if __name__ == '__main__':
    unittest.main()
