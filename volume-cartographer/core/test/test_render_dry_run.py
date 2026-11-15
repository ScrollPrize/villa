from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


RENDERER = str(Path(sys.argv.pop(1)).resolve())
FIXTURE = Path(__file__).parent / 'data/render/dry-run-PHerc0800'


class RenderDryRunTests(unittest.TestCase):
    def run_plan(self, *options):
        return subprocess.run([
            RENDERER, '--volume', str(FIXTURE / 'volume.zarr'),
            '--segmentation', str(FIXTURE / 'surface.tifxyz'),
            '--scale', '1', '--group-idx', '1', '--num-slices', '8',
            '--cache-gb', '1', '--flip-normals', '--dry-run', *options,
        ], capture_output=True, text=True, timeout=30)

    def count(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        return int(re.search(r'Referenced source chunks: (\d+)', result.stdout).group(1))

    def test_real_surface_without_ct_chunks_or_output(self):
        before = {p.relative_to(FIXTURE): p.read_bytes() for p in FIXTURE.rglob('*') if p.is_file()}
        result = self.run_plan()
        self.assertEqual(self.count(result), 20)
        self.assertIn('Full decoded chunk bytes: 41943040', result.stdout)
        self.assertEqual({p.relative_to(FIXTURE): p.read_bytes() for p in FIXTURE.rglob('*') if p.is_file()}, before)

    def test_output_arguments_do_not_create_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_plan('--tif-output', str(root / 'tiff'), '--zarr-output', str(root / 'zarr'))
            self.assertEqual(self.count(result), 20)
            self.assertEqual(list(root.iterdir()), [])

    def test_rotating_full_canvas_keeps_chunk_set_size(self):
        self.assertEqual(self.count(self.run_plan('--rotate', '90')), 20)

    def test_crop_uses_fewer_chunks(self):
        result = self.run_plan('--crop-x', '100', '--crop-y', '120', '--crop-width', '160', '--crop-height', '170', '--rotate', '90')
        count = self.count(result)
        self.assertEqual(count, 8)

    def test_zarr_pyramid_partition_matches_render_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output.zarr'
            options = ('--num-parts', '2', '--part-id', '1', '--zarr-output', str(output))
            self.assertEqual(self.count(self.run_plan(*options)), 0)
            self.assertGreater(self.count(self.run_plan(*options, '--pyramid', 'false')), 0)
            self.assertFalse(output.exists())

    def test_unsupported_modes_are_rejected(self):
        for options in [('--pre',), ('--resume',), ('--merge-tiff-parts',), ('--cuda',), ('--composite-collapse',), ('--accum-type', 'alpha')]:
            with self.subTest(options=options):
                result = self.run_plan(*options)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('--dry-run', result.stderr)


if __name__ == '__main__':
    unittest.main()
