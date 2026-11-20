from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


RENDERER = str(Path(sys.argv.pop(1)).resolve())
FIXTURE = Path(__file__).parent / 'data/render/PHerc0800'


class TiffMergeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='vc tiff merge ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def add_part(self, name, part):
        shutil.copy2(FIXTURE / f'00.part{part}.tif', self.root / name)

    def snapshot(self):
        return {path.name: path.read_bytes() for path in self.root.iterdir()}

    def merge(self):
        return subprocess.run([
            RENDERER, '--merge-tiff-parts', '--num-parts', '2',
            '--tif-output', str(self.root), '--scale', '1', '--group-idx', '0',
        ], capture_output=True, text=True, timeout=15)

    def assert_incomplete_preserves_files(self, missing):
        before = self.snapshot()
        result = self.merge()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(f'Missing TIFF part: {missing}', result.stderr)
        self.assertEqual(self.snapshot(), before)

    def test_missing_part_preserves_input_and_existing_output(self):
        self.add_part('00.part0.tif', 0)
        self.add_part('00.tif', 1)
        self.assert_incomplete_preserves_files('00.part1.tif')

    def test_later_incomplete_slice_preserves_all_slices(self):
        self.add_part('00.part0.tif', 0)
        self.add_part('00.part1.tif', 1)
        self.add_part('01.part0.tif', 0)
        self.assert_incomplete_preserves_files('01.part1.tif')

    def test_matching_count_with_wrong_part_number_is_rejected(self):
        self.add_part('00.part0.tif', 0)
        self.add_part('00.part2.tif', 1)
        self.assert_incomplete_preserves_files('00.part1.tif')

    def test_complete_parts_are_merged(self):
        self.add_part('00.part0.tif', 0)
        self.add_part('00.part1.tif', 1)
        result = self.merge()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual({p.name for p in self.root.iterdir()}, {'00.tif'})
        self.assertGreater((self.root / '00.tif').stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
