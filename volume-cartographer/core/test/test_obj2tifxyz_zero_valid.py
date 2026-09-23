"""Regression for issue #1320 using the real vc_obj2tifxyz CLI.

Run with:
    python3 test_obj2tifxyz_zero_valid.py /path/to/vc_obj2tifxyz

No scroll download is required. The fixture reproduces the normalized-[0,1] UV
failure mode deterministically, then verifies that an explicit sampling density
still converts successfully.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


OBJ2TIFXYZ = str(Path(sys.argv.pop(1)).resolve())


def write_normalized_uv_obj(path: Path) -> None:
    # Four small triangles touch the global [0,1] UV bounds independently but
    # none contains a corner of the default 2x2 sample grid.
    uv = [
        (0.00, 0.45), (0.10, 0.40), (0.10, 0.50),
        (1.00, 0.45), (0.90, 0.40), (0.90, 0.50),
        (0.45, 0.00), (0.40, 0.10), (0.50, 0.10),
        (0.45, 1.00), (0.40, 0.90), (0.50, 0.90),
    ]
    with path.open("w", encoding="utf-8") as f:
        for i, (u, v) in enumerate(uv):
            f.write(f"v {u * 100.0} {v * 100.0} {10.0 + i * 0.01}\n")
        for u, v in uv:
            f.write(f"vt {u} {v}\n")
        for tri in range(4):
            a = tri * 3 + 1
            f.write(f"f {a}/{a} {a + 1}/{a + 1} {a + 2}/{a + 2}\n")


class Obj2TifxyzZeroValidTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="vc obj2tifxyz zero ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.obj = self.root / "normalized.obj"
        write_normalized_uv_obj(self.obj)

    def run_converter(self, output: Path, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [OBJ2TIFXYZ, str(self.obj), str(output), *extra],
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_zero_valid_default_fails_without_writing_fake_surface(self):
        output = self.root / "empty"
        result = self.run_converter(output)
        log = result.stdout + result.stderr

        self.assertNotEqual(result.returncode, 0, log)
        self.assertIn("Valid grid points: 0 / 4", log)
        self.assertIn("refusing to save an empty tifxyz", log)
        self.assertIn("stretch_factor", log)
        self.assertNotIn("Successfully converted to tifxyz format", log)
        for name in ("x.tif", "y.tif", "z.tif"):
            self.assertFalse((output / name).exists(), f"unexpected output: {name}")

    def test_same_mesh_succeeds_with_explicit_sampling_density(self):
        output = self.root / "sampled"
        result = self.run_converter(output, "20", "1.0")
        log = result.stdout + result.stderr

        self.assertEqual(result.returncode, 0, log)
        self.assertIn("Successfully converted to tifxyz format", log)
        for name in ("x.tif", "y.tif", "z.tif"):
            self.assertTrue((output / name).is_file(), f"missing output: {name}")


if __name__ == "__main__":
    unittest.main()
