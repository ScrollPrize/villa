"""Portable entrypoints and output safety, using an explicit synthetic fixture."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from vc_sampling_audit.provenance import code_sha256


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cwd = Path(self.temp.name)
        self.scripts = Path(__file__).resolve().parents[2]
        self.capture = self.cwd / "grid.json"
        self.out = self.cwd / "report.json"
        self.data = {
            "schema": "vc-sampling-grid-v1",
            "stage": "final-base-and-dirs-before-readMultiSlice",
            "pixel_order": "row-major-baseXYZ-directionXYZ",
            "units": "level-index-voxel",
            "between_pixel_interpolation": "NOT_SPECIFIED_BY_RENDERER",
            "shape_hw": [2, 2], "level": 0, "crop_xy": [0, 0],
            "offsets": [-1, 1], "invalid_pixels": 0,
            "pixels": [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
                       [0, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1]],
        }
        self.capture.write_text(json.dumps(self.data))

    def invoke(self, *, module=False, reference=(0, 0, 1)):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        if module:
            env["PYTHONPATH"] = str(self.scripts)
            command = [sys.executable, "-m", "vc_sampling_audit"]
        else:
            # -E proves the wrapper does not rely on workspace PYTHONPATH.
            command = [sys.executable, "-E", "-B",
                       str(self.scripts / "vc_sampling_grid_audit.py")]
        return subprocess.run(
            command + ["--capture", str(self.capture), "--reference",
                       *map(str, reference), "--out", str(self.out)],
            cwd=self.cwd, env=env, text=True, capture_output=True, timeout=30)

    def test_script_runs_from_unrelated_directory(self):
        proc = self.invoke()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        report = json.loads(self.out.read_text())
        self.assertEqual(report["status"], "LOCAL_SAMPLING_DIAGNOSTIC_ONLY")
        self.assertEqual(report["anatomical_sheet_identity"], "NOT_ASSESSED")

    def test_module_entrypoint(self):
        proc = self.invoke(module=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(self.out.read_text())["code_sha256"], code_sha256())

    def test_provenance_hashes_actual_shared_files(self):
        proc = self.invoke()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        observed = json.loads(self.out.read_text())["code_sha256"]
        self.assertEqual(len(observed), 7)
        for name, value in observed.items():
            self.assertFalse(Path(name).is_absolute())
            self.assertEqual(value, hashlib.sha256((self.scripts / name).read_bytes()).hexdigest())

    def test_existing_report_never_overwritten(self):
        self.out.write_bytes(b"existing evidence")
        proc = self.invoke()
        self.assertNotEqual(proc.returncode, 0)
        self.assertEqual(self.out.read_bytes(), b"existing evidence")

    def test_invalid_capture_does_not_create_report(self):
        self.data["invalid_pixels"] = 1
        self.capture.write_text(json.dumps(self.data))
        proc = self.invoke()
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(self.out.exists())

    def test_wrong_chart_reference_refused(self):
        proc = self.invoke(reference=(0, 0, -1))
        self.assertNotEqual(proc.returncode, 0)
        self.assertFalse(self.out.exists())

    def test_input_bytes_unchanged(self):
        before = self.capture.read_bytes()
        proc = self.invoke()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(self.capture.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
