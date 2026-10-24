"""Small upstream regression test: real scale CLI, generated synthetic inputs only.

Run with Villa's environment and PYTHONPATH=vesuvius/src:
    python patches/test_scale_input_routing.py
The default and selected-group tests fail on pinned unpatched Villa. No CT data,
weights, MetaCentrum paths, network access or fixture downloads are required.
"""
from __future__ import annotations

import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import zarr


class ScaleInputRoutingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "input.zarr"
        self.input.mkdir()
        (self.input / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        self.values = {}
        for level, shape in ((0, (7, 9, 11)), (2, (3, 5, 7))):
            values = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + level * 1000
            # This test targets format-2 stores. Support either installed Zarr API.
            version = {"zarr_format": 2} if int(zarr.__version__.split(".")[0]) >= 3 else {}
            array = zarr.open_array(str(self.input / str(level)), mode="w", shape=shape,
                                   chunks=(2, 3, 4), dtype=values.dtype, **version)
            array[:] = values
            self.values[level] = values

    def run_cli(self, input_path, factor, level=None):
        output = self.root / "output.zarr"
        command = [sys.executable, "-m", "vesuvius.image_proc.run.zarr_tasks", str(input_path),
                   str(output), "--task", "scale", "--scale-factor", str(factor),
                   "--num-levels", "1", "-n", "2"]
        if level is not None:
            command += ["--level", str(level)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, text=True, start_new_session=True)
        try:
            text, _ = process.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            self.fail("Scale CLI exceeded its 60-second regression-test bound")
        return process.returncode, text, output

    def assert_scaled(self, source, factor, result):
        code, log, output = result
        self.assertEqual(code, 0, log[-4000:])
        # Whole-array coordinate oracle, independent of worker chunk boundaries.
        shape = tuple(max(1, round(d * factor)) for d in source.shape)
        coordinates = [np.clip(np.floor(np.arange(n) / factor).astype(int), 0, d - 1)
                       for n, d in zip(shape, source.shape)]
        expected = source[np.ix_(*coordinates)]
        actual = zarr.open_array(str(output / "0"), mode="r")[:]
        self.assertEqual(actual.dtype, expected.dtype)
        np.testing.assert_array_equal(actual, expected)

    def test_group_default_level(self):
        self.assert_scaled(self.values[0], 1.0, self.run_cli(self.input, 1.0))

    def test_group_selected_nonzero_level(self):
        self.assert_scaled(self.values[2], 1.5, self.run_cli(self.input, 1.5, level=2))

    def test_direct_array_input(self):
        self.assert_scaled(self.values[2], 0.5, self.run_cli(self.input / "2", 0.5))

    def test_missing_level_is_rejected(self):
        code, log, _ = self.run_cli(self.input, 1.0, level=99)
        self.assertNotEqual(code, 0)
        self.assertIn("KeyError", log)
        self.assertIn("99", log)


if __name__ == "__main__":
    unittest.main(verbosity=2)
