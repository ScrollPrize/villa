from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dinovol_2 import bench_speed


class BenchmarkLaunchTests(unittest.TestCase):
    def test_launches_preserve_source_and_previous_benchmarks_even_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source run"
            source.mkdir()
            (source / "config.json").write_text(json.dumps({
                "output_dir": str(source), "resume_from": "wrong-checkpoint.pt",
                "model": {}, "dataset": {"object_points": {"slab_thickness": 8}},
            }))
            for name in ("checkpoint_step_150000.pt", "checkpoint_step_160000.pt",
                         "checkpoint_step_170000.pt", "train.log", "train.pid"):
                (source / name).write_bytes(name.encode())
            original = {path.name: path.read_bytes() for path in source.iterdir()}
            bench_root = root / "benchmarks"
            with patch.object(bench_speed, "RUN_DIR", source), \
                 patch.object(bench_speed, "BENCH_ROOT", bench_root), \
                 patch.object(bench_speed.subprocess, "Popen") as popen, \
                 patch.object(bench_speed.subprocess, "check_output") as check_output:
                popen.return_value.pid = 1234
                first = bench_speed.launch_training()
                (first / "checkpoint_step_170000.pt").write_bytes(b"benchmark progress")
                (first / "train.log").write_text("first benchmark log")
                previous = {path.name: path.read_bytes() for path in first.iterdir()}
                second = bench_speed.launch_training()
                self.assertNotEqual(first, second)
                for output in (first, second):
                    config = json.loads((output / "config.json").read_text())
                    self.assertEqual(config["output_dir"], str(output))
                    self.assertEqual(config["resume_from"], str(source / "checkpoint_step_160000.pt"))
                    self.assertTrue(config["resume"])
                    self.assertFalse(config["auto_resume"])
                    self.assertEqual((output / "train.pid").read_text(), "1234\n")
                latest = bench_root / "latest-run.txt"
                self.assertEqual(latest.read_text().strip(), str(second))
                popen.side_effect = OSError("launch failed")
                with self.assertRaisesRegex(OSError, "launch failed"):
                    bench_speed.launch_training()
                self.assertEqual(latest.read_text().strip(), str(second))
                self.assertEqual({p.name: p.read_bytes() for p in source.iterdir()}, original)
                self.assertEqual({p.name: p.read_bytes() for p in first.iterdir()}, previous)
                check_output.assert_not_called()

                (second / "train.log").write_text("latest benchmark log")
                check_output.return_value = "GPU status"
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    bench_speed.check_status()
                self.assertIn("latest benchmark log", stdout.getvalue())
                self.assertNotIn("first benchmark log", stdout.getvalue())

    def test_missing_seed_checkpoint_does_not_launch_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(bench_speed, "RUN_DIR", root / "missing"), \
                 patch.object(bench_speed, "BENCH_ROOT", root / "benchmarks"), \
                 patch.object(bench_speed.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(FileNotFoundError, "seed checkpoint"):
                    bench_speed.launch_training()
                popen.assert_not_called()
                self.assertFalse((root / "benchmarks").exists())


if __name__ == "__main__":
    unittest.main()
