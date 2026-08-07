#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_render_callgrind.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_render_callgrind", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class RenderCallgrindModelTest(unittest.TestCase):
    def setUp(self):
        self.calibration = {
            "schema_version": 1,
            "model_version": 2,
            "formula": {
                "modeled_ipc": 4.0,
                "l1_miss_cycles": 2.0,
                "last_level_miss_cycles": 10.0,
                "branch_mispredict_cycles": 5.0,
            },
            "effective_parallelism": {"1": 1.0, "2": 1.5, "4": 2.0},
            "nanoseconds_per_modeled_cycle": 0.2,
            "reference": {"host_cpu": "test"},
        }
        self.events = {
            "Ir": 100,
            "I1mr": 1,
            "D1mr": 1,
            "D1mw": 1,
            "ILmr": 1,
            "DLmr": 1,
            "DLmw": 1,
            "Bcm": 1,
            "Bim": 1,
        }

    def test_formula_and_parallel_estimate(self):
        self.assertEqual(
            RUNNER.modeled_work_cycles(self.events, self.calibration), 71.0
        )
        result = RUNNER.calibrated_estimate(
            self.events, 10, "parallel", 2, self.calibration
        )
        self.assertAlmostEqual(result["modeled_cycles_per_pixel"], 71.0 / 15.0)
        self.assertAlmostEqual(
            result["estimated_mpx_per_second"], 1000.0 / (71.0 / 15.0 * 0.2)
        )

    def test_serial_fixture_does_not_apply_worker_normalization(self):
        result = RUNNER.calibrated_estimate(
            self.events, 10, "serial", 4, self.calibration
        )
        self.assertAlmostEqual(result["modeled_cycles_per_pixel"], 7.1)

    def test_missing_required_event_is_rejected(self):
        del self.events["Bim"]
        with self.assertRaisesRegex(RuntimeError, "missing events"):
            RUNNER.modeled_work_cycles(self.events, self.calibration)

    def test_calibration_version_mismatch_is_rejected(self):
        self.calibration["model_version"] = 99
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(json.dumps(self.calibration))
            with self.assertRaisesRegex(RuntimeError, "model version"):
                RUNNER.load_calibration(path)

    def test_parse_separate_thread_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "callgrind.out-02"
            path.write_text("events: Ir D1mr\nsummary: 123 7\n")
            self.assertEqual(RUNNER.parse_callgrind(path), {"Ir": 123, "D1mr": 7})

    def test_parse_pads_omitted_trailing_zero_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "callgrind.out-02"
            path.write_text("events: Ir Bcm Bim\nsummary: 123 7\n")
            self.assertEqual(
                RUNNER.parse_callgrind(path), {"Ir": 123, "Bcm": 7, "Bim": 0}
            )


if __name__ == "__main__":
    unittest.main()
