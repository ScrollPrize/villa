#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "run_render_valgrind_ci.py"
SPEC = importlib.util.spec_from_file_location("run_render_valgrind_ci", SCRIPT)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class RenderValgrindCiTest(unittest.TestCase):
    def test_invalid_tolerances_are_rejected(self):
        for tolerance in (-0.01, 1.0, float("inf"), float("nan")):
            with self.subTest(tolerance=tolerance):
                with self.assertRaisesRegex(RuntimeError, "tolerance"):
                    DRIVER.validate_tolerance(tolerance)

    def test_invalid_scores_are_rejected(self):
        for score in (
            float("nan"),
            float("inf"),
            float("-inf"),
            0.0,
            -1.0,
            "not-a-score",
            None,
        ):
            with self.subTest(score=score):
                with self.assertRaisesRegex(RuntimeError, "finite and positive"):
                    DRIVER.validate_score(score, "test")

    def test_atomic_json_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            DRIVER.write_json_atomic(path, {"complete": True})
            self.assertEqual(json.loads(path.read_text()), {"complete": True})
            self.assertEqual(list(path.parent.glob(".*.tmp-*")), [])

    def test_freeze_model_requires_explicit_unpromoted_approval(self):
        calibration = {
            "renderer_inputs_used": False,
            "candidate_accepted": False,
            "parameters": {"cross_thread_release_ns": 12.5},
            "event_cost_model": {
                "feature_names": list(DRIVER.DATA_READ_FEATURE_NAMES),
                "coefficients_ns": [float(index) for index in range(7)],
                "stall_overlap_fraction": 0.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "calibration.json"
            output = root / "model.json"
            source.write_text(json.dumps(calibration))
            args = SimpleNamespace(
                calibration=source,
                output=output,
                model_id="test-model",
                allow_unpromoted=False,
            )
            with self.assertRaisesRegex(RuntimeError, "not accepted"):
                DRIVER.freeze_model(args)
            args.allow_unpromoted = True
            DRIVER.freeze_model(args)
            model = json.loads(output.read_text())
            self.assertEqual(model["model_id"], "test-model")
            self.assertEqual(model["cross_thread_release_ns"], 12.5)
            self.assertFalse(model["timing_claims_enabled"])

    def test_set_tolerance_preserves_all_reference_cases(self):
        reference = {
            "schema_version": 1,
            "model_sha256": "model-hash",
            "tolerance": 0.10,
            "cases": {},
        }
        for fixture in ("serial", "parallel"):
            for scenario in DRIVER.SCENARIOS:
                reference["cases"][f"{fixture}/{scenario}"] = {
                    "checksum": 123,
                    "modeled_runtime_score_ns": 100.0,
                }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(json.dumps(reference))
            DRIVER.set_tolerance(
                SimpleNamespace(reference=path, output=None, tolerance=0.05)
            )
            updated = json.loads(path.read_text())
            self.assertEqual(updated["tolerance"], 0.05)
            reference["tolerance"] = 0.05
            self.assertEqual(updated, reference)

    def test_freeze_reference_accepts_callgrind_only_native_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.json"
            model.write_text(json.dumps({"model_id": "native-model"}))
            results = []
            for fixture in ("serial", "parallel"):
                for scenario in DRIVER.SCENARIOS:
                    result = root / f"{fixture}-{scenario}.json"
                    result.write_text(
                        json.dumps(
                            {
                                "schema_version": 3,
                                "kind": "evaluation",
                                "case": f"{fixture}/{scenario}",
                                "model_id": "native-model",
                                "checksum": 123,
                                "modeled_runtime_score_ns": 100.0,
                            }
                        )
                    )
                    results.append(result)

            output = root / "reference.json"
            DRIVER.freeze_reference(
                SimpleNamespace(
                    tolerance=0.05,
                    model=model,
                    output=output,
                    results=results,
                )
            )
            reference = json.loads(output.read_text())
            self.assertEqual(reference["tolerance"], 0.05)
            self.assertEqual(len(reference["cases"]), 8)


if __name__ == "__main__":
    unittest.main()
