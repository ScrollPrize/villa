import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "validate_render_thread_calibration.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_render_thread_calibration", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RenderThreadCalibrationValidationTest(unittest.TestCase):
    def parameters(self) -> dict[str, float]:
        return {
            "work_ns_per_iteration": 1.75,
            "fixed_dispatch_ns": 8_000.0,
            "per_future_dispatch_ns": 200.0,
        }

    def test_matrix_has_one_two_four_levels_and_holdout_workers(self) -> None:
        repetitions = {scenario: 10 for scenario in MODULE.SCENARIO_LEVELS}
        cases = MODULE.make_cases(repetitions)
        self.assertEqual(set(MODULE.SCENARIO_LEVELS.values()), {1, 2, 4})
        self.assertEqual({case.workers for case in cases}, set(range(1, 8)))
        self.assertEqual(len(cases), 35)
        self.assertIn("mixed_correlated", MODULE.SCENARIO_LEVELS)
        self.assertIn("mixed_shuffled", MODULE.SCENARIO_LEVELS)

    def test_prediction_uses_no_renderer_coefficient(self) -> None:
        predicted = MODULE.predict_render_nanoseconds(
            self.parameters(), levels=2, workers=5, one_worker_nanoseconds=1_000_000
        )
        one_dispatch = 8_200.0
        five_dispatch = 9_000.0
        expected = (1_000_000.0 - 2 * one_dispatch) / 5 + 2 * five_dispatch
        self.assertEqual(predicted, expected)

    def test_summary_applies_per_case_individual_gate(self) -> None:
        records = []
        for scenario, levels in MODULE.SCENARIO_LEVELS.items():
            for workers in MODULE.VALIDATION_WORKERS:
                one_worker = 1_000_000.0 * levels
                predicted = MODULE.predict_render_nanoseconds(
                    self.parameters(), levels, workers, one_worker
                )
                values = [one_worker] * 5 if workers == 1 else [predicted] * 5
                if scenario == "fallback_1" and workers == 5:
                    values[-1] = predicted / 1.25
                case = MODULE.RenderCase(scenario, levels, workers, 1)
                records.append(
                    {
                        "case": MODULE.asdict(case),
                        "case_id": case.case_id,
                        "samples": [
                            {"nanoseconds_per_render": value} for value in values
                        ],
                    }
                )
        reports, valid = MODULE.summarize(records, self.parameters())
        self.assertFalse(valid)
        failed = [report for report in reports if not report["accepted"]]
        self.assertEqual([report["case_id"] for report in failed], ["renderer-fallback_1-l2-w5"])

    def test_model_loader_rejects_extra_parameters(self) -> None:
        model = {
            "schema_version": 3,
            "parameter_count": 3,
            "parameters": {**self.parameters(), "extra": 1.0},
            "synthetic_calibration_valid": True,
            "domain": {"mode": "futures", "maximum_workers": 7},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(model))
            with self.assertRaises(RuntimeError):
                MODULE.load_model(path)


if __name__ == "__main__":
    unittest.main()
