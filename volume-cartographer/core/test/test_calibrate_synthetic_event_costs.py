import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import passive_event_model as EVENT_MODEL


SCRIPT = SCRIPTS / "calibrate_synthetic_event_costs.py"
SPEC = importlib.util.spec_from_file_location("calibrate_synthetic_event_costs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SyntheticEventCostCalibrationTest(unittest.TestCase):
    def test_matrix_is_overdetermined_and_holdouts_are_disjoint(self):
        self.assertGreaterEqual(
            len(MODULE.FIT_CASES) + len(MODULE.CACHE_FIT_CASES)
            + len(MODULE.MIXED_FIT_CASES) + 5,
            10 * len(MODULE.FEATURE_NAMES),
        )
        self.assertGreaterEqual(len(MODULE.HOLDOUT_CASES), 20)
        fit_ids = {case.case_id.replace("fit-", "", 1) for case in MODULE.FIT_CASES}
        holdout_ids = {
            case.case_id.replace("holdout-", "", 1)
            for case in MODULE.HOLDOUT_CASES
        }
        self.assertTrue(fit_ids.isdisjoint(holdout_ids))
        self.assertTrue(
            {case.kind for case in MODULE.DIAGNOSTIC_CASES}.isdisjoint(
                {
                    case.kind
                    for case in (*MODULE.FIT_CASES, *MODULE.MIXED_FIT_CASES)
                }
            )
        )

    def test_mixed_fit_and_holdout_are_predeclared_and_disjoint(self):
        self.assertGreaterEqual(len(MODULE.MIXED_FIT_CASES), 10)
        self.assertGreaterEqual(len(MODULE.MIXED_HOLDOUT_CASES), 10)
        self.assertEqual(
            {"mixed-grid-phase", "mixed-grid-random"},
            {case.kind for case in MODULE.MIXED_FIT_CASES},
        )
        fit = {
            (case.kind, case.working_set_bytes, case.iterations)
            for case in MODULE.MIXED_FIT_CASES
        }
        holdout = {
            (case.kind, case.working_set_bytes, case.iterations)
            for case in MODULE.MIXED_HOLDOUT_CASES
        }
        self.assertTrue(fit.isdisjoint(holdout))

    def test_fit_kernels_match_renderer_instruction_shapes(self):
        self.assertEqual(
            {"branch", "stream-read", "stream-write", "grid-sample"},
            {
                case.kind
                for case in MODULE.FIT_CASES
                if not case.kind.startswith("cache-")
            },
        )

    def test_cache_fit_and_holdout_cross_configured_cache_boundaries(self):
        fit = MODULE.CACHE_FIT_CASES
        holdout = MODULE.CACHE_HOLDOUT_CASES
        self.assertEqual({"cache-read"}, {case.kind for case in fit})
        self.assertGreater(max(case.working_set_bytes for case in fit), 8 * 1024 * 1024)
        self.assertGreater(
            max(case.working_set_bytes for case in holdout), 8 * 1024 * 1024
        )

    def test_joint_work_model_includes_last_level_misses(self):
        self.assertEqual(
            (
                "non_data_instructions",
                "data_writes",
                "l1_data_misses",
                "last_level_data_misses",
                "branch_misses",
                "branch_weighted_l1_misses",
                "l1_miss_serial_pressure",
            ),
            MODULE.FEATURE_NAMES,
        )

    def test_cache_stress_extends_the_joint_fit_matrix(self):
        self.assertGreater(
            len(MODULE.FIT_CASES) + len(MODULE.CACHE_FIT_CASES),
            10 * len(MODULE.LEGACY_FEATURE_NAMES),
        )

    def test_startup_uses_a_minimal_zero_work_case(self):
        self.assertEqual(0, MODULE.STARTUP_CASE.iterations)
        self.assertEqual(1, MODULE.STARTUP_CASE.rounds)
        self.assertEqual(0, MODULE.STARTUP_CASE.warmup_rounds)
        command = MODULE.benchmark_command(Path("benchmark"), MODULE.STARTUP_CASE)
        self.assertEqual("1", command[command.index("--workers") + 1])

    def test_stall_overlap_is_bounded_to_shared_stall_work(self):
        values = np.ones(len(MODULE.LEGACY_FEATURE_NAMES))
        coefficients = np.ones(len(MODULE.LEGACY_FEATURE_NAMES))
        no_overlap = MODULE.modeled_feature_cost_ns(values, coefficients, 0.0)
        full_overlap = MODULE.modeled_feature_cost_ns(values, coefficients, 1.0)
        self.assertEqual(no_overlap, 6.0)
        self.assertEqual(full_overlap, 4.5)
        self.assertGreaterEqual(full_overlap, 2.0)

    @staticmethod
    def profile(instructions, l1_misses):
        return {
            "Ir": instructions,
            "Dr": l1_misses,
            "Dw": 0,
            "Bc": 0,
            "Bi": 0,
            "I1mr": 0,
            "D1mr": l1_misses,
            "D1mw": 0,
            "ILmr": 0,
            "DLmr": 0,
            "DLmw": 0,
            "Bcm": 0,
            "Bim": 0,
        }

    def test_serial_pressure_is_exact_and_scales_linearly(self):
        profile = self.profile(100, 10)
        values = EVENT_MODEL.profile_features(profile)
        self.assertEqual(values[-1], 1.0)
        scaled = {name: 3 * value for name, value in profile.items()}
        np.testing.assert_allclose(
            EVENT_MODEL.profile_features(scaled), 3.0 * values
        )

    def test_features_sum_homogeneous_threads_before_modeling(self):
        profile = self.profile(100, 10)
        record = {"case_id": "homogeneous", "profiles": {"1": profile, "2": profile}}
        summed = MODULE.features(record)
        merged = self.profile(200, 20)
        np.testing.assert_allclose(summed, EVENT_MODEL.profile_features(merged))

    def test_heterogeneous_thread_density_is_not_merged(self):
        low = self.profile(100, 1)
        high = self.profile(100, 20)
        record = {"case_id": "heterogeneous", "profiles": {"1": low, "2": high}}
        summed = MODULE.features(record)
        merged = EVENT_MODEL.profile_features(self.profile(200, 21))
        self.assertAlmostEqual(summed[-1], 4.01)
        self.assertAlmostEqual(merged[-1], 2.205)

    def test_legacy_schema_remains_exact_and_new_schema_rejects_overlap(self):
        profile = self.profile(100, 10)
        legacy = {
            "feature_names": list(EVENT_MODEL.LEGACY_FEATURE_NAMES),
            "coefficients_ns": [1.0] * len(EVENT_MODEL.LEGACY_FEATURE_NAMES),
            "stall_overlap_fraction": 0.5,
        }
        self.assertEqual(
            EVENT_MODEL.modeled_profile_cost_ns(profile, legacy),
            EVENT_MODEL.modeled_feature_cost_ns(
                EVENT_MODEL.profile_features(profile)[:-1],
                np.ones(len(EVENT_MODEL.LEGACY_FEATURE_NAMES)),
                0.5,
            ),
        )
        candidate = {
            "feature_names": list(EVENT_MODEL.FEATURE_NAMES),
            "coefficients_ns": [1.0] * len(EVENT_MODEL.FEATURE_NAMES),
            "stall_overlap_fraction": 0.5,
        }
        with self.assertRaisesRegex(RuntimeError, "require zero overlap"):
            EVENT_MODEL.modeled_profile_cost_ns(profile, candidate)
        candidate["feature_names"][-1] = "unknown"
        with self.assertRaisesRegex(RuntimeError, "unsupported feature basis"):
            EVENT_MODEL.modeled_profile_cost_ns(profile, candidate)

    def test_serialization_holdouts_are_fresh_controls(self):
        cases = MODULE.SERIALIZATION_HOLDOUT_CASES
        self.assertEqual(
            {
                "pointer", "cache-read", "grid-sample",
                "mixed-grid-phase", "mixed-grid-random",
            },
            {case.kind for case in cases},
        )
        opened_sizes = {
            case.working_set_bytes
            for case in (*MODULE.DIAGNOSTIC_CASES, *MODULE.CACHE_HOLDOUT_CASES,
                         *MODULE.MIXED_HOLDOUT_CASES)
        }
        self.assertTrue(
            {case.working_set_bytes for case in cases}.isdisjoint(opened_sizes)
        )

    def test_frequency_validation_requires_evidence(self):
        self.assertFalse(MODULE.all_frequency_reports_valid({}))
        self.assertTrue(
            MODULE.all_frequency_reports_valid(
                {"fit": {"within_tolerance": True}, "holdout": {"within_tolerance": True}}
            )
        )
        self.assertFalse(
            MODULE.all_frequency_reports_valid(
                {"fit": {"within_tolerance": True}, "holdout": {"within_tolerance": False}}
            )
        )


if __name__ == "__main__":
    unittest.main()
