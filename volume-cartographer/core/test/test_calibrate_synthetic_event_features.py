import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import passive_event_model as EVENT_MODEL

BASE_SPEC = importlib.util.spec_from_file_location(
    "calibrate_synthetic_event_costs", SCRIPTS / "calibrate_synthetic_event_costs.py"
)
BASE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
sys.modules[BASE_SPEC.name] = BASE
BASE_SPEC.loader.exec_module(BASE)

SPEC = importlib.util.spec_from_file_location(
    "calibrate_synthetic_event_features",
    SCRIPTS / "calibrate_synthetic_event_features.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SyntheticEventFeatureCalibrationTest(unittest.TestCase):
    @staticmethod
    def events():
        return {
            "Ir": 100, "Dr": 20, "Dw": 10,
            "Bc": 0, "Bi": 0, "I1mr": 0,
            "D1mr": 7, "D1mw": 3,
            "ILmr": 0, "DLmr": 2, "DLmw": 1,
            "Bcm": 4, "Bim": 1,
        }

    def test_named_feature_equations_are_exact(self):
        values = EVENT_MODEL.profile_feature_values(self.events())
        self.assertEqual(values["non_data_instructions"], 70)
        self.assertEqual(values["data_reads"], 20)
        self.assertEqual(values["data_writes"], 10)
        self.assertEqual(values["l1_data_read_misses"], 7)
        self.assertEqual(values["l1_data_write_misses"], 3)
        self.assertEqual(values["last_level_data_read_misses"], 2)
        self.assertEqual(values["last_level_data_write_misses"], 1)
        self.assertEqual(values["branch_weighted_l1_misses"], 0.5)

    def test_existing_serialized_schemas_keep_golden_predictions(self):
        legacy = {
            "feature_names": list(EVENT_MODEL.LEGACY_FEATURE_NAMES),
            "coefficients_ns": [1.0] * 6,
            "stall_overlap_fraction": 0.0,
        }
        serialization = {
            "feature_names": list(EVENT_MODEL.FEATURE_NAMES),
            "coefficients_ns": [1.0] * 7,
            "stall_overlap_fraction": 0.0,
        }
        self.assertEqual(EVENT_MODEL.modeled_profile_cost_ns(self.events(), legacy), 98.5)
        self.assertEqual(
            EVENT_MODEL.modeled_profile_cost_ns(self.events(), serialization), 99.5
        )

    def test_new_zero_overlap_schemas_are_named_linear_sums(self):
        for schema in MODULE.CANDIDATE_SCHEMAS.values():
            values = EVENT_MODEL.features_for_model(self.events(), schema)
            coefficients = np.arange(1.0, len(schema) + 1.0)
            model = {
                "feature_names": list(schema),
                "coefficients_ns": list(coefficients),
                "stall_overlap_fraction": 0.0,
            }
            self.assertEqual(
                EVENT_MODEL.modeled_profile_cost_ns(self.events(), model),
                float(values @ coefficients),
            )
            model["stall_overlap_fraction"] = 0.5
            with self.assertRaisesRegex(RuntimeError, "require zero overlap"):
                EVENT_MODEL.modeled_profile_cost_ns(self.events(), model)

    def test_feature_count_does_not_select_schema(self):
        unknown = list(EVENT_MODEL.DATA_READ_FEATURE_NAMES)
        unknown[1] = "unknown"
        with self.assertRaisesRegex(RuntimeError, "unsupported feature basis"):
            EVENT_MODEL.features_for_model(self.events(), tuple(unknown))

    def test_density_fit_and_fresh_cases_are_predeclared(self):
        self.assertEqual(
            {"read-four", "read-eight", "write-eight"},
            {case.kind for case in MODULE.DENSITY_FIT_CASES},
        )
        self.assertEqual(36, len(MODULE.DENSITY_FIT_CASES))
        self.assertEqual(
            {
                "line-read-one", "line-read-eight",
                "line-write-one", "line-write-eight",
            },
            {case.kind for case in MODULE.ACCESS_SEPARATION_FIT_CASES},
        )
        self.assertEqual(48, len(MODULE.ACCESS_SEPARATION_FIT_CASES))
        self.assertEqual(
            {"line-r1-w1", "line-r8-w1", "line-r1-w8", "line-r8-w8"},
            {case.kind for case in MODULE.CROSSED_ACCESS_FIT_CASES},
        )
        self.assertEqual(48, len(MODULE.CROSSED_ACCESS_FIT_CASES))
        self.assertEqual(132, len(MODULE.ALL_DENSITY_FIT_CASES))
        self.assertEqual(72, len(MODULE.FRESH_HOLDOUT_CASES))
        self.assertEqual(
            {
                "branch", "stream-read", "stream-write", "cache-read",
                "grid-sample", "mixed-grid-phase", "mixed-grid-random",
                "read-four", "read-eight", "write-eight",
                "line-read-one", "line-read-eight",
                "line-write-one", "line-write-eight",
                "line-r1-w1", "line-r8-w1", "line-r1-w8", "line-r8-w8",
            },
            {case.kind for case in MODULE.FRESH_HOLDOUT_CASES},
        )

    def test_fresh_sizes_do_not_overlap_opened_static_matrices(self):
        opened_sizes = {
            case.working_set_bytes
            for case in (
                *BASE.FIT_CASES, *BASE.CACHE_FIT_CASES, *BASE.HOLDOUT_CASES,
                *BASE.CACHE_HOLDOUT_CASES, *BASE.MIXED_FIT_CASES,
                *BASE.MIXED_HOLDOUT_CASES, *BASE.SERIALIZATION_HOLDOUT_CASES,
                *BASE.DIAGNOSTIC_CASES, *MODULE.ALL_DENSITY_FIT_CASES,
            )
        }
        opened_sizes.update((98_304, 1_572_864, 6_291_456))
        fresh_sizes = {case.working_set_bytes for case in MODULE.FRESH_HOLDOUT_CASES}
        self.assertTrue(fresh_sizes.isdisjoint(opened_sizes))

    def test_data_matrix_diagnostics_report_rank_and_correlation(self):
        records = []
        for index, name in enumerate(EVENT_MODEL.LEGACY_FEATURE_NAMES):
            profile = {event: 0 for event in BASE.EVENT_NAMES}
            profile["Ir"] = 1
            if name == "data_writes":
                profile.update({"Dr": 1, "Dw": 1})
            elif name == "l1_data_misses":
                profile.update({"Dr": 1, "D1mr": 1})
            elif name == "last_level_data_misses":
                profile.update({"Dr": 1, "DLmr": 1})
            elif name == "branch_misses":
                profile["Bcm"] = 1
            elif name == "branch_weighted_l1_misses":
                profile.update({"Dr": 1, "D1mr": 1, "Bcm": 1})
            records.append({
                "case_id": str(index),
                "native_work_median_ns": 1.0,
                "profiles": {"1": profile},
            })
        diagnostics = MODULE.matrix_diagnostics(
            records, EVENT_MODEL.LEGACY_FEATURE_NAMES
        )
        self.assertEqual(diagnostics["rank"], 6)
        self.assertLess(diagnostics["maximum_absolute_parameter_correlation"], 1.0)

    def test_native_range_detects_multimodal_measurements(self):
        record = {"native_work_samples_ns": [100.0, 101.0, 99.0, 300.0, 301.0]}
        self.assertGreater(MODULE.relative_native_range_percent(record), 100.0)


if __name__ == "__main__":
    unittest.main()
