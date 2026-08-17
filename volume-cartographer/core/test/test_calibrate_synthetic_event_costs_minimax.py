import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

BASE_SPEC = importlib.util.spec_from_file_location(
    "calibrate_synthetic_event_costs", SCRIPTS / "calibrate_synthetic_event_costs.py"
)
BASE = importlib.util.module_from_spec(BASE_SPEC)
assert BASE_SPEC.loader is not None
sys.modules[BASE_SPEC.name] = BASE
BASE_SPEC.loader.exec_module(BASE)

SPEC = importlib.util.spec_from_file_location(
    "calibrate_synthetic_event_costs_minimax",
    SCRIPTS / "calibrate_synthetic_event_costs_minimax.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(case_id, row, measured=1.0, kind="branch", role="fit"):
    profile = {name: 0 for name in BASE.EVENT_NAMES}
    profile.update(row)
    return {
        "case_id": case_id,
        "case": {"kind": kind, "role": role},
        "native_work_median_ns": measured,
        "native_work_samples_ns": [measured] * 5,
        "profiles": {"1": profile},
    }


class MinimaxEventCostCalibrationTest(unittest.TestCase):
    def test_known_exact_solution_and_nonnegative_bounds(self):
        rows = []
        for index, name in enumerate(BASE.LEGACY_FEATURE_NAMES):
            events = {"Ir": 1}
            if name == "non_data_instructions":
                events = {"Ir": 1}
            elif name == "data_writes":
                events = {"Ir": 1, "Dr": 1, "Dw": 1}
            elif name == "l1_data_misses":
                events = {"Ir": 1, "Dr": 1, "D1mr": 1}
            elif name == "last_level_data_misses":
                events = {"Ir": 1, "Dr": 1, "DLmr": 1}
            elif name == "branch_misses":
                events = {"Ir": 1, "Bcm": 1}
            else:
                events = {"Ir": 1, "Dr": 1, "D1mr": 1, "Bcm": 1}
            rows.append(record(str(index), events, measured=float(index + 1)))
        target = np.arange(1.0, 7.0)
        matrix = np.asarray([BASE.features(value, BASE.LEGACY_FEATURE_NAMES) for value in rows])
        for value, measured in zip(rows, matrix @ target):
            value["native_work_median_ns"] = float(measured)
        model = BASE.fit_minimax(rows, 0.0, list(target))
        self.assertLess(model["optimal_maximum_relative_error"], 1e-8)
        np.testing.assert_allclose(model["coefficients_ns"], target, rtol=1e-6)
        self.assertEqual(model["bound_hit_count"], 0)

    def test_uniform_event_and_time_scaling_preserves_coefficients(self):
        rows = []
        for index in range(len(BASE.LEGACY_FEATURE_NAMES)):
            events = {
                "Ir": 100 + index,
                "Dr": 10 + index,
                "Dw": 2 + index,
                "D1mr": 3 + index,
                "DLmr": index,
                "Bcm": 1 + index,
            }
            rows.append(record(str(index), events, measured=1.0))
        target = np.arange(1.0, 7.0)
        matrix = np.asarray([BASE.features(value, BASE.LEGACY_FEATURE_NAMES) for value in rows])
        for value, measured in zip(rows, matrix @ target):
            value["native_work_median_ns"] = float(measured)
        first = BASE.fit_minimax(rows, 0.0, list(target))
        scaled = []
        for value in rows:
            copy = dict(value)
            copy["profiles"] = {
                thread: {name: 7 * count for name, count in profile.items()}
                for thread, profile in value["profiles"].items()
            }
            copy["native_work_median_ns"] = 7 * value["native_work_median_ns"]
            scaled.append(copy)
        second = BASE.fit_minimax(scaled, 0.0, list(target))
        np.testing.assert_allclose(
            first["coefficients_ns"], second["coefficients_ns"], rtol=1e-6
        )

    def test_fresh_cases_are_unique_and_use_all_expected_kinds(self):
        signatures = {MODULE.case_signature(case) for case in MODULE.FRESH_HOLDOUT_CASES}
        self.assertEqual(len(signatures), len(MODULE.FRESH_HOLDOUT_CASES))
        self.assertEqual(
            {"branch", "stream-read", "stream-write", "cache-read", "grid-sample", "mixed-grid-phase", "mixed-grid-random"},
            {case.kind for case in MODULE.FRESH_HOLDOUT_CASES},
        )
        opened = {
            MODULE.case_signature(case)
            for case in (
                *BASE.FIT_CASES, *BASE.CACHE_FIT_CASES, *BASE.HOLDOUT_CASES,
                *BASE.CACHE_HOLDOUT_CASES, *BASE.MIXED_FIT_CASES,
                *BASE.MIXED_HOLDOUT_CASES, *BASE.SERIALIZATION_HOLDOUT_CASES,
                *BASE.DIAGNOSTIC_CASES,
            )
        }
        self.assertTrue(signatures.isdisjoint(opened))

    def test_fit_records_exclude_serialization_and_diagnostics(self):
        expected = ("branch", "stream-read", "stream-write", "cache-read", "grid-sample", "mixed-grid-phase", "mixed-grid-random")
        observations = {"fit": [], "cache_fit": [], "mixed_fit": [], "serialization_fit": [record("pointer", {"Ir": 1}, kind="pointer")]}
        for index, kind in enumerate(expected):
            target = "cache_fit" if kind == "cache-read" else "mixed_fit" if kind.startswith("mixed-") else "fit"
            observations[target].append(record(str(index), {"Ir": 1}, kind=kind))
        result = MODULE.fit_records(observations)
        self.assertEqual(set(expected), {value["case"]["kind"] for value in result})
        self.assertNotIn("pointer", {value["case"]["kind"] for value in result})

    def test_native_noise_uses_relative_mad(self):
        value = record("noise", {"Ir": 1}, measured=100.0)
        value["native_work_samples_ns"] = [90.0, 100.0, 100.0, 110.0, 120.0]
        self.assertEqual(MODULE.relative_native_mad_percent(value), 10.0)


if __name__ == "__main__":
    unittest.main()
