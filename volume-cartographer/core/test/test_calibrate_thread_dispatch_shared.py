import importlib.util
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "calibrate_thread_dispatch_shared.py"
)
SPEC = importlib.util.spec_from_file_location("calibrate_thread_dispatch_shared", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MinimalDispatchCalibrationTest(unittest.TestCase):
    def parameters(self) -> dict[str, float]:
        return {
            "work_ns_per_iteration": 1.75,
            "fixed_dispatch_ns": 8_000.0,
            "per_future_dispatch_ns": 200.0,
        }

    def sample(self, nanoseconds: float) -> dict[str, float]:
        return {
            "nanoseconds_per_round": nanoseconds,
            "schedstat_wait_fraction": 0.0,
            "actual_idle_nanoseconds_per_round": 0.0,
            "raw_dispatch_nanoseconds_per_round": nanoseconds,
            "clock_overhead_nanoseconds_per_round": 0.0,
        }

    def record(self, case: MODULE.Case, values: list[float]) -> dict:
        return {
            "case": MODULE.asdict(case),
            "case_id": case.case_id,
            "rounds": 100,
            "samples": [self.sample(value) for value in values],
        }

    def test_fit_and_holdout_workers_are_disjoint(self) -> None:
        fit = {case.workers for case in MODULE.make_dispatch_cases(
            MODULE.DISPATCH_FIT_WORKERS, "fit"
        )}
        holdout = {case.workers for case in MODULE.make_dispatch_cases(
            MODULE.HOLDOUT_WORKERS, "holdout"
        )}
        self.assertEqual(fit, {2, 4, 6})
        self.assertEqual(holdout, {1, 3, 5, 7})
        self.assertTrue(fit.isdisjoint(holdout))

    def test_work_holdouts_cover_crossover_and_long_work(self) -> None:
        cases = MODULE.make_work_holdout_cases()
        self.assertEqual(
            {case.work_iterations for case in cases}, {7_500, 350_000, 1_400_000}
        )
        self.assertTrue(all(case.mode == "futures" for case in cases))
        self.assertTrue(all(case.tasks <= case.workers for case in cases))

    def test_diagnostics_are_nonproduction_modes_or_roles(self) -> None:
        gate = MODULE.make_gate_diagnostic_cases()
        self.assertEqual(
            {case.mode for case in gate},
            {"futures", "futures-gate-open", "futures-gate-closed"},
        )
        self.assertEqual(
            {case.idle_nanoseconds for case in MODULE.make_idle_diagnostic_cases()},
            {0, 100_000, 1_000_000},
        )
        self.assertTrue(
            all(case.mode == "lifecycle" for case in MODULE.make_lifecycle_diagnostic_cases())
        )

    def test_affinity_uses_fixed_single_ccd_domain(self) -> None:
        expected = set(range(8))
        for workers in range(1, 8):
            self.assertEqual(
                {int(value) for value in MODULE.affinity(workers).split(",")},
                expected,
            )

    def test_odd_worker_half_wave_uses_floor(self) -> None:
        self.assertEqual(MODULE.wave_task_counts(1), (1,))
        self.assertEqual(MODULE.wave_task_counts(3), (1, 3))
        self.assertEqual(MODULE.wave_task_counts(5), (2, 5))
        self.assertEqual(MODULE.wave_task_counts(7), (3, 7))

    def test_prediction_adds_dispatch_and_work(self) -> None:
        short = {
            "workers": 4,
            "tasks": 4,
            "work_iterations": 1_000,
            "mode": "futures",
        }
        long = {**short, "work_iterations": 100_000}
        self.assertEqual(MODULE.predict_round(self.parameters(), short), 10_550.0)
        self.assertEqual(MODULE.predict_round(self.parameters(), long), 183_800.0)

    def test_prediction_rejects_diagnostics_and_worker_eight(self) -> None:
        observation = {
            "workers": 4,
            "tasks": 4,
            "work_iterations": 100,
            "mode": "futures-gate-open",
        }
        with self.assertRaises(ValueError):
            MODULE.predict_round(self.parameters(), observation)
        with self.assertRaises(ValueError):
            MODULE.predict_round(
                self.parameters(), {**observation, "workers": 8, "mode": "futures"}
            )

    def test_measures_serial_work_scale_and_trial_variation(self) -> None:
        records = []
        for work in MODULE.SERIAL_FIT_WORK:
            case = MODULE.Case(1, "serial", 1, work, "serial-fit")
            records.append(
                self.record(case, [2_000.0 + slope * work for slope in (1.7, 1.75, 1.8)])
            )
        scale, report = MODULE.measure_work_scale(records)
        self.assertAlmostEqual(scale, 1.75)
        np.testing.assert_allclose(
            report["trial_slopes_ns_per_iteration"], [1.7, 1.75, 1.8]
        )

    def test_recovers_dispatch_parameters(self) -> None:
        records = []
        for case in MODULE.make_dispatch_cases(MODULE.DISPATCH_FIT_WORKERS, "fit"):
            value = 8_000.0 + 200.0 * case.tasks
            records.append(self.record(case, [value] * 3))
        parameters, report = MODULE.fit_dispatch(records)
        np.testing.assert_allclose(parameters, [8_000.0, 200.0], rtol=1e-5)
        self.assertEqual(report["jacobian_rank"], 2)
        self.assertFalse(report["parameters_hit_bounds"])

    def test_topology_preflight_accepts_one_ccd_and_remote_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cpu in range(9):
                topology = root / f"cpu{cpu}" / "topology"
                cache = root / f"cpu{cpu}" / "cache" / "index3"
                topology.mkdir(parents=True)
                cache.mkdir(parents=True)
                (topology / "physical_package_id").write_text("0\n")
                (topology / "core_id").write_text(f"{cpu}\n")
                (cache / "level").write_text("3\n")
                (cache / "id").write_text(f"{int(cpu == 8)}\n")
            report = MODULE.validate_topology(root)
            self.assertEqual(report["calibration_llc"], 0)
            self.assertEqual(report["monitor_llc"], 1)

    def test_frequency_preflight_and_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cpu"
            recovery = Path(directory) / "recovery.json"
            recovery.write_text("{}\n")
            (root / "cpufreq").mkdir(parents=True)
            (root / "cpufreq" / "boost").write_text("0\n")
            for number in range(9):
                policy = root / "cpufreq" / f"policy{number}"
                policy.mkdir()
                (policy / "scaling_min_freq").write_text("3401000\n")
                (policy / "scaling_max_freq").write_text("3401000\n")
                (policy / "scaling_governor").write_text("performance\n")
                (policy / "scaling_cur_freq").write_text("3390000\n")
            state, policies = MODULE.validate_fixed_frequency(root, recovery)
            monitor = MODULE.FrequencyMonitor(policies)
            with mock.patch.object(MODULE, "MIN_FREQUENCY_SAMPLES_PER_POLICY", 1):
                with monitor:
                    time.sleep(0.06)
                report = monitor.report(state["target_khz"])
            self.assertTrue(report["within_tolerance"])


if __name__ == "__main__":
    unittest.main()
