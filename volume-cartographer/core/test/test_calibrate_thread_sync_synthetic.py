import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "calibrate_thread_sync_synthetic.py"
SPEC = importlib.util.spec_from_file_location("calibrate_thread_sync_synthetic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

import run_thread_sync_replay as REPLAY


class SyntheticSyncCalibrationTest(unittest.TestCase):
    def test_fit_and_holdout_axes_are_disjoint_and_generic(self):
        fit_workers = {case.workers for case in MODULE.FIT_CASES}
        holdout_workers = {case.workers for case in MODULE.HOLDOUT_CASES}
        self.assertEqual(fit_workers, {2, 4, 6})
        self.assertEqual(holdout_workers, {1, 3, 5, 7})
        self.assertTrue(fit_workers.isdisjoint(holdout_workers))
        self.assertTrue(
            all(
                case.tasks <= case.workers
                for case in (*MODULE.FIT_CASES, *MODULE.HOLDOUT_CASES)
            )
        )
        self.assertEqual(
            {case.workers for case in MODULE.DEPENDENCY_FIT_CASES},
            {2, 4, 6},
        )
        self.assertEqual(
            {case.workers for case in MODULE.DEPENDENCY_VALIDATION_CASES},
            {3, 5},
        )
        self.assertEqual(
            {case.workers for case in MODULE.DEPENDENCY_HOLDOUT_CASES},
            {7},
        )

    def test_dependency_pairs_have_matched_total_work(self):
        cases = (
            *MODULE.DEPENDENCY_FIT_CASES,
            *MODULE.DEPENDENCY_VALIDATION_CASES,
            *MODULE.DEPENDENCY_HOLDOUT_CASES,
        )
        pairs = {}
        for case in cases:
            per_round = sum(
                case.work_iterations + index * case.task_iteration_skew
                for index in range(case.tasks)
            )
            pairs.setdefault(case.pair, []).append(per_round * case.rounds)
        self.assertTrue(all(len(values) == 2 for values in pairs.values()))
        self.assertTrue(all(values[0] == values[1] for values in pairs.values()))

    def test_runner_command_selects_only_synthetic_dispatch_fixture(self):
        command = MODULE.runner_command(
            Path("runner.py"),
            Path("dispatch-benchmark"),
            Path("out"),
            MODULE.FIT_CASES[0],
            3,
            5,
            Path("event-model.json"),
        )
        self.assertIn("synthetic", command)
        self.assertIn("dispatch-benchmark", command)
        self.assertNotIn("renderer", " ".join(command))
        skewed = MODULE.runner_command(
            Path("runner.py"),
            Path("dispatch-benchmark"),
            Path("out"),
            MODULE.DEPENDENCY_FIT_CASES[-1],
            3,
            5,
            Path("event-model.json"),
        )
        self.assertNotEqual(
            "0", skewed[skewed.index("--task-iteration-skew") + 1]
        )

    def point(self, case, events, process, handoff):
        point = MODULE.ReplayPoint(
            case=case,
            events=events,
            core_counts=(1, case.workers + 1),
            native_medians={},
            native_samples={},
            result_path="synthetic",
            fixed_process_ns=process,
        )
        values = np.asarray([handoff])
        for cores in point.core_counts:
            native = MODULE.predict(point, cores, "handoff", values)
            point.native_medians[cores] = native
            point.native_samples[cores] = [native] * 3
        return point

    def test_handoff_model_recovers_synthetic_coefficient(self):
        process = 1_000_000.0
        handoff = 5_000.0
        points = [
            self.point(
                MODULE.FIT_CASES[0],
                [REPLAY.TraceEvent(0, 1, "work", duration=100_000_000.0)],
                process,
                handoff,
            ),
            self.point(
                MODULE.FIT_CASES[1],
                [
                    REPLAY.TraceEvent(0, 1, "work", duration=80_000_000.0),
                    REPLAY.TraceEvent(1, 2, "work", duration=80_000_000.0),
                ],
                process,
                handoff,
            ),
            self.point(
                MODULE.FIT_CASES[2],
                [
                    REPLAY.TraceEvent(0, 1, "work", duration=50_000_000.0),
                    REPLAY.TraceEvent(
                        1,
                        2,
                        "work",
                        dependencies=[(0, "drd_happens_before")],
                        duration=50_000_000.0,
                    ),
                ],
                process,
                handoff,
            ),
            self.point(
                MODULE.FIT_CASES[0],
                [
                    REPLAY.TraceEvent(0, 1, "work", duration=30_000_000.0),
                    REPLAY.TraceEvent(
                        1,
                        2,
                        "work",
                        dependencies=[(0, "drd_happens_before")],
                        duration=30_000_000.0,
                    ),
                    REPLAY.TraceEvent(
                        2,
                        1,
                        "work",
                        dependencies=[(1, "drd_happens_before")],
                        duration=30_000_000.0,
                    ),
                ],
                process,
                handoff,
            ),
        ]
        fitted = MODULE.fit_model("handoff", points)
        self.assertEqual(fitted["jacobian_rank"], 1)
        self.assertAlmostEqual(
            fitted["parameters"]["cross_thread_release_ns"], handoff, delta=300.0
        )

    def test_dependency_model_recovers_coefficient_with_frozen_handoff(self):
        handoff = 7_500.0
        expected = np.asarray([0.35])
        point_specs = [
            [
                REPLAY.TraceEvent(index, index + 1, "work", duration=duration)
                for index, duration in enumerate(
                    (30_000.0, 30_000.0, 20_000.0, 20_000.0, 20_000.0)
                )
            ],
            [
                REPLAY.TraceEvent(0, 1, "work", duration=40_000.0),
                REPLAY.TraceEvent(
                    1,
                    2,
                    "work",
                    dependencies=[(0, "drd_happens_before")],
                    duration=20_000.0,
                ),
                REPLAY.TraceEvent(2, 3, "work", duration=30_000.0),
                REPLAY.TraceEvent(3, 4, "work", duration=20_000.0),
            ],
            [
                REPLAY.TraceEvent(index, index + 1, "work", duration=duration)
                for index, duration in enumerate(
                    (70_000.0, 50_000.0, 40_000.0, 30_000.0, 20_000.0)
                )
            ],
        ]
        points = []
        for index, events in enumerate(point_specs):
            case = MODULE.DEPENDENCY_FIT_CASES[index]
            point = MODULE.ReplayPoint(
                case=case,
                events=events,
                core_counts=(1, 2),
                native_medians={},
                native_samples={},
                result_path="synthetic",
            )
            for cores in point.core_counts:
                native = MODULE.predict(
                    point, cores, "dependency", expected, handoff
                )
                point.native_medians[cores] = native
                point.native_samples[cores] = [native] * 3
            points.append(point)
        fitted = MODULE.fit_model("dependency", points, handoff)
        self.assertAlmostEqual(
            fitted["parameters"]["dependency_excess_scale"],
            expected[0],
            delta=0.03,
        )
        profile = MODULE.dependency_profile(points, handoff)
        self.assertAlmostEqual(
            profile["best_grid_point"]["scale"], expected[0], delta=0.03
        )

    def test_worker_startup_fit_uses_zero_work_worker_count_sweep(self):
        base = 2_000_000.0
        per_worker = 125_000.0
        startup = {
            "records": [
                {
                    "workers": workers,
                    "median_overhead_ns": base + (workers - 1) * per_worker,
                }
                for workers in range(1, 8)
            ]
        }
        fitted = MODULE.fit_worker_startup(startup)
        self.assertAlmostEqual(fitted["base_process_ns"], base)
        self.assertAlmostEqual(
            fitted["per_worker_startup_ns"], per_worker, delta=1.0
        )

    def test_evaluate_reports_speedup_and_absolute_error(self):
        values = np.asarray([5_000.0])
        point = self.point(
            MODULE.HOLDOUT_CASES[0],
            [
                REPLAY.TraceEvent(0, 1, "work", duration=100_000_000.0),
                REPLAY.TraceEvent(1, 2, "work", duration=50_000_000.0),
            ],
            1_000_000.0,
            values[0],
        )
        point.native_medians[point.core_counts[1]] *= 1.1

        report = MODULE.evaluate("handoff", values, [point])

        self.assertEqual(len(report["speedups"]), 1)
        speedup = report["speedups"][0]
        self.assertAlmostEqual(speedup["speedup_error_percent"], 10.0)
        self.assertAlmostEqual(speedup["absolute_speedup_error_percent"], 10.0)
        self.assertAlmostEqual(
            report["maximum_absolute_speedup_error_percent"], 10.0
        )


if __name__ == "__main__":
    unittest.main()
