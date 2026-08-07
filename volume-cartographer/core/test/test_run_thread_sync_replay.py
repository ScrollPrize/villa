#!/usr/bin/env python3

import importlib.util
import math
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_thread_sync_replay.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_thread_sync_replay", SCRIPT)
REPLAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPLAY
SPEC.loader.exec_module(REPLAY)

from native_thread_sync_replay import (  # noqa: E402
    NativeReplayEngine,
    attribution_request,
    replay_request,
)


class ThreadSyncReplayTest(unittest.TestCase):
    def test_parse_pairs_blocking_futex_with_wake(self):
        trace = """\
--00:00:00:00.001 7-- SCHED[1]: entering VG_(scheduler)
SYSCALL[7,1](56) sys_clone ( 0, 0, 0, 0, 0 )
--00:00:00:00.002 7-- SCHED[2]: entering VG_(scheduler)
SYSCALL[7,2](202) sys_futex ( 0xabc, 393, 0, 0x0, 0x0 ) --> [async] ...
--00:00:00:00.003 7-- SCHED[2]: releasing lock (VG_(client_syscall)[async]) -> VgTs_WaitSys
SYSCALL[7,1](202) sys_futex ( 0xabc, 129, 1, 0x0, 0x0 ) --> [async] ...
--00:00:00:00.004 7-- SCHED[2]:  acquired lock (VG_(client_syscall)[async])
SYSCALL[7,2](202) ... [async] --> Success(0x0)
--00:00:00:00.005 7-- SCHED[2]: releasing lock (VG_(scheduler):timeslice) -> VgTs_Yielding
--00:00:00:00.006 7-- SCHED[2]: exiting VG_(scheduler)
--00:00:00:00.007 7-- SCHED[1]: exiting VG_(scheduler)
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.log"
            path.write_text(trace)
            parsed = REPLAY.parse_core_trace(path)
        self.assertEqual(parsed.blocking_waits, 1)
        self.assertEqual(parsed.matched_waits, 1)
        self.assertEqual(parsed.unmatched_waits, 0)
        resume = next(event for event in parsed.events if event.kind == "futex_resume")
        self.assertTrue(any(kind == "futex_wake" for _, kind in resume.dependencies))

    def test_cost_assignment_preserves_thread_total(self):
        events = [
            REPLAY.TraceEvent(0, 1, "thread_start"),
            REPLAY.TraceEvent(
                1, 1, "work_quantum", dependencies=[(0, "program_order")]
            ),
            REPLAY.TraceEvent(
                2, 1, "thread_finish", dependencies=[(1, "program_order")]
            ),
        ]
        REPLAY.assign_costs(events, {1: 120.0}, 0.5, "equal")
        self.assertAlmostEqual(sum(event.duration for event in events), 120.0)

    def test_parse_drd_vector_clock_adds_happens_before_edge(self):
        trace = """\
--4-- New segment for thread 1 with vc [ 1: 1 ]
--4-- New segment for thread 2 with vc [ 1: 1, 2: 1 ]
--4-- New segment for thread 2 with vc [ 1: 1, 2: 2 ]
--4-- New segment for thread 1 with vc [ 1: 2, 2: 2 ]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.log"
            path.write_text(trace)
            parsed = REPLAY.parse_drd_trace(path)
        self.assertEqual(parsed.happens_before_edges, 2)
        self.assertEqual(parsed.unresolved_happens_before, 0)
        last = parsed.events[-1]
        self.assertTrue(
            any(kind == "drd_happens_before" for _, kind in last.dependencies)
        )

    def test_parse_drd_reports_missing_vector_clock_predecessor(self):
        trace = "--4-- New segment for thread 2 with vc [ 1: 7, 2: 1 ]\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.log"
            path.write_text(trace)
            parsed = REPLAY.parse_drd_trace(path)
        self.assertEqual(parsed.unresolved_happens_before, 1)

    def test_futex_dependency_limits_parallel_replay(self):
        events = [
            REPLAY.TraceEvent(0, 1, "thread_start"),
            REPLAY.TraceEvent(1, 2, "thread_start"),
            REPLAY.TraceEvent(2, 2, "futex_wait", dependencies=[(1, "program_order")]),
            REPLAY.TraceEvent(3, 1, "futex_wake", dependencies=[(0, "program_order")]),
            REPLAY.TraceEvent(
                4,
                2,
                "futex_resume",
                dependencies=[(2, "program_order"), (3, "futex_wake")],
            ),
        ]
        events[3].duration = 10.0
        events[4].duration = 10.0
        result = REPLAY.simulate(events, 2, "fifo")
        self.assertEqual(result["modeled_makespan"], 20.0)
        self.assertGreater(result["logical_sync_delay"], 0.0)

    def test_cross_thread_latency_applies_to_drd_release(self):
        events = [
            REPLAY.TraceEvent(0, 1, "work", duration=10.0),
            REPLAY.TraceEvent(
                1,
                2,
                "work",
                dependencies=[(0, "drd_happens_before")],
                duration=5.0,
            ),
        ]
        baseline = REPLAY.simulate(events, 2, "fifo")
        delayed = REPLAY.simulate(events, 2, "fifo", cross_thread_latency=7.0)
        self.assertEqual(
            delayed["modeled_makespan"], baseline["modeled_makespan"] + 7.0
        )

    def test_idle_scale_interpolates_between_lower_bound_and_raw_replay(self):
        events = [
            REPLAY.TraceEvent(index, index + 1, "work", duration=duration)
            for index, duration in enumerate((3.0, 3.0, 2.0, 2.0, 2.0))
        ]
        raw = REPLAY.simulate_adjusted(events, 2, "fifo", replay_idle_scale=1.0)
        lower = REPLAY.simulate_adjusted(events, 2, "fifo", replay_idle_scale=0.0)
        half = REPLAY.simulate_adjusted(events, 2, "fifo", replay_idle_scale=0.5)
        self.assertEqual(raw["raw_replay_makespan"], 7.0)
        self.assertEqual(lower["modeled_makespan"], 6.0)
        self.assertEqual(half["modeled_makespan"], 6.5)
        self.assertEqual(raw["modeled_makespan"], raw["raw_replay_makespan"])

    def test_dependency_scale_changes_only_inferred_drd_excess(self):
        events = [
            REPLAY.TraceEvent(0, 1, "work", duration=10.0),
            REPLAY.TraceEvent(
                1,
                2,
                "work",
                dependencies=[(0, "drd_happens_before")],
                duration=10.0,
            ),
        ]
        zero = REPLAY.simulate_adjusted(events, 2, "fifo", dependency_excess_scale=0.0)
        half = REPLAY.simulate_adjusted(events, 2, "fifo", dependency_excess_scale=0.5)
        full = REPLAY.simulate_adjusted(events, 2, "fifo", dependency_excess_scale=1.0)
        self.assertEqual(zero["hard_schedule_lower_bound"], 10.0)
        self.assertEqual(zero["dependency_excess"], 10.0)
        self.assertEqual(zero["modeled_makespan"], 10.0)
        self.assertEqual(half["modeled_makespan"], 15.0)
        self.assertEqual(full["modeled_makespan"], 20.0)

    def test_dependency_scale_preserves_hard_task_lower_bound(self):
        events = [
            REPLAY.TraceEvent(index, index + 1, "work", duration=duration)
            for index, duration in enumerate((70.0, 10.0, 10.0, 10.0))
        ]
        result = REPLAY.simulate_adjusted(
            events, 4, "fifo", dependency_excess_scale=0.0
        )
        self.assertEqual(result["work_per_core_lower_bound"], 25.0)
        self.assertEqual(result["hard_schedule_lower_bound"], 70.0)
        self.assertEqual(result["modeled_makespan"], 70.0)

    def test_dependency_scale_preserves_fifo_excess(self):
        events = [
            REPLAY.TraceEvent(index, index + 1, "work", duration=duration)
            for index, duration in enumerate((3.0, 3.0, 2.0, 2.0, 2.0))
        ]
        result = REPLAY.simulate_adjusted(
            events, 2, "fifo", dependency_excess_scale=0.0
        )
        self.assertEqual(result["raw_replay_excess"], 1.0)
        self.assertEqual(result["modeled_makespan"], 7.0)

    def test_dependency_scale_rejects_invalid_bounds(self):
        with self.assertRaisesRegex(RuntimeError, "dependency excess scale"):
            REPLAY.simulate_adjusted([], 1, "fifo", dependency_excess_scale=1.1)

    def test_replay_model_schema_keeps_scale_kinds_disjoint(self):
        self.assertEqual(
            REPLAY.replay_scales_for_model(
                {"schema_version": 2, "parameters": {"replay_idle_scale": 0.4}}
            ),
            (0.4, 1.0),
        )
        self.assertEqual(
            REPLAY.replay_scales_for_model(
                {
                    "schema_version": 3,
                    "parameters": {"dependency_excess_scale": 0.6},
                }
            ),
            (1.0, 0.6),
        )
        with self.assertRaisesRegex(RuntimeError, "schema 3"):
            REPLAY.replay_scales_for_model(
                {"schema_version": 3, "parameters": {"replay_idle_scale": 0.5}}
            )

    def test_synthetic_workload_uses_drd_and_spare_caller_core(self):
        self.assertEqual(REPLAY._parallel_core_count("synthetic", 7), 8)
        self.assertIn("--tool=drd", REPLAY._valgrind_trace_tool_options("synthetic"))
        self.assertEqual(
            REPLAY._valgrind_scheduler_options("synthetic"),
            ["--fair-sched=yes"],
        )

    def test_renderer_command_and_fair_scheduler_are_passive(self):
        args = Namespace(
            workload="renderer",
            benchmark=Path("/tmp/bench_render_synthetic"),
            scenario="fallback_1",
            repetitions=17,
        )
        self.assertEqual(
            REPLAY._benchmark_args(args),
            [
                "/tmp/bench_render_synthetic",
                "--fixture",
                "parallel",
                "--scenario",
                "fallback_1",
                "--repetitions",
                "17",
                "--native-trials",
                "1",
            ],
        )
        self.assertEqual(
            REPLAY._valgrind_scheduler_options("renderer"),
            ["--fair-sched=yes"],
        )
        self.assertEqual(REPLAY._callgrind_benchmark_args(args)[-1], "--callgrind")
        native = REPLAY._benchmark_args(args, repetitions=512)
        self.assertEqual(native[native.index("--repetitions") + 1], "512")
        self.assertEqual(REPLAY._valgrind_scheduler_options("dispatch"), [])
        self.assertEqual(
            REPLAY._valgrind_trace_tool_options("renderer"),
            ["--tool=drd", "--trace-segment=yes", "--trace-csw=yes"],
        )
        self.assertEqual(
            REPLAY._valgrind_trace_tool_options("dispatch"), ["--tool=none"]
        )
        self.assertEqual(REPLAY._parallel_core_count("renderer", 7), 8)
        self.assertEqual(REPLAY._parallel_core_count("dispatch", 7), 7)

    @unittest.skipUnless(
        os.environ.get("VC_THREAD_SYNC_REPLAY_BIN"),
        "native replay executable is not configured",
    )
    def test_native_batch_matches_python_reference_for_all_policies(self):
        events = [
            REPLAY.TraceEvent(0, 1, "thread_start"),
            REPLAY.TraceEvent(1, 2, "thread_start"),
            REPLAY.TraceEvent(
                2, 1, "work_quantum", dependencies=[(0, "program_order")]
            ),
            REPLAY.TraceEvent(
                3,
                2,
                "work",
                dependencies=[(1, "program_order"), (2, "drd_happens_before")],
            ),
            REPLAY.TraceEvent(4, 1, "futex_wake", dependencies=[(2, "program_order")]),
            REPLAY.TraceEvent(
                5,
                2,
                "futex_resume",
                dependencies=[(3, "program_order"), (4, "futex_wake")],
            ),
            REPLAY.TraceEvent(
                6, 1, "thread_finish", dependencies=[(4, "program_order")]
            ),
            REPLAY.TraceEvent(
                7, 2, "thread_finish", dependencies=[(5, "program_order")]
            ),
        ]
        costs = {1: 100.0, 2: 80.0}
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "events.jsonl"
            REPLAY.write_event_stream(event_path, events)
            with NativeReplayEngine(
                Path(os.environ["VC_THREAD_SYNC_REPLAY_BIN"])
            ) as engine:
                engine.load_graph("parity", event_path)
                attributions = []
                jobs = []
                expected = {}
                for residual in (0.0, 0.5, 1.0):
                    for split in ("front", "equal", "back"):
                        attribution_id = f"r{residual:g}/{split}"
                        attributions.append(
                            attribution_request(attribution_id, costs, residual, split)
                        )
                        REPLAY.assign_costs(events, costs, residual, split)
                        for tie in ("fifo", "round_robin"):
                            for cores in (1, 2):
                                job_id = f"{attribution_id}/{tie}/{cores}"
                                jobs.append(
                                    replay_request(
                                        job_id,
                                        attribution_id,
                                        cores,
                                        tie,
                                        wake_latency=3.0,
                                        cross_thread_latency=7.0,
                                        replay_idle_scale=0.75,
                                        dependency_excess_scale=0.25,
                                    )
                                )
                                expected[job_id] = REPLAY.simulate_adjusted(
                                    events,
                                    cores,
                                    tie,
                                    wake_latency=3.0,
                                    cross_thread_latency=7.0,
                                    replay_idle_scale=0.75,
                                    dependency_excess_scale=0.25,
                                )
                engine.register_attributions("parity", attributions)
                actual = engine.replay_batch("parity", jobs)
        self.assertEqual(actual.keys(), expected.keys())
        for job_id in expected:
            self.assertEqual(actual[job_id].keys(), expected[job_id].keys())
            for name, value in expected[job_id].items():
                self.assertTrue(
                    math.isclose(
                        actual[job_id][name], value, rel_tol=1e-12, abs_tol=1e-9
                    ),
                    f"{job_id} {name}: {actual[job_id][name]} != {value}",
                )


if __name__ == "__main__":
    unittest.main()
