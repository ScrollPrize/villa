"""Tests for the structured /events stream.

Covers event sequencing and cursor-overrun semantics, progress/metric
coalescing, duplicate suppression between structured records and the
tee-captured log relay, /logs schema compatibility, and rank routing for
child-rank records.
"""

import json
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import spiral_service
from spiral_service import (ServiceEventBuffer, ServiceLogBuffer,
                            ServiceState)
from test_spiral_service_v2 import HttpServiceFixture


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _running_status(iteration=1, step=1, total=10, stage="Optimizing"):
    return {
        "state": "Running",
        "phase": stage,
        "current_iteration": iteration,
        "target_iteration": total,
        "latest_metrics": {"total_loss": 1.5, "learning_rate": 3e-5,
                           "losses": {"patch": 1.0}},
        "warnings": [],
        "error": None,
        "progress": {
            "operation": "optimizing",
            "stage_name": stage,
            "detail": None,
            "step": step,
            "total_steps": total,
            "unit": "iterations",
            "elapsed_seconds": 0.25 * iteration,
            "eta_seconds": None,
        },
    }


class EventBufferTests(unittest.TestCase):
    def test_sequences_are_monotonic_and_reads_are_incremental(self):
        events = ServiceEventBuffer()
        events.append("log", "one")
        events.append("log", "two")
        first = events.read_after(0)
        self.assertEqual([e["sequence"] for e in first["events"]], [1, 2])
        self.assertEqual(first["next_cursor"], 2)
        self.assertEqual(first["latest_sequence"], 2)
        self.assertFalse(first["overrun"])
        self.assertFalse(first["cursor_reset"])
        events.append("log", "three")
        second = events.read_after(first["next_cursor"])
        self.assertEqual([e["text"] for e in second["events"]], ["three"])
        self.assertEqual(second["next_cursor"], 3)

    def test_records_carry_the_full_schema(self):
        events = ServiceEventBuffer()
        events.append("error", "boom", severity="error", source="fitter",
                      rank=2, session_generation=7, operation="optimizing",
                      payload={"a": 1})
        record = events.read_after(0)["events"][0]
        self.assertEqual(
            set(record), {"sequence", "timestamp", "severity", "kind",
                          "source", "rank", "session_generation",
                          "operation", "text", "payload"})
        self.assertEqual(record["kind"], "error")
        self.assertEqual(record["severity"], "error")
        self.assertEqual(record["rank"], 2)
        self.assertEqual(record["session_generation"], 7)
        self.assertEqual(record["operation"], "optimizing")
        self.assertEqual(record["payload"], {"a": 1})

    def test_overrun_is_reported_when_cursor_precedes_the_ring(self):
        events = ServiceEventBuffer(max_entries=2)
        for text in ("one", "two", "three"):
            events.append("log", text)
        result = events.read_after(0)
        self.assertTrue(result["overrun"])
        self.assertEqual(result["dropped"], 1)
        self.assertEqual(result["dropped_from"], 1)
        self.assertEqual([e["text"] for e in result["events"]],
                         ["two", "three"])
        # A cursor at the ring start reads without an overrun indication.
        aligned = events.read_after(1)
        self.assertFalse(aligned["overrun"])
        self.assertEqual(aligned["dropped"], 0)
        self.assertIsNone(aligned["dropped_from"])

    def test_stale_future_cursor_resets_to_the_ring_start(self):
        events = ServiceEventBuffer()
        events.append("log", "one")
        result = events.read_after(50)
        self.assertTrue(result["cursor_reset"])
        self.assertEqual([e["text"] for e in result["events"]], ["one"])

    def test_read_limit_bounds_each_page(self):
        events = ServiceEventBuffer()
        for index in range(5):
            events.append("log", str(index))
        page = events.read_after(0, limit=2)
        self.assertEqual([e["text"] for e in page["events"]], ["0", "1"])
        self.assertEqual(page["next_cursor"], 2)
        rest = events.read_after(page["next_cursor"], limit=100)
        self.assertEqual([e["text"] for e in rest["events"]],
                         ["2", "3", "4"])

    def test_coalescing_keeps_at_most_one_record_per_interval(self):
        clock = FakeClock()
        events = ServiceEventBuffer(coalesce_seconds=1.0, clock=clock)
        for iteration in range(10):
            events.append("metric", f"iteration {iteration}",
                          coalesce_key=("metric", 0),
                          payload={"iteration": iteration})
            clock.advance(0.05)
        first = events.read_after(0)
        self.assertEqual(len(first["events"]), 1)
        self.assertEqual(first["events"][0]["payload"]["iteration"], 0)
        # Once the interval elapses, the newest pending record is flushed.
        clock.advance(1.0)
        flushed = events.read_after(first["next_cursor"])
        self.assertEqual(len(flushed["events"]), 1)
        self.assertEqual(flushed["events"][0]["payload"]["iteration"], 9)

    def test_forced_records_bypass_coalescing(self):
        clock = FakeClock()
        events = ServiceEventBuffer(coalesce_seconds=1.0, clock=clock)
        events.append("progress", "start", coalesce_key=("progress", 0))
        clock.advance(0.1)
        events.append("progress", "done", coalesce_key=("progress", 0),
                      force=True)
        self.assertEqual(
            [e["text"] for e in events.read_after(0)["events"]],
            ["start", "done"])


class EventIngestTests(unittest.TestCase):
    def test_progress_update_produces_exactly_one_record(self):
        state = ServiceState()
        status = _running_status()
        state._session_event(0, status)
        # A repeated identical snapshot (elapsed time aside) is not a new
        # progress event.
        repeat = json.loads(json.dumps(status))
        repeat["progress"]["elapsed_seconds"] = 99.0
        state._session_event(0, repeat)
        records = [record for record in state.events.read_after(0)["events"]
                   if record["kind"] == "progress"]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["operation"], "optimizing")
        self.assertEqual(record["rank"], 0)
        self.assertEqual(record["payload"]["step"], 1)
        self.assertNotIn("eta_seconds", record["payload"])

    def test_structured_console_lines_do_not_reenter_as_log_events(self):
        events = ServiceEventBuffer()
        logs = ServiceLogBuffer(events=events)
        logs.write("stdout", "PROGRESS Optimizing — 1/10 iterations\n")
        logs.write("stdout", "step 200: loss = 12.5, patch = 3.0\n")
        logs.write("stdout", "useful fitter output\n")
        # Only lines not already covered by structured progress/metric
        # records become log events.
        log_events = [record for record in events.read_after(0)["events"]
                      if record["kind"] == "log"]
        self.assertEqual([record["text"] for record in log_events],
                         ["useful fitter output"])
        self.assertEqual(log_events[0]["source"], "stdout")

    def test_metric_records_coalesce_to_the_latest_iteration(self):
        clock = FakeClock()
        state = ServiceState(events=ServiceEventBuffer(
            coalesce_seconds=1.0, clock=clock))
        for iteration in range(1, 11):
            state._session_event(
                0, _running_status(iteration=iteration, step=iteration))
            clock.advance(0.05)
        clock.advance(1.0)
        metrics = [record for record in state.events.read_after(0)["events"]
                   if record["kind"] == "metric"]
        self.assertEqual([record["payload"]["iteration"]
                          for record in metrics], [1, 10])
        self.assertEqual(metrics[0]["payload"]["total_loss"], 1.5)

    def test_child_rank_records_carry_their_rank(self):
        state = ServiceState()
        state._session_event(1, _running_status())
        records = state.events.read_after(0)["events"]
        self.assertTrue(records)
        self.assertTrue(all(record["rank"] == 1 for record in records))

    def test_error_state_emits_one_error_record(self):
        state = ServiceState()
        status = {"state": "Error", "error": "RuntimeError: boom",
                  "latest_metrics": {}, "progress": None}
        state._session_event(0, status)
        state._session_event(0, status)
        errors = [record for record in state.events.read_after(0)["events"]
                  if record["kind"] == "error"]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["severity"], "error")
        self.assertEqual(errors[0]["text"], "RuntimeError: boom")

    def test_preview_publish_progress_is_published_as_service_events(self):
        state = ServiceState()
        state._preview.claim("spiral-test-1", 3)
        state._update_preview_publish(
            3, state="indexing", stage_name="Indexing preview files",
            step=0, total_steps=4, overall_progress=0.0)
        records = state.events.read_after(0)["events"]
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["kind"], "progress")
        self.assertEqual(record["source"], "service")
        self.assertEqual(record["operation"], "publishing_preview")
        self.assertEqual(record["payload"]["generation"], 3)


class RuntimeEventPlumbingTests(unittest.TestCase):
    def test_in_process_session_reports_as_rank_zero(self):
        from spiral_runtime import InteractiveFitSession

        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = "Running"
        session._phase = "Optimizing"
        session._completed = 3
        session._target = 5
        session._pending = 2
        session._latest_metrics = {"total_loss": 1.0}
        session._warnings = []
        session._error = None
        session._preview_manifest = None
        session._preview_generation = 0
        session._context = None
        session.input_manifest = {}
        session._run_config = None
        session._run_config_limits = None
        session._default_advanced_config = None
        session._applied_config = None
        received = []
        session._event_callback = \
            lambda rank, status: received.append((rank, status))
        session._status_callback = None
        session._publish_status()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0][0], 0)
        self.assertEqual(received[0][1]["current_iteration"], 3)

    def test_distributed_proxy_routes_events_with_their_rank(self):
        from spiral_runtime import DistributedInteractiveFitSession

        proxy = DistributedInteractiveFitSession.__new__(
            DistributedInteractiveFitSession)
        received = []
        proxy._event_callback = \
            lambda rank, status: received.append((rank, status))
        proxy._publish_rank_event(2, {"state": "Running"})
        self.assertEqual(received, [(2, {"state": "Running"})])
        # A callback failure never kills the coordinator listener.
        proxy._event_callback = lambda rank, status: 1 / 0
        proxy._publish_rank_event(1, {})


class EventHttpTests(HttpServiceFixture):
    def test_events_endpoint_serves_incremental_reads(self):
        self.state.events.append("log", "hello")
        status, payload, _ = self.request("GET", "/events")
        self.assertEqual(status, 200)
        first = json.loads(payload)
        self.assertEqual([e["text"] for e in first["events"]], ["hello"])
        self.assertEqual(first["next_cursor"], 1)
        status, payload, _ = self.request(
            "GET", f"/events?cursor={first['next_cursor']}")
        self.assertEqual(json.loads(payload)["events"], [])

    def test_events_cursor_validation_and_authentication(self):
        status, _, _ = self.request("GET", "/events?cursor=not-a-number")
        self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/events?cursor=-1")
        self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/events?cursor=0&limit=0")
        self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/events", token=None)
        self.assertEqual(status, 401)

    def test_events_limit_pages_the_stream(self):
        for index in range(3):
            self.state.events.append("log", str(index))
        status, payload, _ = self.request("GET", "/events?cursor=0&limit=2")
        self.assertEqual(status, 200)
        page = json.loads(payload)
        self.assertEqual([e["text"] for e in page["events"]], ["0", "1"])
        self.assertEqual(page["latest_sequence"], 3)

    def test_console_lines_reach_clients_only_as_log_events(self):
        """The tee has no relay of its own; /events is the whole surface."""
        self.assertEqual(self.request("GET", "/logs?after=0")[0], 404)
        self.state.logs.write("stdout", "hello\n")
        status, payload, _ = self.request("GET", "/events?cursor=0")
        self.assertEqual(status, 200)
        records = [record for record in json.loads(payload)["events"]
                   if record["kind"] == "log"]
        self.assertEqual([record["text"] for record in records], ["hello"])
        self.assertEqual(records[0]["source"], "stdout")


if __name__ == "__main__":
    unittest.main()
