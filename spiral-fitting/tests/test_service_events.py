"""Tests for the structured /events stream.

Covers event sequencing, cursor overruns, metric coalescing and console
output reaching clients without duplicating structured records.
"""

import json
import unittest
from spiral_service import ServiceEventBuffer, ServiceState
from service_fixtures import HttpServiceFixture


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


class EventIngestTests(unittest.TestCase):
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


class EventHttpTests(HttpServiceFixture):
    def test_events_cursor_validation_and_authentication(self):
        status, _, _ = self.request("GET", "/events?cursor=not-a-number")
        self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/events?cursor=-1")
        self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/events?cursor=0&limit=0")
        self.assertEqual(status, 400)
        status, _, _ = self.request("GET", "/events", token=None)
        self.assertEqual(status, 401)

    def test_console_lines_reach_clients_only_as_log_events(self):
        self.state.logs.write("stdout", "PROGRESS Optimizing — 1/10 iterations\n")
        self.state.logs.write("stdout", "step 200: loss = 12.5, patch = 3.0\n")
        self.state.logs.write("stdout", "hello\n")
        status, payload, _ = self.request("GET", "/events?cursor=0")
        self.assertEqual(status, 200)
        records = [record for record in json.loads(payload)["events"]
                   if record["kind"] == "log"]
        self.assertEqual([record["text"] for record in records], ["hello"])
        self.assertEqual(records[0]["source"], "stdout")
