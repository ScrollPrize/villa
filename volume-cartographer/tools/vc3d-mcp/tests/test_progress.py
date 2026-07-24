"""Connection lifecycle, job waiting, progress, and schema tests."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from tests.support import FakeAgentBridgeServer
from vc3d_mcp import core
from vc3d_mcp.bridge_client import (
    BridgeClient,
    BridgeClientConfig,
    BridgeConnectionError,
)
from vc3d_mcp.tools.atlas import vc3d_atlas_search_start
from vc3d_mcp.tools.jobs import vc3d_wait_job
from vc3d_mcp.tools.segmentation import (
    vc3d_grow_segment,
    vc3d_run_trace,
    vc3d_save_segment,
)

class _FakeCtx:
    """Stand-in for a FastMCP Context: records progress reports, or raises a
    preset exception to model an unavailable/failing progress sink."""

    def __init__(self, fail: BaseException | None = None) -> None:
        self.reports: list[tuple[int, str | None]] = []
        self._fail = fail

    async def report_progress(self, progress, total=None, message=None) -> None:
        if self._fail is not None:
            raise self._fail
        self.reports.append((progress, message))


class _HangingCtx:
    def __init__(self) -> None:
        self.calls = 0

    async def report_progress(self, progress, total=None, message=None) -> None:
        self.calls += 1
        await asyncio.Event().wait()


class _SlowCtx(_FakeCtx):
    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay
        self.calls = 0

    async def report_progress(self, progress, total=None, message=None) -> None:
        self.calls += 1
        await asyncio.sleep(self.delay)
        await super().report_progress(progress, total, message)


class BridgeClientLifecycleTest(unittest.IsolatedAsyncioTestCase):
    """Connection/reader lifecycle: EOF reset, connect race, cancel leak,
    write-failure invalidation, idempotent close."""

    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-mcp-life-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-agent-bridge.sock")
        self.fake_server = FakeAgentBridgeServer(self.socket_path)
        await self.fake_server.start()
        self.client = BridgeClient(BridgeClientConfig(socket=self.socket_path, request_timeout=5))

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.fake_server.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def _wait_until(self, predicate, timeout=2.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("condition not met within timeout")

    async def test_reconnects_after_peer_eof(self) -> None:
        self.assertEqual((await self.client.call("ping"))["protocolVersion"], 1)
        # Drop the peer: stopping the server EOFs the client's reader.
        await self.fake_server.stop()
        await self._wait_until(lambda: self.client.connected is False)
        self.assertFalse(self.client.connected)
        # Bring the server back and confirm the next call transparently reconnects.
        self.fake_server = FakeAgentBridgeServer(self.socket_path)
        await self.fake_server.start()
        result = await self.client.call("ping")
        self.assertEqual(result["pong"], True)

    async def test_two_concurrent_first_calls_open_one_connection(self) -> None:
        results = await asyncio.gather(self.client.call("ping"), self.client.call("ping"))
        self.assertEqual(len(results), 2)
        self.assertEqual(self.fake_server.connections, 1)

    async def test_cancelled_call_leaves_pending_empty(self) -> None:
        task = asyncio.create_task(self.client.call("never_replies"))
        await self._wait_until(
            lambda: self.client._conn is not None and len(self.client._conn.pending) == 1
        )
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.client._conn.pending, {})

    async def test_write_failure_resets_connection(self) -> None:
        await self.client.call("ping")  # establish a live connection
        real_writer = self.client._conn.writer

        class BadWriter:
            def is_closing(self):
                return False

            def write(self, data):
                raise OSError("simulated write failure")

            async def drain(self):
                return None

            def close(self):
                real_writer.close()

        self.client._conn.writer = BadWriter()
        with self.assertRaises(BridgeConnectionError):
            await self.client.call("ping")
        await real_writer.wait_closed()
        self.assertFalse(self.client.connected)
        self.assertIsNone(self.client._conn)

    async def test_close_is_idempotent(self) -> None:
        await self.client.call("ping")
        await self.client.close()
        await self.client.close()  # must not raise
        self.assertFalse(self.client.connected)
        with self.assertRaises(BridgeConnectionError):
            await self.client.connect()

    async def test_cancel_while_queued_for_write_lock_leaves_pending_empty(self) -> None:
        # A call cancelled while queued for _write_lock (before the write) must
        # not leak its pending entry.
        await self.client.call("ping")  # establish a live connection
        await self.client._write_lock.acquire()
        try:
            task = asyncio.create_task(self.client.call("ping"))
            # It registers in pending, then blocks awaiting the held write lock.
            await self._wait_until(lambda: len(self.client._conn.pending) == 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            self.client._write_lock.release()
        self.assertEqual(self.client._conn.pending, {})

    async def test_connect_after_close_does_not_resurrect(self) -> None:
        await self.client.close()
        with self.assertRaises(BridgeConnectionError):
            await self.client.connect()
        self.assertTrue(self.client._closed)
        self.assertFalse(self.client.connected)

    async def test_close_racing_pending_connect_stays_closed(self) -> None:
        # close() concurrent with an in-flight connect() must not leave the
        # client connected. Delay open_unix_connection so close() runs while
        # connect() is pending; close() then tears down whatever connect built.
        fresh = BridgeClient(BridgeClientConfig(socket=self.socket_path, request_timeout=5))
        started = asyncio.Event()
        release = asyncio.Event()
        real_open = asyncio.open_unix_connection

        async def slow_open(*args, **kwargs):
            started.set()
            await release.wait()
            return await real_open(*args, **kwargs)

        with mock.patch(
            "vc3d_mcp.bridge_client.asyncio.open_unix_connection", slow_open
        ):
            conn_task = asyncio.create_task(fresh.connect())
            await asyncio.wait_for(started.wait(), timeout=2.0)
            close_task = asyncio.create_task(fresh.close())
            await asyncio.sleep(0.02)  # let close() set _closed + block on the lock
            release.set()
            try:
                await asyncio.wait_for(conn_task, timeout=2.0)
            except BridgeConnectionError:
                pass
            await asyncio.wait_for(close_task, timeout=2.0)
        self.assertTrue(fresh._closed)
        self.assertFalse(fresh.connected)


class WaitAndProgressTest(unittest.IsolatedAsyncioTestCase):
    """Notification-driven waits with bounded replay and status fallback."""

    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-mcp-wait-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-agent-bridge.sock")
        self.fake_server = FakeAgentBridgeServer(self.socket_path)
        await self.fake_server.start()
        self.client = core.configure_client(self.socket_path, request_timeout=5)
        # Poll fast so wait-loop tests run in milliseconds, not seconds.
        self._orig_poll = core.POLL_INTERVAL_S
        core.POLL_INTERVAL_S = 0.01

    async def asyncTearDown(self) -> None:
        core.POLL_INTERVAL_S = self._orig_poll
        await self.client.close()
        await self.fake_server.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_run_trace_no_wait_returns_job_id(self) -> None:
        result = await vc3d_run_trace("seg-1", wait=False)
        self.assertEqual(result["jobId"], "job-5")
        self.assertNotIn("state", result)

    async def test_run_trace_wait_returns_terminal_status(self) -> None:
        result = await vc3d_run_trace("seg-1", wait=True)
        self.assertEqual(result["jobId"], "job-5")
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("consoleTail", result)

    async def test_atlas_search_no_wait_returns_job_id(self) -> None:
        result = await vc3d_atlas_search_start(wait=False)
        self.assertEqual(result["jobId"], "job-6")
        self.assertNotIn("state", result)

    async def test_atlas_search_wait_returns_terminal_status(self) -> None:
        result = await vc3d_atlas_search_start(wait=True)
        self.assertEqual(result["jobId"], "job-6")
        self.assertEqual(result["state"], "succeeded")

    async def test_wait_reads_authoritative_job_status(self) -> None:
        result = await vc3d_run_trace("seg-1", wait=True)
        self.assertEqual(result["state"], "succeeded")
        polls = [r for r in self.fake_server.received_requests if r.get("method") == "job.status"]
        self.assertGreaterEqual(len(polls), 1)

    async def test_progress_forwarded_in_sequence(self) -> None:
        ctx = _FakeCtx()
        result = await vc3d_grow_segment(steps=1, wait=True, ctx=ctx)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(
            [m for _, m in ctx.reports[:2]], ["line A", "line B"]
        )
        self.assertEqual([s for s, _ in ctx.reports], [1, 2, 3])
        self.assertEqual(
            json.loads(ctx.reports[-1][1]),
            {
                "message": "finished",
                "outputPath": None,
                "phase": "finished",
                "result": None,
                "success": True,
            },
        )
        polls = [r for r in self.fake_server.received_requests if r.get("method") == "job.status"]
        self.assertGreaterEqual(len(polls), 1)

    async def test_buffered_progress_replays_after_job_finished(self) -> None:
        started = await vc3d_grow_segment(steps=1, wait=False)
        await asyncio.sleep(0.1)
        ctx = _FakeCtx()
        result = await vc3d_wait_job(started["jobId"], ctx=ctx)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual([s for s, _ in ctx.reports], [1, 2, 3])
        self.assertEqual([m for _, m in ctx.reports[:2]], ["line A", "line B"])

    async def test_update_between_subscribe_and_snapshot_is_not_duplicated(self) -> None:
        started = await vc3d_grow_segment(steps=1, wait=False)
        await asyncio.sleep(0.1)
        self.fake_server.rebroadcast_latest_on_status = True
        ctx = _FakeCtx()
        result = await vc3d_wait_job(started["jobId"], ctx=ctx)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual([s for s, _ in ctx.reports], [1, 2, 3])

    async def test_terminal_without_progress_uses_status_fallback(self) -> None:
        self.fake_server.emit_job_progress = False
        ctx = _FakeCtx()
        result = await vc3d_grow_segment(steps=1, wait=True, ctx=ctx)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(ctx.reports, [])

    async def test_wait_true_fails_promptly_on_bridge_disconnect(self) -> None:
        self.fake_server.finish_jobs = False  # job never reaches a terminal state
        task = asyncio.create_task(vc3d_run_trace("seg-1", wait=True))
        await asyncio.sleep(0.05)  # let the job start + the first poll happen
        await self.fake_server.stop()  # peer EOF -> the next poll must fail
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(task, timeout=2.0)

    async def test_wait_cap_returns_timed_out(self) -> None:
        self.fake_server.finish_jobs = False  # stays running past the cap
        orig_cap = core.DEFAULT_WAIT_TIMEOUT_S
        core.DEFAULT_WAIT_TIMEOUT_S = 0.05
        try:
            result = await asyncio.wait_for(vc3d_run_trace("seg-1", wait=True), timeout=3.0)
        finally:
            core.DEFAULT_WAIT_TIMEOUT_S = orig_cap
        self.assertTrue(result.get("waitTimedOut"))
        self.assertEqual(result["jobId"], "job-5")

    async def test_wait_job_returns_terminal_status(self) -> None:
        # Launch a job with wait=false so the fake server holds a live,
        # simulated record, then park on it: vc3d_wait_job must poll it to the
        # terminal state and return the merged job.status.
        started = await vc3d_grow_segment(steps=1, wait=False)
        job_id = started["jobId"]
        result = await vc3d_wait_job(job_id)
        self.assertEqual(result["jobId"], job_id)
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("consoleTail", result)

    async def test_wait_job_forwards_console_progress(self) -> None:
        started = await vc3d_grow_segment(steps=1, wait=False)
        ctx = _FakeCtx()
        result = await vc3d_wait_job(started["jobId"], ctx=ctx)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual([m for _, m in ctx.reports[:2]], ["line A", "line B"])

    async def test_wait_job_cap_returns_timed_out(self) -> None:
        self.fake_server.finish_jobs = False  # stays running past the cap
        started = await vc3d_grow_segment(steps=1, wait=False)
        orig_cap = core.DEFAULT_WAIT_TIMEOUT_S
        core.DEFAULT_WAIT_TIMEOUT_S = 0.05
        try:
            result = await asyncio.wait_for(
                vc3d_wait_job(started["jobId"]), timeout=3.0
            )
        finally:
            core.DEFAULT_WAIT_TIMEOUT_S = orig_cap
        self.assertTrue(result.get("waitTimedOut"))
        self.assertEqual(result["jobId"], started["jobId"])

    async def test_noop_ctx_does_not_fail_the_tool(self) -> None:
        # ctx=None models an MCP client with no progress support.
        result = await vc3d_run_trace("seg-1", wait=True, ctx=None)
        self.assertEqual(result["state"], "succeeded")

    async def test_progress_report_failure_does_not_fail_the_tool(self) -> None:
        ctx = _FakeCtx(fail=RuntimeError("progress sink is down"))
        result = await vc3d_run_trace("seg-1", wait=True, ctx=ctx)
        self.assertEqual(result["state"], "succeeded")

    async def test_progress_report_timeout_disables_further_reports(self) -> None:
        ctx = _HangingCtx()
        original_timeout = core.PROGRESS_REPORT_TIMEOUT_S
        core.PROGRESS_REPORT_TIMEOUT_S = 0.01
        try:
            result = await vc3d_grow_segment(steps=1, wait=True, ctx=ctx)
        finally:
            core.PROGRESS_REPORT_TIMEOUT_S = original_timeout
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(ctx.calls, 1)

    async def test_buffered_replay_has_one_timeout_budget(self) -> None:
        started = await vc3d_grow_segment(steps=1, wait=False)
        await asyncio.sleep(0.1)
        record = self.fake_server._jobs[started["jobId"]]
        record["progressHistory"] = [
            {
                "jobId": started["jobId"],
                "kind": record["kind"],
                "seq": seq,
                "phase": "output",
                "message": f"line {seq}",
            }
            for seq in range(1, 65)
        ]
        ctx = _SlowCtx(delay=0.02)
        original_timeout = core.PROGRESS_REPORT_TIMEOUT_S
        core.PROGRESS_REPORT_TIMEOUT_S = 0.03
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        try:
            result = await vc3d_wait_job(started["jobId"], ctx=ctx)
        finally:
            core.PROGRESS_REPORT_TIMEOUT_S = original_timeout
        self.assertEqual(result["state"], "succeeded")
        self.assertLess(loop.time() - started_at, 0.15)
        self.assertLessEqual(ctx.calls, 2)

    async def test_progress_report_cancellation_is_not_swallowed(self) -> None:
        ctx = _FakeCtx(fail=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await vc3d_run_trace("seg-1", wait=True, ctx=ctx)

    async def test_save_segment_forwards_progress_context(self) -> None:
        ctx = _FakeCtx()
        result = await vc3d_save_segment(wait=True, ctx=ctx)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual([s for s, _ in ctx.reports], [1, 2, 3])


class ToolSchemaTest(unittest.IsolatedAsyncioTestCase):
    """Official-SDK schema assertions over the whole registered tool surface."""

    # Assert the exact registered count so schema drift (added/removed tools) is caught.
    EXPECTED_TOOL_COUNT = 118

    async def test_tool_surface_and_schemas(self) -> None:
        tools = await core.mcp.list_tools()
        by_name = {t.name: t.inputSchema for t in tools}
        self.assertEqual(len(tools), self.EXPECTED_TOOL_COUNT)

        # FastMCP-injected Context must never appear in any input schema.
        for name, schema in by_name.items():
            props = schema.get("properties") or {}
            self.assertNotIn("ctx", props, f"{name} leaked ctx into its schema")

        # Both formerly-missing wait params are present.
        self.assertIn("wait", by_name["vc3d_run_trace"]["properties"])
        self.assertIn("wait", by_name["vc3d_atlas_search_start"]["properties"])

        # A sampling of required enum schemas are emitted as JSON-schema enums.
        grow = by_name["vc3d_grow_segment"]["properties"]
        self.assertEqual(grow["method"]["enum"], ["tracer", "corrections", "patch_tracer"])
        self.assertEqual(
            grow["direction"]["enum"], ["all", "up", "down", "left", "right", "fill"]
        )
        self.assertEqual(
            by_name["vc3d_switch_workspace"]["properties"]["name"]["enum"],
            ["main", "lasagna", "fiber_slice"],
        )
        self.assertEqual(
            by_name["vc3d_atlas_search_start"]["properties"]["mode"]["enum"],
            ["atlas_to_non_atlas", "non_atlas_only"],
        )
        self.assertEqual(
            by_name["vc3d_render_tifxyz"]["properties"]["output_format"]["enum"],
            ["zarr", "tif_stack"],
        )
