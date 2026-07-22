#!/usr/bin/env python3
"""Self-test for vc3d-mcp, run without a real VC3D instance.

Stands up a trivial fake JSON-RPC-over-AF_UNIX-socket server (standing in for
apps/VC3D/agent_bridge/AgentBridgeServer, so the suite stays hermetic -- no Qt
app needed) and confirms:

  * BridgeClient can connect, send a request, and parse a response
    (including JSON-RPC error replies, per SPEC.md section 2.5);
  * job.progress notifications (SPEC.md section 3.18) are picked up by the
    background reader and folded into JobTracker, including the console
    tail and the "wait until finished" convenience;
  * the actual MCP tool functions in vc3d_mcp.server wire all of the above
    together correctly (configure_client -> tool call -> bridge round trip).

Run directly:
    cd tools/vc3d-mcp && python3 test_mcp_bridge.py -v
or via unittest discovery:
    python3 -m unittest test_mcp_bridge -v
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vc3d_mcp.bridge_client import (  # noqa: E402
    BridgeClient,
    BridgeClientConfig,
    BridgeConnectionError,
    BridgeError,
    discover_registry_socket,
)
from vc3d_mcp import server as server_module  # noqa: E402


class FakeAgentBridgeServer:
    """Minimal stand-in for AgentBridgeServer: AF_UNIX, newline-delimited
    JSON-RPC 2.0, a couple of canned methods, and the ability to simulate a
    segmentation.grow job's job.progress notification sequence."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._server: asyncio.base_events.Server | None = None
        self._writers: list[asyncio.StreamWriter] = []
        self.received_requests: list[dict] = []
        # job_id -> kind, so job.status/job.progress can echo the right kind for
        # both the segmentation.grow and lasagna.optimize job flows.
        self._job_kinds: dict[str, str] = {}
        # Segments that have been materialized (fetched). "seg-ph" starts as an
        # unfetched open-data placeholder; fetching it adds it here so a
        # subsequent segments.activate succeeds -- the fetch+retry compose.
        self._fetched: set[str] = set()
        # Per-plane rotation state for viewer.rotate (mirrors the C++ controller's
        # _segXZRotationDeg / _segYZRotationDeg members).
        self._rotation: dict[str, float] = {}
        # Axis-aligned slice mode (viewer.set_axis_aligned_slices / state.get).
        self._axis_enabled: bool = False
        # Same-wrap annotation state (SPEC §3.9d): the mode checkbox and whether a
        # chunked viewer holds an uncommitted preview (seeded by shift-click).
        self._wrap_enabled: bool = False
        self._wrap_has_preview: bool = False
        # segmentation.save: when True, model the "nothing to flush" idle response
        # (jobId:null); when False, model a running "autosave" job that succeeds.
        self.save_idle: bool = False
        # Count of accepted client connections (for the connect-race test).
        self.connections: int = 0
        # When False, simulated jobs emit started/output but never "finished",
        # so a wait: true call blocks until the peer drops (disconnect test).
        self.finish_jobs: bool = True

    async def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path)

    async def stop(self) -> None:
        for w in list(self._writers):
            w.close()
        if self._server is not None:
            self._server.close()
            # Bound wait_closed(): on some CPython versions Server.wait_closed()
            # can block indefinitely if a client opened and immediately dropped a
            # connection (e.g. connect()'s abort-when-closed path) so it was
            # never fully accepted. That's a harness/stdlib quirk, not product
            # behavior; don't let it hang teardown.
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        self._writers.append(writer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                await self._handle_line(line, writer)
        finally:
            if writer in self._writers:
                self._writers.remove(writer)

    async def _handle_line(self, raw: bytes, writer: asyncio.StreamWriter) -> None:
        msg = json.loads(raw.decode("utf-8"))
        self.received_requests.append(msg)
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "ping":
            await self._reply(writer, req_id, result={"pong": True, "pid": 4242, "version": "test-0.0"})
        elif method == "state.get":
            await self._reply(writer, req_id, result={"vpkg": None, "volume": None})
        elif method == "segmentation.grow":
            job_id = "job-1"
            self._job_kinds[job_id] = "segmentation.grow"
            await self._reply(writer, req_id, result={"jobId": job_id, "kind": "segmentation.grow"})
            asyncio.create_task(self._simulate_job(job_id, "segmentation.grow"))
        elif method == "segmentation.save":
            if self.save_idle:
                await self._reply(
                    writer, req_id,
                    result={"jobId": None, "kind": "segmentation.save",
                            "state": "idle", "pending": False,
                            "saveInProgress": False, "dirtyAfterSave": False},
                )
            else:
                job_id = "job-4"
                self._job_kinds[job_id] = "segmentation.save"
                await self._reply(
                    writer, req_id,
                    result={"jobId": job_id, "source": "autosave",
                            "kind": "segmentation.save", "state": "running",
                            "label": "Save segment"},
                )
                asyncio.create_task(self._simulate_job(job_id, "segmentation.save"))
        elif method == "tracer.run_trace":
            job_id = "job-5"
            self._job_kinds[job_id] = "tracer.run_trace"
            await self._reply(
                writer, req_id,
                result={"jobId": job_id, "kind": "tracer.run_trace",
                        "source": "tool", "outputDir": "/tmp/traces"},
            )
            asyncio.create_task(self._simulate_job(job_id, "tracer.run_trace"))
        elif method == "atlas.search_start":
            job_id = "job-6"
            self._job_kinds[job_id] = "atlas.fiber_search"
            await self._reply(
                writer, req_id,
                result={"jobId": job_id, "kind": "atlas.fiber_search", "source": "atlas"},
            )
            asyncio.create_task(self._simulate_job(job_id, "atlas.fiber_search"))
        elif method == "never_replies":
            return  # deliberately no response (cancel / disconnect tests)
        elif method == "lasagna.start_optimization":
            job_id = "job-2"
            self._job_kinds[job_id] = "lasagna.optimize"
            await self._reply(
                writer,
                req_id,
                result={"jobId": job_id, "kind": "lasagna.optimize", "source": "lasagna"},
            )
            asyncio.create_task(self._simulate_job(job_id, "lasagna.optimize"))
        elif method == "segments.activate":
            seg = params.get("segmentId")
            if seg == "seg-ready" or seg in self._fetched:
                await self._reply(
                    writer, req_id,
                    result={"activated": True,
                            "segment": {"id": seg, "loaded": True, "active": True},
                            "previousSegmentId": None, "alreadyActive": False},
                )
            elif seg == "seg-ph":
                await self._reply(
                    writer, req_id,
                    error={"code": -32005, "message": "Segment could not be activated",
                           "data": {"detail": f"segment {seg} is an open-data "
                                              "placeholder; fetch it first"}},
                )
            else:
                await self._reply(
                    writer, req_id,
                    error={"code": -32007, "message": "Segment not found",
                           "data": {"kind": "segment", "id": seg}},
                )
        elif method == "segments.fetch":
            seg = params.get("segmentId")
            if seg == "seg-ready" or seg in self._fetched:
                await self._reply(
                    writer, req_id,
                    result={"fetched": True, "alreadyMaterialized": True,
                            "segment": {"id": seg, "placeholder": False}},
                )
            elif seg == "seg-ph":
                job_id = "job-3"
                self._job_kinds[job_id] = "segments.fetch"
                self._fetched.add(seg)  # materialized once the fetch is requested
                await self._reply(
                    writer, req_id,
                    result={"jobId": job_id, "source": "catalog", "fetched": False,
                            "alreadyMaterialized": False,
                            "segment": {"id": seg, "placeholder": True}},
                )
                asyncio.create_task(self._simulate_job(job_id, "segments.fetch"))
            else:
                await self._reply(
                    writer, req_id,
                    error={"code": -32007, "message": "Segment not found",
                           "data": {"kind": "segment", "id": seg}},
                )
        elif method == "viewer.rotate":
            plane = params.get("plane")
            norm = {"xz": "seg xz", "yz": "seg yz",
                    "seg xz": "seg xz", "seg yz": "seg yz"}.get(plane)
            if norm is None:
                await self._reply(
                    writer, req_id,
                    error={"code": -32602, "message": "unknown plane",
                           "data": {"param": "plane", "value": plane}},
                )
            else:
                prev = self._rotation.get(norm, 0.0)
                deg = params.get("degrees", 0.0)
                target = prev + deg if params.get("relative", True) else deg
                # match the C++ remainder(., 360) normalization
                target = target - 360.0 * round(target / 360.0)
                self._rotation[norm] = target
                await self._reply(
                    writer, req_id,
                    result={"plane": norm, "degrees": target,
                            "previousDegrees": prev,
                            "relative": params.get("relative", True)},
                )
        elif method == "viewer.set_axis_aligned_slices":
            enabled = params.get("enabled")
            if not isinstance(enabled, bool):
                await self._reply(
                    writer, req_id,
                    error={"code": -32602, "message": "enabled (bool) is required",
                           "data": {"param": "enabled"}},
                )
            else:
                self._axis_enabled = enabled  # idempotent set, like setChecked
                await self._reply(writer, req_id, result={"enabled": self._axis_enabled})
        elif method == "wrap_annotation.set_mode":
            self._wrap_enabled = bool(params.get("enabled"))
            await self._reply(writer, req_id, result={"enabled": self._wrap_enabled})
        elif method == "wrap_annotation.commit":
            if not self._wrap_enabled:
                await self._reply(
                    writer, req_id,
                    error={"code": -32002,
                           "message": "same-wrap annotation mode is not enabled"},
                )
            else:
                had_preview = self._wrap_has_preview
                committed = self._wrap_enabled and self._wrap_has_preview
                self._wrap_has_preview = False  # commit consumes the preview
                await self._reply(
                    writer, req_id,
                    result={"committed": committed, "hadPreview": had_preview},
                )
        elif method == "wrap_annotation.undo":
            undone = self._wrap_has_preview
            self._wrap_has_preview = False
            await self._reply(writer, req_id, result={"undone": undone})
        elif method == "job.status":
            job_id = params.get("jobId", "job-1")
            await self._reply(
                writer,
                req_id,
                result={
                    "jobId": job_id,
                    "kind": self._job_kinds.get(job_id, "segmentation.grow"),
                    "label": "Grow",
                    "state": "succeeded",
                    "message": "finished",
                    "outputPath": None,
                    "consoleTail": ["line A", "line B"],
                },
            )
        elif method == "will_error":
            await self._reply(
                writer,
                req_id,
                error={
                    "code": -32003,
                    "message": "INVALID_COORDINATES",
                    "data": {"point": {"x": 1.0, "y": 2.0, "z": 3.0}},
                },
            )
        else:
            await self._reply(writer, req_id, error={"code": -32601, "message": "Method not found"})

    async def _simulate_job(self, job_id: str, kind: str = "segmentation.grow") -> None:
        await asyncio.sleep(0.02)
        await self._broadcast(
            {"jobId": job_id, "kind": kind, "phase": "started", "message": "starting"}
        )
        await asyncio.sleep(0.02)
        await self._broadcast(
            {"jobId": job_id, "kind": kind, "phase": "output", "message": "line A\nline B"}
        )
        if not self.finish_jobs:
            return  # never terminate: exercises wait-until-disconnect
        await asyncio.sleep(0.02)
        await self._broadcast(
            {
                "jobId": job_id,
                "kind": kind,
                "phase": "finished",
                "success": True,
                "outputPath": None,
            }
        )

    async def _reply(self, writer: asyncio.StreamWriter, req_id, *, result=None, error=None) -> None:
        msg: dict = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        await self._send(writer, msg)

    async def _broadcast(self, params: dict) -> None:
        msg = {"jsonrpc": "2.0", "method": "job.progress", "params": params}
        for w in list(self._writers):
            await self._send(w, msg)

    async def _send(self, writer: asyncio.StreamWriter, obj: dict) -> None:
        writer.write((json.dumps(obj) + "\n").encode("utf-8"))
        await writer.drain()


class BridgeClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-mcp-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-agent-bridge.sock")
        self.fake_server = FakeAgentBridgeServer(self.socket_path)
        await self.fake_server.start()
        self.client = BridgeClient(BridgeClientConfig(socket=self.socket_path, request_timeout=5))

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.fake_server.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_connect_send_request_and_parse_response(self) -> None:
        result = await self.client.call("ping")
        self.assertEqual(result, {"pong": True, "pid": 4242, "version": "test-0.0"})

    async def test_jsonrpc_error_reply_becomes_bridge_error(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            await self.client.call("will_error")
        err = ctx.exception
        self.assertEqual(err.code, -32003)
        self.assertEqual(err.message, "INVALID_COORDINATES")
        self.assertEqual(err.data, {"point": {"x": 1.0, "y": 2.0, "z": 3.0}})
        # str(err) must be a JSON object carrying code/message/data, since
        # that's what ends up as the MCP tool-call error text.
        as_json = json.loads(str(err))
        self.assertEqual(as_json["code"], -32003)
        self.assertEqual(as_json["data"]["point"]["x"], 1.0)

    async def test_method_not_found(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            await self.client.call("nope.not_a_real_method")
        self.assertEqual(ctx.exception.code, -32601)

    async def test_job_progress_notifications_feed_tracker(self) -> None:
        result = await self.client.call(
            "segmentation.grow", {"method": "tracer", "direction": "all", "steps": 1}
        )
        job_id = result["jobId"]
        record = await self.client.jobs.wait_finished(job_id, timeout=5)
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "succeeded")
        self.assertTrue(record.success)
        self.assertTrue(any("line A" in line for line in record.console_tail))

    async def test_resolve_socket_path_absolute_existing_path(self) -> None:
        self.assertEqual(BridgeClient.resolve_socket_path(self.socket_path), self.socket_path)

    def test_handshake_line_parsing(self) -> None:
        parsed = BridgeClient.socket_path_from_handshake(
            "VC3D-AGENT-BRIDGE: listening name=vc3d-agent-1234 path=/tmp/vc3d-agent-1234\n"
        )
        self.assertEqual(parsed, ("vc3d-agent-1234", "/tmp/vc3d-agent-1234"))
        self.assertIsNone(BridgeClient.socket_path_from_handshake("not a handshake line"))


class ToolLayerTest(unittest.IsolatedAsyncioTestCase):
    """Exercises the actual @mcp.tool() functions, not just BridgeClient."""

    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-mcp-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-agent-bridge.sock")
        self.fake_server = FakeAgentBridgeServer(self.socket_path)
        await self.fake_server.start()
        server_module.configure_client(self.socket_path, request_timeout=5)

    async def asyncTearDown(self) -> None:
        await server_module._get_client().close()
        await self.fake_server.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_vc3d_ping_tool_roundtrip(self) -> None:
        result = await server_module.vc3d_ping()
        self.assertTrue(result["pong"])
        self.assertEqual(result["pid"], 4242)

    async def test_vc3d_grow_segment_wait_returns_terminal_status(self) -> None:
        result = await server_module.vc3d_grow_segment(steps=1, wait=True)
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("consoleTail", result)
        self.assertEqual(result["jobId"], "job-1")

    async def test_vc3d_grow_segment_no_wait_returns_immediately(self) -> None:
        result = await server_module.vc3d_grow_segment(steps=1, wait=False)
        self.assertEqual(result["jobId"], "job-1")
        self.assertNotIn("state", result)

    async def test_vc3d_save_segment_idle_returns_immediately(self) -> None:
        self.fake_server.save_idle = True
        result = await server_module.vc3d_save_segment()
        self.assertIsNone(result["jobId"])
        self.assertEqual(result["state"], "idle")
        self.assertFalse(result["saveInProgress"])

    async def test_vc3d_save_segment_wait_returns_terminal_status(self) -> None:
        result = await server_module.vc3d_save_segment(wait=True)
        self.assertEqual(result["jobId"], "job-4")
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("consoleTail", result)

    async def test_vc3d_save_segment_no_wait_returns_job_id(self) -> None:
        result = await server_module.vc3d_save_segment(wait=False)
        self.assertEqual(result["jobId"], "job-4")
        self.assertEqual(result["state"], "running")

    async def test_vc3d_lasagna_start_optimization_no_wait_returns_immediately(self) -> None:
        result = await server_module.vc3d_lasagna_start_optimization(mode="reoptimize", wait=False)
        self.assertEqual(result["jobId"], "job-2")
        self.assertEqual(result["source"], "lasagna")
        self.assertEqual(result["kind"], "lasagna.optimize")
        self.assertNotIn("state", result)

    async def test_vc3d_lasagna_start_optimization_wait_returns_terminal_status(self) -> None:
        result = await server_module.vc3d_lasagna_start_optimization(mode="reoptimize", wait=True)
        self.assertEqual(result["jobId"], "job-2")
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["kind"], "lasagna.optimize")
        self.assertIn("consoleTail", result)

    async def test_vc3d_fetch_segment_already_materialized_is_synchronous(self) -> None:
        result = await server_module.vc3d_fetch_segment("seg-ready")
        self.assertTrue(result["fetched"])
        self.assertTrue(result["alreadyMaterialized"])
        self.assertNotIn("jobId", result)

    async def test_vc3d_fetch_segment_placeholder_waits_for_job(self) -> None:
        result = await server_module.vc3d_fetch_segment("seg-ph", wait=True)
        self.assertEqual(result["jobId"], "job-3")
        self.assertEqual(result["state"], "succeeded")

    async def test_vc3d_fetch_segment_placeholder_no_wait_returns_job_id(self) -> None:
        result = await server_module.vc3d_fetch_segment("seg-ph", wait=False)
        self.assertEqual(result["jobId"], "job-3")
        self.assertNotIn("state", result)

    async def test_vc3d_activate_segment_plain_success(self) -> None:
        result = await server_module.vc3d_activate_segment("seg-ready")
        self.assertTrue(result["activated"])
        self.assertNotIn("fetched", result)

    async def test_vc3d_activate_segment_auto_fetches_placeholder(self) -> None:
        # seg-ph refuses activation until fetched; auto_fetch (default) should
        # fetch then retry, yielding activated + fetched.
        result = await server_module.vc3d_activate_segment("seg-ph")
        self.assertTrue(result["activated"])
        self.assertTrue(result["fetched"])

    async def test_vc3d_activate_segment_no_auto_fetch_raises_placeholder(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            await server_module.vc3d_activate_segment("seg-ph", auto_fetch=False)
        self.assertEqual(ctx.exception.code, -32005)
        self.assertIn("placeholder", str(ctx.exception).lower())

    async def test_vc3d_rotate_viewer_relative_default_accumulates(self) -> None:
        first = await server_module.vc3d_rotate_viewer("seg yz", 30.0)
        self.assertEqual(first["plane"], "seg yz")
        self.assertEqual(first["previousDegrees"], 0.0)
        self.assertAlmostEqual(first["degrees"], 30.0)
        # relative is the default -- a second call adds to the first.
        second = await server_module.vc3d_rotate_viewer("seg yz", 15.0)
        self.assertAlmostEqual(second["previousDegrees"], 30.0)
        self.assertAlmostEqual(second["degrees"], 45.0)

    async def test_vc3d_rotate_viewer_absolute_sets_angle(self) -> None:
        await server_module.vc3d_rotate_viewer("xz", 90.0)
        result = await server_module.vc3d_rotate_viewer("xz", 10.0, relative=False)
        # shorthand "xz" normalizes to "seg xz"; absolute overrides accumulation.
        self.assertEqual(result["plane"], "seg xz")
        self.assertAlmostEqual(result["degrees"], 10.0)

    async def test_vc3d_rotate_viewer_unknown_plane_raises(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            await server_module.vc3d_rotate_viewer("seg xy", 10.0)
        self.assertEqual(ctx.exception.code, -32602)

    async def test_vc3d_set_axis_aligned_slices_toggles(self) -> None:
        on = await server_module.vc3d_set_axis_aligned_slices(True)
        self.assertEqual(on["enabled"], True)
        # idempotent: setting the same value again stays on.
        again = await server_module.vc3d_set_axis_aligned_slices(True)
        self.assertEqual(again["enabled"], True)
        off = await server_module.vc3d_set_axis_aligned_slices(False)
        self.assertEqual(off["enabled"], False)
    async def test_vc3d_set_wrap_annotation_mode_toggles(self) -> None:
        on = await server_module.vc3d_set_wrap_annotation_mode(True)
        self.assertTrue(on["enabled"])
        off = await server_module.vc3d_set_wrap_annotation_mode(False)
        self.assertFalse(off["enabled"])

    async def test_vc3d_commit_wrap_annotation_with_preview(self) -> None:
        await server_module.vc3d_set_wrap_annotation_mode(True)
        # Simulate a shift-click having seeded a preview on a chunked viewer.
        self.fake_server._wrap_has_preview = True
        result = await server_module.vc3d_commit_wrap_annotation()
        self.assertTrue(result["committed"])
        self.assertTrue(result["hadPreview"])
        # The commit consumed the preview: a second commit is a no-op.
        again = await server_module.vc3d_commit_wrap_annotation()
        self.assertFalse(again["committed"])

    async def test_vc3d_commit_wrap_annotation_mode_disabled_raises(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            await server_module.vc3d_commit_wrap_annotation()
        self.assertEqual(ctx.exception.code, -32002)

    async def test_vc3d_undo_wrap_annotation(self) -> None:
        await server_module.vc3d_set_wrap_annotation_mode(True)
        self.fake_server._wrap_has_preview = True
        result = await server_module.vc3d_undo_wrap_annotation()
        self.assertTrue(result["undone"])


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


class BridgeClientLifecycleTest(unittest.IsolatedAsyncioTestCase):
    """P1: connection/reader/waiter lifecycle (EOF reset, connect race, cancel
    leak, write-failure invalidation, idempotent close)."""

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
        self.assertEqual(await self.client.call("ping"), {"pong": True, "pid": 4242, "version": "test-0.0"})
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
        await self._wait_until(lambda: len(self.client._pending) == 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.client._pending, {})

    async def test_write_failure_resets_connection(self) -> None:
        await self.client.call("ping")  # establish a live connection

        class BadWriter:
            def is_closing(self):
                return False

            def write(self, data):
                raise OSError("simulated write failure")

            async def drain(self):
                return None

            def close(self):
                return None

        self.client._writer = BadWriter()
        with self.assertRaises(BridgeConnectionError):
            await self.client.call("ping")
        self.assertFalse(self.client.connected)
        self.assertIsNone(self.client._writer)

    async def test_close_is_idempotent(self) -> None:
        await self.client.call("ping")
        await self.client.close()
        await self.client.close()  # must not raise
        self.assertFalse(self.client.connected)
        with self.assertRaises(BridgeConnectionError):
            await self.client.ensure_connected()

    async def test_cancel_while_queued_for_write_lock_leaves_pending_empty(self) -> None:
        # Finding 4: a call cancelled while queued for _write_lock (before the
        # write) must not leak its _pending entry.
        await self.client.call("ping")  # establish a live connection
        await self.client._write_lock.acquire()
        try:
            task = asyncio.create_task(self.client.call("ping"))
            # It registers in _pending, then blocks awaiting the held write lock.
            await self._wait_until(lambda: len(self.client._pending) == 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            self.client._write_lock.release()
        self.assertEqual(self.client._pending, {})

    async def test_transport_reset_while_queued_for_write_lock_fails_cleanly(self) -> None:
        # Finding 5: if the transport is reset (and a reconnect bumps the
        # generation) while a call is queued for the write lock, the call must
        # raise BridgeConnectionError -- not AttributeError on a None writer, and
        # must NOT write the stale request onto the newer connection.
        await self.client.call("ping")

        class RecordingWriter:
            def __init__(self) -> None:
                self.writes: list[bytes] = []

            def is_closing(self) -> bool:
                return False

            def write(self, data) -> None:
                self.writes.append(data)

            async def drain(self) -> None:
                return None

            def close(self) -> None:
                return None

        await self.client._write_lock.acquire()
        new_writer = RecordingWriter()
        try:
            task = asyncio.create_task(self.client.call("ping"))
            await self._wait_until(lambda: len(self.client._pending) == 1)
            # Simulate an EOF drop that failed our pending future, followed by a
            # reconnect that installed a NEW writer and bumped the generation.
            self.client._on_disconnect(BridgeConnectionError("peer dropped"))
            self.client._writer = new_writer
            self.client._generation += 1
        finally:
            self.client._write_lock.release()
        with self.assertRaises(BridgeConnectionError):
            await task
        # The stale request must never have reached the new connection.
        self.assertEqual(new_writer.writes, [])

    async def test_connect_after_close_does_not_resurrect(self) -> None:
        # Finding 6: connect() must re-check _closed after acquiring the
        # transport and bail rather than resurrecting a closed client.
        await self.client.close()
        with self.assertRaises(BridgeConnectionError):
            await self.client.connect()
        self.assertTrue(self.client._closed)
        self.assertFalse(self.client.connected)

    async def test_close_racing_pending_connect_stays_closed(self) -> None:
        # Finding 6: close() concurrent with an in-flight connect() must not
        # leave the client connected. Delay open_unix_connection so close() runs
        # while connect() is pending.
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
            conn_task = asyncio.create_task(fresh.ensure_connected())
            await asyncio.wait_for(started.wait(), timeout=2.0)
            close_task = asyncio.create_task(fresh.close())
            await asyncio.sleep(0.02)  # let close() set _closed + block on the lock
            release.set()
            with self.assertRaises(BridgeConnectionError):
                await asyncio.wait_for(conn_task, timeout=2.0)
            await asyncio.wait_for(close_task, timeout=2.0)
        self.assertTrue(fresh._closed)
        self.assertFalse(fresh.connected)


class WaitAndProgressTest(unittest.IsolatedAsyncioTestCase):
    """P3/P4: run_trace + atlas wait semantics, and best-effort progress
    forwarding through _wait_for_job (buffered/new output, race-free terminal,
    disconnect wake, tolerant of an unavailable/failing progress sink)."""

    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-mcp-wait-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-agent-bridge.sock")
        self.fake_server = FakeAgentBridgeServer(self.socket_path)
        await self.fake_server.start()
        self.client = server_module.configure_client(self.socket_path, request_timeout=5)

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.fake_server.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # -- P3: the two newly wait-capable tools --

    async def test_run_trace_no_wait_returns_job_id(self) -> None:
        result = await server_module.vc3d_run_trace("seg-1", wait=False)
        self.assertEqual(result["jobId"], "job-5")
        self.assertNotIn("state", result)

    async def test_run_trace_wait_returns_terminal_status(self) -> None:
        result = await server_module.vc3d_run_trace("seg-1", wait=True)
        self.assertEqual(result["jobId"], "job-5")
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("consoleTail", result)

    async def test_atlas_search_no_wait_returns_job_id(self) -> None:
        result = await server_module.vc3d_atlas_search_start(wait=False)
        self.assertEqual(result["jobId"], "job-6")
        self.assertNotIn("state", result)

    async def test_atlas_search_wait_returns_terminal_status(self) -> None:
        result = await server_module.vc3d_atlas_search_start(wait=True)
        self.assertEqual(result["jobId"], "job-6")
        self.assertEqual(result["state"], "succeeded")

    # -- P4: progress forwarding (driven at the tracker level for determinism) --

    async def test_buffered_output_before_wait_is_forwarded(self) -> None:
        jobs = self.client.jobs
        jobs.on_progress({"jobId": "jb", "phase": "started"})
        jobs.on_progress({"jobId": "jb", "phase": "output", "message": "buffered-1\nbuffered-2"})
        jobs.on_progress({"jobId": "jb", "phase": "finished", "success": True})
        ctx = _FakeCtx()
        result = await server_module._wait_for_job("jb", True, {"jobId": "jb"}, ctx)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual([m for _, m in ctx.reports], ["buffered-1", "buffered-2"])
        # local sequence is monotonic starting at 1
        self.assertEqual([s for s, _ in ctx.reports], [1, 2])

    async def test_new_output_during_wait_is_forwarded_in_order(self) -> None:
        jobs = self.client.jobs
        jobs.on_progress({"jobId": "jd", "phase": "output", "message": "first"})
        ctx = _FakeCtx()
        task = asyncio.create_task(
            server_module._wait_for_job("jd", True, {"jobId": "jd"}, ctx)
        )
        await asyncio.sleep(0.02)  # let the waiter drain the buffered line and park
        jobs.on_progress({"jobId": "jd", "phase": "output", "message": "second"})
        jobs.on_progress({"jobId": "jd", "phase": "finished", "success": True})
        result = await task
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual([m for _, m in ctx.reports], ["first", "second"])

    async def test_terminal_before_wait_is_race_free(self) -> None:
        jobs = self.client.jobs
        # Everything, including finished, arrives before the wait registers.
        jobs.on_progress({"jobId": "jt", "phase": "output", "message": "done-line"})
        jobs.on_progress({"jobId": "jt", "phase": "finished", "success": True})
        ctx = _FakeCtx()
        result = await asyncio.wait_for(
            server_module._wait_for_job("jt", True, {"jobId": "jt"}, ctx), timeout=2.0
        )
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual([m for _, m in ctx.reports], ["done-line"])

    async def test_disconnect_wakes_waiter_promptly(self) -> None:
        jobs = self.client.jobs
        jobs.on_progress({"jobId": "jx", "phase": "started"})
        ctx = _FakeCtx()
        task = asyncio.create_task(
            server_module._wait_for_job("jx", True, {"jobId": "jx"}, ctx)
        )
        await asyncio.sleep(0.02)
        jobs.fail_all(BridgeConnectionError("peer dropped"))
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(task, timeout=2.0)

    async def test_wait_true_fails_promptly_on_bridge_disconnect(self) -> None:
        self.fake_server.finish_jobs = False  # job never sends a finished notification
        task = asyncio.create_task(server_module.vc3d_run_trace("seg-1", wait=True))
        await asyncio.sleep(0.05)  # let the job start + the wait register
        await self.fake_server.stop()  # peer EOF -> waiters must fail promptly
        with self.assertRaises((RuntimeError, BridgeConnectionError)):
            await asyncio.wait_for(task, timeout=2.0)

    async def test_noop_ctx_does_not_fail_the_tool(self) -> None:
        # ctx=None models an MCP client with no progress support.
        result = await server_module.vc3d_run_trace("seg-1", wait=True, ctx=None)
        self.assertEqual(result["state"], "succeeded")

    async def test_progress_report_failure_does_not_fail_the_tool(self) -> None:
        jobs = self.client.jobs
        jobs.on_progress({"jobId": "jf", "phase": "output", "message": "line"})
        jobs.on_progress({"jobId": "jf", "phase": "finished", "success": True})
        ctx = _FakeCtx(fail=RuntimeError("progress sink is down"))
        result = await server_module._wait_for_job("jf", True, {"jobId": "jf"}, ctx)
        self.assertEqual(result["state"], "succeeded")

    async def test_progress_report_cancellation_is_not_swallowed(self) -> None:
        jobs = self.client.jobs
        jobs.on_progress({"jobId": "jc", "phase": "output", "message": "line"})
        jobs.on_progress({"jobId": "jc", "phase": "finished", "success": True})
        ctx = _FakeCtx(fail=asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await server_module._wait_for_job("jc", True, {"jobId": "jc"}, ctx)

    async def test_disconnect_survives_racing_reset_in_wait_for_job(self) -> None:
        # Finding 7: a disconnect latched while a waiter is parked must be
        # observed even if a racing reconnect clears the GLOBAL error first.
        jobs = self.client.jobs
        jobs.on_progress({"jobId": "jr", "phase": "started"})
        ctx = _FakeCtx()
        task = asyncio.create_task(
            server_module._wait_for_job("jr", True, {"jobId": "jr"}, ctx)
        )
        await asyncio.sleep(0.02)  # let the waiter park
        jobs.fail_all(BridgeConnectionError("peer dropped"))
        jobs.reset_error()  # racing reconnect clears the global error immediately
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(task, timeout=2.0)

    async def test_disconnect_survives_racing_reset_in_wait_finished(self) -> None:
        # Finding 7: same latch guarantee for JobTracker.wait_finished.
        jobs = self.client.jobs
        jobs.on_progress({"jobId": "jw", "phase": "started"})
        task = asyncio.create_task(jobs.wait_finished("jw", timeout=2.0))
        await asyncio.sleep(0.02)
        jobs.fail_all(BridgeConnectionError("peer dropped"))
        jobs.reset_error()
        with self.assertRaises(BridgeConnectionError):
            await asyncio.wait_for(task, timeout=2.0)

    async def test_blocked_progress_sink_does_not_defeat_wait_cap(self) -> None:
        # Finding 8: a progress sink that blocks forever must not hold the tool
        # past the wait cap. With the cap shortened, the call returns
        # waitTimedOut instead of hanging on the report.
        import vc3d_mcp.core as core

        jobs = self.client.jobs
        # Output present but NO finished notification: the job is still running.
        jobs.on_progress({"jobId": "jp", "phase": "output", "message": "l1"})
        blocker = asyncio.Event()  # never set -> report_progress hangs

        class HangingCtx:
            def __init__(self) -> None:
                self.calls = 0

            async def report_progress(self, progress, total=None, message=None) -> None:
                self.calls += 1
                await blocker.wait()

        ctx = HangingCtx()
        orig_cap = core.DEFAULT_WAIT_TIMEOUT_S
        core.DEFAULT_WAIT_TIMEOUT_S = 0.2
        try:
            result = await asyncio.wait_for(
                server_module._wait_for_job("jp", True, {"jobId": "jp"}, ctx), timeout=3.0
            )
        finally:
            core.DEFAULT_WAIT_TIMEOUT_S = orig_cap
        self.assertTrue(result.get("waitTimedOut"))
        self.assertGreaterEqual(ctx.calls, 1)  # it did attempt delivery

    async def test_trailing_output_with_finished_is_flushed(self) -> None:
        # ALSO: output that lands together with `finished` while an earlier
        # progress report is being awaited must still be forwarded, via the
        # final rescan, rather than dropped when the loop breaks on `finished`.
        jobs = self.client.jobs
        jobs.on_progress({"jobId": "je", "phase": "output", "message": "first"})

        injected = {"done": False}

        class InjectingCtx:
            def __init__(self) -> None:
                self.reports: list[tuple[int, str | None]] = []

            async def report_progress(self, progress, total=None, message=None) -> None:
                self.reports.append((progress, message))
                if not injected["done"]:
                    injected["done"] = True
                    # While the first line is "in flight", a trailing output line
                    # and the finished notification arrive together -- exactly the
                    # window the pre-fix loop would drop without a final rescan.
                    jobs.on_progress(
                        {"jobId": "je", "phase": "output", "message": "tail-line"}
                    )
                    jobs.on_progress({"jobId": "je", "phase": "finished", "success": True})

        ctx = InjectingCtx()
        result = await asyncio.wait_for(
            server_module._wait_for_job("je", True, {"jobId": "je"}, ctx), timeout=2.0
        )
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("tail-line", [m for _, m in ctx.reports])


class ToolSchemaTest(unittest.IsolatedAsyncioTestCase):
    """Official-SDK schema assertions over the whole registered tool surface."""

    # The historical surface was 75 tools; this branch adds fetch, rotate, the
    # axis-aligned-slice toggle, and the same-wrap annotation trio, so the real
    # current count is 79. Assert the real number so schema drift is caught.
    EXPECTED_TOOL_COUNT = 79

    async def test_tool_surface_and_schemas(self) -> None:
        tools = await server_module.mcp.list_tools()
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


class RegistryDiscoveryTest(unittest.TestCase):
    """discover_registry_socket: newest live entry wins, dead entries reaped.

    Mirrors AgentBridgeServer's registry-file convention (~/.vc3d/agent_bridge/
    <pid>.json with {pid, name, path, startedAt}) and the stale-PID cleanup of
    LasagnaServiceManager::discoverServices()."""

    def setUp(self) -> None:
        self.registry_dir = tempfile.mkdtemp(prefix="vc3d-registry-test-")

    def tearDown(self) -> None:
        shutil.rmtree(self.registry_dir, ignore_errors=True)

    def _write_entry(self, pid: int, path: str, started_at: float, name: str = "vc3d-agent") -> str:
        file_path = os.path.join(self.registry_dir, f"{pid}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                {"pid": pid, "name": f"{name}-{pid}", "path": path, "startedAt": started_at},
                f,
            )
        return file_path

    def test_missing_directory_returns_none(self) -> None:
        self.assertIsNone(
            discover_registry_socket(os.path.join(self.registry_dir, "does-not-exist"))
        )

    def test_empty_directory_returns_none(self) -> None:
        self.assertIsNone(discover_registry_socket(self.registry_dir))

    def test_dead_pid_entry_is_filtered_and_removed(self) -> None:
        # A PID that is extremely unlikely to be alive. os.kill(pid, 0) must
        # raise ProcessLookupError, so the entry is treated as stale.
        dead_pid = 2_000_000_000
        dead_file = self._write_entry(dead_pid, "/tmp/dead-bridge.sock", started_at=1000.0)

        result = discover_registry_socket(self.registry_dir)

        self.assertIsNone(result)
        self.assertFalse(
            os.path.exists(dead_file), "stale dead-PID registry file should be removed"
        )

    def test_live_entry_is_found(self) -> None:
        # os.getpid() is by definition alive right now.
        live_path = "/tmp/live-bridge.sock"
        live_file = self._write_entry(os.getpid(), live_path, started_at=1234.0)

        result = discover_registry_socket(self.registry_dir)

        self.assertEqual(result, live_path)
        self.assertTrue(os.path.exists(live_file), "live registry file must be kept")

    def test_newest_live_entry_wins_and_dead_reaped(self) -> None:
        # Two live entries (both this process's pid is only one file, so use the
        # real pid for the newest and the parent pid for an older-but-live one).
        older_path = "/tmp/older-bridge.sock"
        newer_path = "/tmp/newer-bridge.sock"
        dead_file = self._write_entry(2_000_000_001, "/tmp/dead2.sock", started_at=9999.0)
        # Newest startedAt should win even though the dead entry has a larger one.
        self._write_entry(os.getppid(), older_path, started_at=100.0)
        self._write_entry(os.getpid(), newer_path, started_at=500.0)

        result = discover_registry_socket(self.registry_dir)

        self.assertEqual(result, newer_path)
        self.assertFalse(os.path.exists(dead_file), "dead entry must be reaped")

    def test_malformed_file_is_reaped(self) -> None:
        bad_file = os.path.join(self.registry_dir, "99999.json")
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write("{ not valid json")

        self.assertIsNone(discover_registry_socket(self.registry_dir))
        self.assertFalse(os.path.exists(bad_file), "malformed registry file should be removed")


class TeardownReapTest(unittest.TestCase):
    """Finding 13: _terminate_launched_process must always reap the child, even
    when the post-kill() wait() times out."""

    def tearDown(self) -> None:
        server_module._launched_process = None
        server_module._drain_thread = None
        server_module._log_thread = None

    def test_reap_process_waits(self) -> None:
        proc = subprocess.Popen(["true"])  # exits immediately
        server_module._reap_process(proc)
        self.assertIsNotNone(proc.returncode)

    def test_kill_timeout_schedules_background_reaper(self) -> None:
        # A child that survives terminate() and the timed wait()s after kill():
        # _terminate must spawn a background waiter that performs a final
        # unconditional (blocking) wait() so it can't linger as a zombie.
        class StubbornProc:
            def __init__(self) -> None:
                self.terminated = False
                self.killed = False
                self._reaped = threading.Event()

            def poll(self):
                return 0 if self._reaped.is_set() else None

            def terminate(self) -> None:
                self.terminated = True

            def kill(self) -> None:
                self.killed = True

            def wait(self, timeout=None):
                if timeout is not None:
                    # terminate()'s and kill()'s timed waits both "time out".
                    raise subprocess.TimeoutExpired(cmd="stubborn", timeout=timeout)
                # The background reaper's unconditional blocking wait reaps it.
                self._reaped.set()
                return 0

        proc = StubbornProc()
        server_module._launched_process = proc
        server_module._terminate_launched_process()
        self.assertTrue(proc.terminated)
        self.assertTrue(proc.killed)
        # Background reaper must eventually perform the blocking wait().
        self.assertTrue(proc._reaped.wait(timeout=3.0), "child was never reaped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
