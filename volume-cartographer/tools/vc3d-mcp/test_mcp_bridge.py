#!/usr/bin/env python3
"""Self-test for vc3d-mcp, run without a real VC3D instance.

Stands up a trivial fake JSON-RPC-over-AF_UNIX-socket server (standing in for
apps/VC3D/agent_bridge/AgentBridgeServer, which doesn't exist yet at the time
this was written) and confirms:

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
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vc3d_mcp.bridge_client import (  # noqa: E402
    BridgeClient,
    BridgeClientConfig,
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

    async def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path)

    async def stop(self) -> None:
        for w in list(self._writers):
            w.close()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
