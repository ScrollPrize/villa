#!/usr/bin/env python3
"""Self-test for vc3d-mcp, run without a real VC3D instance.

Stands up a trivial fake JSON-RPC-over-AF_UNIX-socket server (standing in for
apps/VC3D/agent_bridge/AgentBridgeServer, so the suite stays hermetic -- no Qt
app needed) and confirms:

  * BridgeClient can connect, send a request, and parse a response
    (including JSON-RPC error replies, per SPEC.md section 2.5);
  * wait: true streams sequenced job.progress updates until job.status is
    terminal, with bounded replay and polling as a delivery fallback;
  * the actual MCP tool functions wire all of the above together correctly
    (configure_client -> tool call -> bridge round trip).

Run directly:
    cd tools/vc3d-mcp && python3 test_mcp_bridge.py -v
or via unittest discovery:
    python3 -m unittest test_mcp_bridge -v
"""

from __future__ import annotations

import asyncio
import base64
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

from mcp.server.fastmcp import Image  # noqa: E402

from vc3d_mcp.bridge_client import (  # noqa: E402
    BridgeClient,
    BridgeClientConfig,
    BridgeConnectionError,
    BridgeError,
    discover_registry_socket,
)
from vc3d_mcp import core, server as server_module  # noqa: E402
from vc3d_mcp.tools.atlas import vc3d_atlas_search_start  # noqa: E402
from vc3d_mcp.tools.lasagna import vc3d_lasagna_start_optimization  # noqa: E402
from vc3d_mcp.tools.catalog_volume import vc3d_list_volumes  # noqa: E402
from vc3d_mcp.tools.misc import (  # noqa: E402
    vc3d_cancel_job,
    vc3d_ping,
    vc3d_screenshot,
    vc3d_wait_job,
)
from vc3d_mcp.tools.segmentation import (  # noqa: E402
    vc3d_activate_segment,
    vc3d_delete_segment,
    vc3d_fetch_segment,
    vc3d_grow_segment,
    vc3d_rename_segment,
    vc3d_run_trace,
    vc3d_save_segment,
)
from vc3d_mcp.tools.viewer import (  # noqa: E402
    vc3d_get_render_settings,
    vc3d_rotate_viewer,
    vc3d_set_axis_aligned_slices,
    vc3d_set_render_settings,
)
from vc3d_mcp.tools.wrap import (  # noqa: E402
    vc3d_commit_wrap_annotation,
    vc3d_set_wrap_annotation_mode,
    vc3d_undo_wrap_annotation,
)


# A minimal valid 1x1 opaque-red PNG (matches the on-the-wire "base64" the real
# screenshot.capture bridge method returns for an inline, no-filePath capture).
TINY_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08"
    b"\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00"
    b"\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)
TINY_PNG_B64 = base64.b64encode(TINY_PNG_BYTES).decode("ascii")


class FakeAgentBridgeServer:
    """Minimal stand-in for AgentBridgeServer: AF_UNIX, newline-delimited
    JSON-RPC 2.0, a couple of canned methods, and stateful jobs whose
    job.status progress history grows and then reaches a terminal state."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._server: asyncio.base_events.Server | None = None
        self._writers: list[asyncio.StreamWriter] = []
        self.received_requests: list[dict] = []
        # job_id -> kind, so job.status can echo the right kind for each flow.
        self._job_kinds: dict[str, str] = {}
        # job_id -> live record answered by job.status (state grows to terminal).
        self._jobs: dict[str, dict] = {}
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
        # When False, jobs stay "running" forever (never terminal), so a
        # wait: true call polls until the cap or the peer drops.
        self.finish_jobs: bool = True
        # When False, jobs change state without emitting progress. This models
        # notification loss and terminal jobs that never produced output.
        self.emit_job_progress: bool = True
        # Re-send the latest update immediately before job.status replies to
        # exercise subscription/snapshot ordering and sequence deduplication.
        self.rebroadcast_latest_on_status: bool = False
        # When True, screenshot.capture returns a null/absent base64 even for an
        # inline (no filePath) capture -- models the "bridge unexpectedly gave
        # us no image bytes" edge the tool must degrade gracefully on.
        self.screenshot_drop_base64: bool = False
        # Global viewer render settings (viewer.get_render_settings /
        # viewer.set_render_settings). set merges its params over this and echoes
        # the full merged object back, so a wrapper round-trip is meaningful.
        self._render_settings: dict = {
            "intersectionOpacity": 0.5,
            "intersectionThickness": 1.0,
            "overlayOpacity": 0.25,
            "intersectionMaxSurfaces": 8,
            "planeIntersectionLinesVisible": True,
            "showSurfaceNormals": False,
            "showDirectionHints": False,
            "surfaceOverlayEnabled": True,
            "highlightedSurfaceIds": [],
        }

    async def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()

        writers = list(self._writers)
        for w in writers:
            w.close()
        if writers:
            await asyncio.gather(
                *(w.wait_closed() for w in writers), return_exceptions=True
            )

        if self._server is not None:
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
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _handle_line(self, raw: bytes, writer: asyncio.StreamWriter) -> None:
        msg = json.loads(raw.decode("utf-8"))
        self.received_requests.append(msg)
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "ping":
            await self._reply(
                writer,
                req_id,
                result={
                    "pong": True,
                    "pid": 4242,
                    "version": "test-0.0",
                    "protocolVersion": 1,
                },
            )
        elif method == "state.get":
            await self._reply(writer, req_id, result={"vpkg": None, "volume": None})
        elif method == "segmentation.grow":
            job_id = "job-1"
            self._job_kinds[job_id] = "segmentation.grow"
            await self._reply(writer, req_id, result={"jobId": job_id, "kind": "segmentation.grow"})
            self._start_job(job_id)
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
                self._start_job(job_id)
        elif method == "tracer.run_trace":
            job_id = "job-5"
            self._job_kinds[job_id] = "tracer.run_trace"
            await self._reply(
                writer, req_id,
                result={"jobId": job_id, "kind": "tracer.run_trace",
                        "source": "tool", "outputDir": "/tmp/traces"},
            )
            self._start_job(job_id)
        elif method == "atlas.search_start":
            job_id = "job-6"
            self._job_kinds[job_id] = "atlas.fiber_search"
            await self._reply(
                writer, req_id,
                result={"jobId": job_id, "kind": "atlas.fiber_search", "source": "atlas"},
            )
            self._start_job(job_id)
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
            self._start_job(job_id)
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
                self._start_job(job_id)
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
        elif method == "screenshot.capture":
            # Mirror the real bridge contract: an inline capture (no filePath)
            # returns the PNG as base64; a to-disk capture (filePath set) writes
            # the file and returns a dict with base64 null plus the path.
            file_path = params.get("filePath")
            if file_path is not None:
                await self._reply(
                    writer, req_id,
                    result={"target": params.get("target"), "filePath": file_path,
                            "base64": None, "width": 1, "height": 1},
                )
            else:
                b64 = None if self.screenshot_drop_base64 else TINY_PNG_B64
                await self._reply(
                    writer, req_id,
                    result={"target": params.get("target"), "filePath": None,
                            "base64": b64, "width": 1, "height": 1},
                )
        elif method == "job.status":
            job_id = params.get("jobId", "job-1")
            rec = self._jobs.get(job_id)
            if rec is None:
                rec = {
                    "jobId": job_id,
                    "kind": self._job_kinds.get(job_id, "segmentation.grow"),
                    "state": "succeeded", "message": "finished",
                    "success": True, "consoleTail": [], "progressHistory": [],
                }
            if self.rebroadcast_latest_on_status and rec["progressHistory"]:
                await self._broadcast(rec["progressHistory"][-1])
            await self._reply(
                writer,
                req_id,
                result={
                    "jobId": rec["jobId"],
                    "kind": rec["kind"],
                    "label": "Grow",
                    "state": rec["state"],
                    "message": rec.get("message"),
                    "success": rec.get("success"),
                    "outputPath": rec.get("outputPath"),
                    "consoleTail": list(rec["consoleTail"]),
                    "progressHistory": list(rec["progressHistory"]),
                },
            )
        elif method == "job.cancel":
            job_id = params.get("jobId")
            source = params.get("source")
            if not job_id and not source:
                await self._reply(
                    writer, req_id,
                    error={"code": -32602, "message": "jobId or source is required",
                           "data": {"param": "jobId"}},
                )
            else:
                resolved = job_id or "job-for-" + str(source)
                await self._reply(
                    writer, req_id,
                    result={"cancelRequested": True, "jobId": resolved,
                            "source": source or "growth",
                            "kind": self._job_kinds.get(resolved, "segmentation.grow")},
                )
        elif method == "volume.list":
            await self._reply(
                writer, req_id,
                result={"volumeIds": ["vol-a", "vol-b"], "currentVolumeId": "vol-a"},
            )
        elif method == "segments.delete":
            seg = params.get("segmentId")
            if params.get("confirm") is not True:
                await self._reply(
                    writer, req_id,
                    error={"code": -32602, "message": "destructive; pass confirm=true",
                           "data": {"param": "confirm",
                                    "reason": "destructive; pass confirm=true"}},
                )
            elif seg in ("seg-ready", "seg-ph") or seg in self._fetched:
                self._fetched.discard(seg)
                await self._reply(writer, req_id, result={"deleted": [seg]})
            else:
                await self._reply(
                    writer, req_id,
                    error={"code": -32007, "message": "Segment not found",
                           "data": {"kind": "segment", "id": seg}},
                )
        elif method == "segments.rename":
            seg = params.get("segmentId")
            new_name = params.get("newName")
            await self._reply(
                writer, req_id, result={"oldId": seg, "newId": new_name},
            )
        elif method == "viewer.get_render_settings":
            await self._reply(writer, req_id, result=dict(self._render_settings))
        elif method == "viewer.set_render_settings":
            # Merge the provided subset over current state and echo the full set,
            # mirroring the real "reuse get logic" reply contract.
            self._render_settings.update(params)
            await self._reply(writer, req_id, result=dict(self._render_settings))
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

    def _start_job(self, job_id: str) -> None:
        """Create a running job record (synchronously, so job.status can be
        polled immediately) and spawn its simulated progression."""
        kind = self._job_kinds.get(job_id, "segmentation.grow")
        self._jobs[job_id] = {
            "jobId": job_id, "kind": kind, "state": "running",
            "message": "starting", "success": None,
            "outputPath": None, "consoleTail": [], "progressHistory": [],
            "nextSeq": 1,
        }
        asyncio.create_task(self._simulate_job(job_id))

    async def _emit_job_progress(self, rec: dict, **fields) -> None:
        update = {
            "jobId": rec["jobId"],
            "kind": rec["kind"],
            "seq": rec["nextSeq"],
            **fields,
        }
        rec["nextSeq"] += 1
        rec["progressHistory"].append(update)
        rec["progressHistory"] = rec["progressHistory"][-64:]
        await self._broadcast(update)

    async def _simulate_job(self, job_id: str) -> None:
        rec = self._jobs.get(job_id)
        if rec is None:
            return
        kind = rec["kind"]
        for text in ("line A", "line B"):
            await asyncio.sleep(0.02)
            rec["consoleTail"].append(text)
            if self.emit_job_progress:
                await self._emit_job_progress(
                    rec, phase="output", message=text
                )
        if not self.finish_jobs:
            return  # stays "running": exercises the wait cap / disconnect
        await asyncio.sleep(0.02)
        rec.update(state="succeeded", success=True, message="finished")
        if self.emit_job_progress:
            await self._emit_job_progress(
                rec, phase="finished", success=True, message="finished",
                outputPath=None, result=None,
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
        self.assertEqual(result["protocolVersion"], 1)

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

    async def test_progress_subscription_is_bounded(self) -> None:
        await self.client.call("ping")
        async with self.client.subscribe_job_progress("job-q") as queue:
            for seq in range(1, 71):
                message = {
                    "jsonrpc": "2.0",
                    "method": "job.progress",
                    "params": {
                        "jobId": "job-q",
                        "seq": seq,
                        "phase": "output",
                        "message": str(seq),
                    },
                }
                self.client._dispatch_line(
                    self.client._conn, json.dumps(message).encode("utf-8")
                )
            self.assertEqual(queue.qsize(), 64)
            self.assertEqual(queue.get_nowait()["seq"], 7)

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
        core.configure_client(self.socket_path, request_timeout=5)
        self._orig_poll = core.POLL_INTERVAL_S
        core.POLL_INTERVAL_S = 0.01

    async def asyncTearDown(self) -> None:
        core.POLL_INTERVAL_S = self._orig_poll
        await core._get_client().close()
        await self.fake_server.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_vc3d_ping_tool_roundtrip(self) -> None:
        result = await vc3d_ping()
        self.assertTrue(result["pong"])
        self.assertEqual(result["pid"], 4242)

    async def test_vc3d_screenshot_inline_returns_image(self) -> None:
        # No file_path -> the tool decodes the bridge's base64 PNG into a
        # FastMCP Image (so the model SEES the screenshot), not a base64 dict.
        result = await vc3d_screenshot(target="window")
        self.assertIsInstance(result, Image)
        self.assertEqual(result.data, TINY_PNG_BYTES)
        # It converts to a proper MCP image content object.
        content = result.to_image_content()
        self.assertEqual(content.type, "image")
        self.assertEqual(content.mimeType, "image/png")
        self.assertEqual(base64.b64decode(content.data), TINY_PNG_BYTES)
        self.assertEqual(
            self.fake_server.received_requests[-1]["params"]["maxDim"], 2048
        )

    async def test_vc3d_screenshot_to_file_returns_dict(self) -> None:
        # file_path given -> the PNG is written to disk; the tool returns the
        # raw dict (base64 null), never an Image.
        result = await vc3d_screenshot(target="window", file_path="/tmp/shot.png")
        self.assertNotIsInstance(result, Image)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["filePath"], "/tmp/shot.png")
        self.assertIsNone(result["base64"])
        self.assertNotIn("maxDim", self.fake_server.received_requests[-1]["params"])

    async def test_vc3d_screenshot_honors_explicit_max_dim(self) -> None:
        await vc3d_screenshot(target="window", max_dim=1024)
        self.assertEqual(
            self.fake_server.received_requests[-1]["params"]["maxDim"], 1024
        )

    async def test_vc3d_screenshot_missing_base64_falls_back_to_dict(self) -> None:
        # Bridge returns no image bytes for an inline capture -> fall back to
        # the raw dict rather than crashing on decode.
        self.fake_server.screenshot_drop_base64 = True
        result = await vc3d_screenshot(target="window")
        self.assertNotIsInstance(result, Image)
        self.assertIsInstance(result, dict)

    async def test_vc3d_grow_segment_wait_returns_terminal_status(self) -> None:
        result = await vc3d_grow_segment(steps=1, wait=True)
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("consoleTail", result)
        self.assertEqual(result["jobId"], "job-1")

    async def test_vc3d_grow_segment_no_wait_returns_immediately(self) -> None:
        result = await vc3d_grow_segment(steps=1, wait=False)
        self.assertEqual(result["jobId"], "job-1")
        self.assertNotIn("state", result)

    async def test_vc3d_save_segment_idle_returns_immediately(self) -> None:
        self.fake_server.save_idle = True
        result = await vc3d_save_segment()
        self.assertIsNone(result["jobId"])
        self.assertEqual(result["state"], "idle")
        self.assertFalse(result["saveInProgress"])

    async def test_vc3d_save_segment_wait_returns_terminal_status(self) -> None:
        result = await vc3d_save_segment(wait=True)
        self.assertEqual(result["jobId"], "job-4")
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("consoleTail", result)

    async def test_vc3d_save_segment_no_wait_returns_job_id(self) -> None:
        result = await vc3d_save_segment(wait=False)
        self.assertEqual(result["jobId"], "job-4")
        self.assertEqual(result["state"], "running")

    async def test_vc3d_lasagna_start_optimization_no_wait_returns_immediately(self) -> None:
        result = await vc3d_lasagna_start_optimization(mode="reoptimize", wait=False)
        self.assertEqual(result["jobId"], "job-2")
        self.assertEqual(result["source"], "lasagna")
        self.assertEqual(result["kind"], "lasagna.optimize")
        self.assertNotIn("state", result)

    async def test_vc3d_lasagna_start_optimization_wait_returns_terminal_status(self) -> None:
        result = await vc3d_lasagna_start_optimization(mode="reoptimize", wait=True)
        self.assertEqual(result["jobId"], "job-2")
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["kind"], "lasagna.optimize")
        self.assertIn("consoleTail", result)

    async def test_vc3d_fetch_segment_already_materialized_is_synchronous(self) -> None:
        result = await vc3d_fetch_segment("seg-ready")
        self.assertTrue(result["fetched"])
        self.assertTrue(result["alreadyMaterialized"])
        self.assertNotIn("jobId", result)

    async def test_vc3d_fetch_segment_placeholder_waits_for_job(self) -> None:
        result = await vc3d_fetch_segment("seg-ph", wait=True)
        self.assertEqual(result["jobId"], "job-3")
        self.assertEqual(result["state"], "succeeded")

    async def test_vc3d_fetch_segment_placeholder_no_wait_returns_job_id(self) -> None:
        result = await vc3d_fetch_segment("seg-ph", wait=False)
        self.assertEqual(result["jobId"], "job-3")
        self.assertNotIn("state", result)

    async def test_vc3d_activate_segment_plain_success(self) -> None:
        result = await vc3d_activate_segment("seg-ready")
        self.assertTrue(result["activated"])
        self.assertNotIn("fetched", result)

    async def test_vc3d_activate_segment_auto_fetches_placeholder(self) -> None:
        # seg-ph refuses activation until fetched; auto_fetch (default) should
        # fetch then retry, yielding activated + fetched.
        result = await vc3d_activate_segment("seg-ph")
        self.assertTrue(result["activated"])
        self.assertTrue(result["fetched"])

    async def test_vc3d_activate_segment_no_auto_fetch_raises_placeholder(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            await vc3d_activate_segment("seg-ph", auto_fetch=False)
        self.assertEqual(ctx.exception.code, -32005)
        self.assertIn("placeholder", str(ctx.exception).lower())

    async def test_vc3d_rotate_viewer_relative_default_accumulates(self) -> None:
        first = await vc3d_rotate_viewer("seg yz", 30.0)
        self.assertEqual(first["plane"], "seg yz")
        self.assertEqual(first["previousDegrees"], 0.0)
        self.assertAlmostEqual(first["degrees"], 30.0)
        # relative is the default -- a second call adds to the first.
        second = await vc3d_rotate_viewer("seg yz", 15.0)
        self.assertAlmostEqual(second["previousDegrees"], 30.0)
        self.assertAlmostEqual(second["degrees"], 45.0)

    async def test_vc3d_rotate_viewer_absolute_sets_angle(self) -> None:
        await vc3d_rotate_viewer("xz", 90.0)
        result = await vc3d_rotate_viewer("xz", 10.0, relative=False)
        # shorthand "xz" normalizes to "seg xz"; absolute overrides accumulation.
        self.assertEqual(result["plane"], "seg xz")
        self.assertAlmostEqual(result["degrees"], 10.0)

    async def test_vc3d_rotate_viewer_unknown_plane_raises(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            await vc3d_rotate_viewer("seg xy", 10.0)
        self.assertEqual(ctx.exception.code, -32602)

    async def test_vc3d_set_axis_aligned_slices_toggles(self) -> None:
        on = await vc3d_set_axis_aligned_slices(True)
        self.assertEqual(on["enabled"], True)
        # idempotent: setting the same value again stays on.
        again = await vc3d_set_axis_aligned_slices(True)
        self.assertEqual(again["enabled"], True)
        off = await vc3d_set_axis_aligned_slices(False)
        self.assertEqual(off["enabled"], False)
    async def test_vc3d_set_wrap_annotation_mode_toggles(self) -> None:
        on = await vc3d_set_wrap_annotation_mode(True)
        self.assertTrue(on["enabled"])
        off = await vc3d_set_wrap_annotation_mode(False)
        self.assertFalse(off["enabled"])

    async def test_vc3d_commit_wrap_annotation_with_preview(self) -> None:
        await vc3d_set_wrap_annotation_mode(True)
        # Simulate a shift-click having seeded a preview on a chunked viewer.
        self.fake_server._wrap_has_preview = True
        result = await vc3d_commit_wrap_annotation()
        self.assertTrue(result["committed"])
        self.assertTrue(result["hadPreview"])
        # The commit consumed the preview: a second commit is a no-op.
        again = await vc3d_commit_wrap_annotation()
        self.assertFalse(again["committed"])

    async def test_vc3d_commit_wrap_annotation_mode_disabled_raises(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            await vc3d_commit_wrap_annotation()
        self.assertEqual(ctx.exception.code, -32002)

    async def test_vc3d_undo_wrap_annotation(self) -> None:
        await vc3d_set_wrap_annotation_mode(True)
        self.fake_server._wrap_has_preview = True
        result = await vc3d_undo_wrap_annotation()
        self.assertTrue(result["undone"])

    async def test_vc3d_cancel_job_forwards_id_and_source(self) -> None:
        result = await vc3d_cancel_job(job_id="job-1", source="growth")
        self.assertTrue(result["cancelRequested"])
        self.assertEqual(result["jobId"], "job-1")
        # None args are stripped, but both were given here; the wrapper forwards
        # them under the camelCase wire keys.
        sent = self.fake_server.received_requests[-1]["params"]
        self.assertEqual(sent, {"jobId": "job-1", "source": "growth"})

    async def test_vc3d_cancel_job_strips_none_args(self) -> None:
        result = await vc3d_cancel_job(source="lasagna")
        self.assertTrue(result["cancelRequested"])
        sent = self.fake_server.received_requests[-1]["params"]
        self.assertEqual(sent, {"source": "lasagna"})  # jobId=None dropped

    async def test_vc3d_list_volumes_roundtrip(self) -> None:
        result = await vc3d_list_volumes()
        self.assertEqual(result["volumeIds"], ["vol-a", "vol-b"])
        self.assertEqual(result["currentVolumeId"], "vol-a")

    async def test_vc3d_delete_segment_confirm_true_forwarded(self) -> None:
        result = await vc3d_delete_segment("seg-ready", confirm=True)
        self.assertEqual(result["deleted"], ["seg-ready"])
        sent = self.fake_server.received_requests[-1]["params"]
        self.assertEqual(sent, {"segmentId": "seg-ready", "confirm": True})

    async def test_vc3d_delete_segment_default_confirm_false_is_forwarded_and_refused(self) -> None:
        # confirm defaults to False and IS forwarded (the bridge enforces the
        # guard); the fake refuses it with -32602.
        with self.assertRaises(BridgeError) as ctx:
            await vc3d_delete_segment("seg-ready")
        self.assertEqual(ctx.exception.code, -32602)
        sent = self.fake_server.received_requests[-1]["params"]
        self.assertEqual(sent, {"segmentId": "seg-ready", "confirm": False})

    async def test_vc3d_rename_segment_forwards_new_name(self) -> None:
        result = await vc3d_rename_segment("seg-ready", "seg_renamed")
        self.assertEqual(result["oldId"], "seg-ready")
        self.assertEqual(result["newId"], "seg_renamed")
        sent = self.fake_server.received_requests[-1]["params"]
        self.assertEqual(sent, {"segmentId": "seg-ready", "newName": "seg_renamed"})

    async def test_vc3d_get_render_settings_roundtrip(self) -> None:
        result = await vc3d_get_render_settings()
        self.assertEqual(result["intersectionOpacity"], 0.5)
        self.assertIn("highlightedSurfaceIds", result)

    async def test_vc3d_set_render_settings_maps_and_strips_none(self) -> None:
        result = await vc3d_set_render_settings(
            intersection_opacity=0.9,
            show_surface_normals=True,
            highlighted_surface_ids=["s1", "s2"],
        )
        # Only the three provided fields are sent, each under its camelCase key;
        # all the omitted (None) args are dropped.
        sent = self.fake_server.received_requests[-1]["params"]
        self.assertEqual(
            sent,
            {"intersectionOpacity": 0.9, "showSurfaceNormals": True,
             "highlightedSurfaceIds": ["s1", "s2"]},
        )
        # The fake echoes the merged full settings: changed fields updated,
        # untouched fields preserved.
        self.assertEqual(result["intersectionOpacity"], 0.9)
        self.assertTrue(result["showSurfaceNormals"])
        self.assertEqual(result["highlightedSurfaceIds"], ["s1", "s2"])
        self.assertEqual(result["overlayOpacity"], 0.25)  # untouched


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
    EXPECTED_TOOL_COUNT = 116

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


class AutoLaunchTest(unittest.TestCase):
    def test_volume_package_uses_its_own_cli_option(self) -> None:
        self.assertEqual(
            server_module._launch_command("/tmp/VC3D", "/tmp/demo.volpkg.json"),
            ["/tmp/VC3D", "--agent-bridge", "--volpkg", "/tmp/demo.volpkg.json"],
        )

    def test_path_binary_is_used_before_repo_builds(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(server_module.shutil, "which", return_value="/usr/bin/VC3D"),
            mock.patch.object(server_module, "default_vc3d_binary") as fallback,
        ):
            self.assertEqual(server_module.resolve_launch_binary(None), "/usr/bin/VC3D")
            fallback.assert_not_called()

    def test_invalid_explicit_binary_does_not_silently_fall_back(self) -> None:
        with (
            mock.patch.object(server_module, "_is_executable", return_value=False),
            mock.patch.object(server_module.shutil, "which") as path_lookup,
        ):
            self.assertIsNone(server_module.resolve_launch_binary("/missing/VC3D"))
            path_lookup.assert_not_called()

    def test_protocol_check_rejects_stale_bridge(self) -> None:
        with self.assertRaisesRegex(BridgeConnectionError, "expected 1, got None"):
            server_module._validate_protocol({"pong": True})


class NewTailLinesTest(unittest.TestCase):
    """_new_tail_lines: which consoleTail lines are new versus already seen."""

    def test_empty_prev_returns_all(self) -> None:
        self.assertEqual(core._new_tail_lines([], ["a", "b", "c"]), ["a", "b", "c"])

    def test_identical_lists_return_nothing(self) -> None:
        self.assertEqual(core._new_tail_lines(["a", "b"], ["a", "b"]), [])

    def test_rolling_window_slide(self) -> None:
        prev = [str(n) for n in range(2, 52)]   # 2..51
        cur = [str(n) for n in range(4, 54)]    # 4..53
        self.assertEqual(core._new_tail_lines(prev, cur), ["52", "53"])

    def test_disjoint_returns_all_of_cur(self) -> None:
        self.assertEqual(core._new_tail_lines(["a", "b"], ["x", "y"]), ["x", "y"])

    def test_cur_shorter_full_suffix_overlap(self) -> None:
        self.assertEqual(core._new_tail_lines(["a", "b", "c"], ["b", "c"]), [])


class TeardownReapTest(unittest.TestCase):
    """_terminate_launched_process must always reap the child, even when the
    post-kill() wait() times out."""

    def tearDown(self) -> None:
        server_module._launched_process = None

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
