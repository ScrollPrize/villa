#!/usr/bin/env python3
"""Core vc3d-mcp transport and tool tests, run without a real VC3D instance.

Stands up a trivial fake JSON-RPC-over-AF_UNIX-socket server (standing in for
apps/VC3D/agent_bridge/AgentBridgeServer, so the suite stays hermetic -- no Qt
app needed) and confirms:

  * BridgeClient can connect, send a request, and parse a response
    (including JSON-RPC error replies, per SPEC.md section 2.5);
  * wait: true streams sequenced job.progress updates until job.status is
    terminal, with bounded replay and polling as a delivery fallback;
  * the actual MCP tool functions wire all of the above together correctly
    (configure_client -> tool call -> bridge round trip).

Run the complete suite, including progress, runtime, contract, and domain tests:
    cd tools/vc3d-mcp && python3 -m unittest discover -v
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import tempfile
import unittest

from mcp.server.fastmcp import Image

from tests.support import (
    FakeAgentBridgeServer,
    TINY_PNG_BYTES,
)
from vc3d_mcp.bridge_client import (
    BridgeClient,
    BridgeClientConfig,
    BridgeError,
)
from vc3d_mcp import core
from vc3d_mcp.tools.lasagna import vc3d_lasagna_start_optimization
from vc3d_mcp.tools.catalog_volume import (
    vc3d_attach_volume,
    vc3d_create_project,
    vc3d_list_volumes,
)
from vc3d_mcp.tools.jobs import vc3d_cancel_job
from vc3d_mcp.tools.session import (
    vc3d_ping,
    vc3d_screenshot,
)
from vc3d_mcp.tools.segmentation import (
    vc3d_activate_segment,
    vc3d_delete_segment,
    vc3d_fetch_segment,
    vc3d_grow_segment,
    vc3d_rename_segment,
    vc3d_save_segment,
)
from vc3d_mcp.tools.viewer import (
    vc3d_get_render_settings,
    vc3d_rotate_viewer,
    vc3d_set_axis_aligned_slices,
    vc3d_set_render_settings,
)
from vc3d_mcp.tools.wrap import (
    vc3d_commit_wrap_annotation,
    vc3d_set_wrap_annotation_mode,
    vc3d_undo_wrap_annotation,
)


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

    async def test_vc3d_attach_volume_returns_job_and_forwards_tags(self) -> None:
        result = await vc3d_attach_volume(
            "/tmp/overlay.zarr",
            tags=["role:overlay"],
        )
        self.assertEqual(result["jobId"], "job-7")
        self.assertNotIn("state", result)
        self.assertEqual(
            self.fake_server.received_requests[-1]["params"],
            {
                "location": "/tmp/overlay.zarr",
                "tags": ["role:overlay"],
            },
        )

    async def test_vc3d_attach_volume_waits_for_terminal_result(self) -> None:
        result = await vc3d_attach_volume(
            "https://example.test/overlay.zarr",
            wait=True,
        )
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["result"]["volumeId"], "vol-c")
        self.assertEqual(
            result["result"]["location"],
            "https://example.test/overlay.zarr",
        )

    async def test_vc3d_create_project_forwards_defaults_and_returns_created_path(
        self,
    ) -> None:
        result = await vc3d_create_project("/tmp/new-project", "/tmp/volume")
        self.assertEqual(
            result,
            {
                "path": "/tmp/new-project.volpkg.json",
                "name": "new-project",
                "volume": "/tmp/volume",
            },
        )
        self.assertEqual(
            self.fake_server.received_requests[-1]["params"],
            {
                "path": "/tmp/new-project",
                "volume": "/tmp/volume",
                "overwrite": False,
            },
        )

    async def test_vc3d_create_project_forwards_optional_values(self) -> None:
        await vc3d_create_project(
            "/tmp/new-project.volpkg.json",
            "https://example.test/volume.zarr",
            name="Example",
            tags=["source:test"],
            overwrite=True,
        )
        self.assertEqual(
            self.fake_server.received_requests[-1]["params"],
            {
                "path": "/tmp/new-project.volpkg.json",
                "volume": "https://example.test/volume.zarr",
                "name": "Example",
                "tags": ["source:test"],
                "overwrite": True,
            },
        )

    async def test_vc3d_create_project_derives_name_case_insensitively(self) -> None:
        result = await vc3d_create_project(
            "/tmp/Upper.VOLPKG.JSON",
            "/tmp/volume",
        )
        self.assertEqual(result["name"], "Upper")

    async def test_vc3d_create_project_preserves_bridge_type_errors(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            await vc3d_create_project(123, "/tmp/volume")  # type: ignore[arg-type]
        self.assertEqual(ctx.exception.code, -32602)
        self.assertEqual(ctx.exception.data["param"], "path")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
