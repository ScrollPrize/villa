#!/usr/bin/env python3
"""Self-test for the viewer round-2 tools (overlay + intersects), run without
a real VC3D instance.

Purpose-built fake bridge (model: FakeAgentBridgeServer in test_mcp_bridge.py)
stands in for AgentBridgeServer just long enough to confirm each new/extended
vc3d_mcp.tools.viewer function sends the right method + camelCase params,
strips None args, and passes the fake's response straight through.

Run directly:
    cd tools/vc3d-mcp && python3 test_viewer2.py -v
or via unittest discovery:
    python3 -m unittest test_viewer2 -v
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

from vc3d_mcp import core  # noqa: E402
from vc3d_mcp.bridge_client import BridgeError  # noqa: E402
from vc3d_mcp.tools.viewer import (  # noqa: E402
    vc3d_get_overlay,
    vc3d_get_render_settings,
    vc3d_list_overlay_volumes,
    vc3d_set_intersects,
    vc3d_set_overlay,
    vc3d_set_render_settings,
)


# Mirrors AgentBridgeHandlers_viewer.cpp's isKnownOverlayColormap() -- the
# same ids vc::specs() (core/src/render/Colormaps.cpp) exposes, plus "" for
# "no explicit choice" -- so the fake can reject an unknown id like the real
# bridge does instead of always echoing back whatever it was sent.
_KNOWN_COLORMAPS = {
    "", "fire", "viridis", "magma", "red", "green", "blue", "cyan", "magenta", "glasbey_black0",
}


class FakeAgentBridgeServer:
    """Minimal AF_UNIX, newline-delimited JSON-RPC 2.0 stand-in that echoes
    back whatever params it received (merged over a small canned state), so
    each test can assert on the request it produced and the response passed
    through."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._server: asyncio.base_events.Server | None = None
        self._writers: list[asyncio.StreamWriter] = []
        self.received_requests: list[dict] = []
        self._render_settings: dict = {
            "intersectionOpacity": 0.5,
            "volumeWindow": {"low": 0.0, "high": 255.0},
            "normalArrowLengthScale": 1.0,
            "normalMaxArrows": 32,
            "segmentationCursorMirroring": False,
            "samplingStride": 1,
            "zScrollSensitivity": 1.0,
            "highlightedSurfaceIds": [],
        }
        self._overlay: dict = {
            "volumeId": "",
            "colormap": "",
            "opacity": 0.5,
            "threshold": 0.0,
            "windowLow": 0.0,
            "windowHigh": 255.0,
            "maxDisplayedResolution": 0,
            "composite": {"enabled": False, "method": "max", "layersFront": 8, "layersBehind": 0},
        }

    async def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path)

    async def stop(self) -> None:
        for w in list(self._writers):
            w.close()
        if self._server is not None:
            self._server.close()
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
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
                if line:
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

        if method == "viewer.get_render_settings":
            result = dict(self._render_settings)
        elif method == "viewer.set_render_settings":
            # Mirror handleViewerSetRenderSettings' clamping of the normal-arrow
            # fields to the GUI slider ranges (0.1-2.0 / 4-100), so a test can
            # confirm the MCP tool surfaces the clamped echo rather than the
            # raw out-of-range value it sent.
            if "normalArrowLengthScale" in params:
                params["normalArrowLengthScale"] = max(0.1, min(2.0, params["normalArrowLengthScale"]))
            if "normalMaxArrows" in params:
                params["normalMaxArrows"] = max(4, min(100, params["normalMaxArrows"]))
            self._render_settings.update(params)
            result = dict(self._render_settings)
        elif method == "viewer.get_overlay":
            result = dict(self._overlay)
        elif method == "viewer.set_overlay":
            if "colormap" in params and params["colormap"] not in _KNOWN_COLORMAPS:
                await self._send(writer, {"jsonrpc": "2.0", "id": req_id,
                                           "error": {"code": -32602,
                                                     "message": "colormap must be one of the known overlay colormap ids"}})
                return
            if params.get("clear") or params.get("volumeId") == "":
                self._overlay["volumeId"] = ""
            elif "volumeId" in params:
                self._overlay["volumeId"] = params["volumeId"]
            for key in ("colormap", "opacity", "threshold", "maxDisplayedResolution"):
                if key in params:
                    self._overlay[key] = params[key]
            if "window" in params:
                self._overlay["windowLow"] = params["window"]["low"]
                self._overlay["windowHigh"] = params["window"]["high"]
            if "composite" in params:
                self._overlay["composite"].update(params["composite"])
            result = dict(self._overlay)
        elif method == "viewer.list_overlay_volumes":
            result = {
                "volumes": [{"id": "0", "current": True}, {"id": "1", "current": False}],
                "overlayVolumeId": self._overlay["volumeId"],
            }
        elif method == "viewer.set_intersects":
            ids = sorted(set(params.get("surfaceIds", [])) | {"segmentation"})
            applied = [params["viewer"]] if params.get("viewer") else ["v1", "v2"]
            result = {"surfaceIds": ids, "appliedToViewers": applied}
        else:
            await self._send(writer, {"jsonrpc": "2.0", "id": req_id,
                                       "error": {"code": -32601, "message": "Method not found"}})
            return
        await self._send(writer, {"jsonrpc": "2.0", "id": req_id, "result": result})

    async def _send(self, writer: asyncio.StreamWriter, obj: dict) -> None:
        writer.write((json.dumps(obj) + "\n").encode("utf-8"))
        await writer.drain()


class ViewerRound2ToolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-mcp-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-agent-bridge.sock")
        self.fake_server = FakeAgentBridgeServer(self.socket_path)
        await self.fake_server.start()
        core.configure_client(self.socket_path, request_timeout=5)

    async def asyncTearDown(self) -> None:
        await core._get_client().close()
        await self.fake_server.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _last_params(self) -> dict:
        return self.fake_server.received_requests[-1]["params"]

    async def test_get_render_settings_roundtrip(self) -> None:
        result = await vc3d_get_render_settings()
        self.assertEqual(result["volumeWindow"], {"low": 0.0, "high": 255.0})
        self.assertEqual(result["normalMaxArrows"], 32)

    async def test_set_render_settings_part_a_keys_mapped_and_none_stripped(self) -> None:
        result = await vc3d_set_render_settings(
            volume_window={"low": 10.0, "high": 200.0},
            normal_arrow_length_scale=1.5,
            normal_max_arrows=50,
            segmentation_cursor_mirroring=True,
            sampling_stride=2,
            z_scroll_sensitivity=5.0,
        )
        self.assertEqual(
            self._last_params(),
            {
                "volumeWindow": {"low": 10.0, "high": 200.0},
                "normalArrowLengthScale": 1.5,
                "normalMaxArrows": 50,
                "segmentationCursorMirroring": True,
                "samplingStride": 2,
                "zScrollSensitivity": 5.0,
            },
        )
        self.assertEqual(result["samplingStride"], 2)
        self.assertTrue(result["segmentationCursorMirroring"])

    async def test_set_render_settings_clamps_normal_arrow_fields(self) -> None:
        # handleViewerSetRenderSettings clamps to the GUI slider ranges
        # (length scale 0.1-2.0, max arrows 4-100) rather than passing an
        # out-of-range value straight through to the renderer.
        result = await vc3d_set_render_settings(
            normal_arrow_length_scale=50.0,
            normal_max_arrows=-5,
        )
        self.assertEqual(result["normalArrowLengthScale"], 2.0)
        self.assertEqual(result["normalMaxArrows"], 4)

    async def test_get_overlay_roundtrip(self) -> None:
        result = await vc3d_get_overlay()
        self.assertEqual(result["colormap"], "")
        self.assertEqual(result["composite"]["method"], "max")

    async def test_set_overlay_sends_camel_case_and_strips_none(self) -> None:
        result = await vc3d_set_overlay(
            volume_id="1",
            colormap="viridis",
            opacity=0.8,
            threshold=10.0,
            window={"low": 5.0, "high": 250.0},
            max_displayed_resolution=2,
            composite={"enabled": True, "method": "mean"},
        )
        self.assertEqual(
            self._last_params(),
            {
                "volumeId": "1",
                "colormap": "viridis",
                "opacity": 0.8,
                "threshold": 10.0,
                "window": {"low": 5.0, "high": 250.0},
                "maxDisplayedResolution": 2,
                "composite": {"enabled": True, "method": "mean"},
            },
        )
        self.assertEqual(result["volumeId"], "1")
        self.assertEqual(result["windowLow"], 5.0)
        self.assertTrue(result["composite"]["enabled"])
        self.assertEqual(result["composite"]["method"], "mean")

    async def test_set_overlay_clear_only_sends_clear(self) -> None:
        await vc3d_set_overlay(clear=True)
        self.assertEqual(self._last_params(), {"clear": True})

    async def test_set_overlay_unknown_colormap_raises(self) -> None:
        # AgentBridgeHandlers_viewer.cpp validates colormap against vc::specs()
        # and rejects an unrecognized id with -32602 instead of silently
        # falling back at render time; confirm the MCP tool surfaces that.
        with self.assertRaises(BridgeError) as ctx:
            await vc3d_set_overlay(colormap="not-a-real-colormap")
        self.assertEqual(ctx.exception.code, -32602)

    async def test_set_overlay_empty_colormap_is_accepted(self) -> None:
        # Empty string is the "no explicit choice" sentinel, not an error.
        result = await vc3d_set_overlay(colormap="")
        self.assertEqual(result["colormap"], "")

    async def test_list_overlay_volumes_roundtrip(self) -> None:
        result = await vc3d_list_overlay_volumes()
        self.assertEqual(result["volumes"], [{"id": "0", "current": True}, {"id": "1", "current": False}])
        self.assertIn("overlayVolumeId", result)

    async def test_set_intersects_with_viewer_sends_camel_case(self) -> None:
        result = await vc3d_set_intersects(surface_ids=["seg xz", "seg yz"], viewer="v1")
        self.assertEqual(self._last_params(), {"viewer": "v1", "surfaceIds": ["seg xz", "seg yz"]})
        self.assertEqual(result["appliedToViewers"], ["v1"])
        self.assertIn("segmentation", result["surfaceIds"])

    async def test_set_intersects_without_viewer_strips_none(self) -> None:
        result = await vc3d_set_intersects(surface_ids=["seg xz"])
        self.assertEqual(self._last_params(), {"surfaceIds": ["seg xz"]})
        self.assertEqual(result["appliedToViewers"], ["v1", "v2"])


if __name__ == "__main__":
    unittest.main()
