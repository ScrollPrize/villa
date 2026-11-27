#!/usr/bin/env python3
"""Self-test for the keyboard and wheel MCP tools.

An echo bridge records each request, so the tests check the bridge method,
the camelCase params on the wire, None stripping, and the hold timeout.

Run with:
    python3 -m unittest tests.tools.test_input -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

from tests.support import EchoBridgeServer
from vc3d_mcp import core
from vc3d_mcp.tools.viewer import vc3d_key, vc3d_wheel


class InputToolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-input-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-bridge.sock")
        self.fake = EchoBridgeServer(self.socket_path)
        await self.fake.start()
        core.configure_client(self.socket_path, request_timeout=5)

    async def asyncTearDown(self) -> None:
        await core._get_client().close()
        await self.fake.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _assert_wire(self, method: str, params: dict) -> None:
        req = self.fake.received_requests[-1]
        self.assertEqual(req["method"], method)
        self.assertEqual(req["params"], params)

    async def test_key_tap_defaults(self) -> None:
        await vc3d_key(key="Ctrl+Z")
        self._assert_wire("canvas.key", {"key": "Ctrl+Z", "action": "tap"})

    async def test_key_press_with_viewer_and_modifiers(self) -> None:
        await vc3d_key(key="S", viewer="v1", modifiers=["shift"], action="press")
        self._assert_wire(
            "canvas.key",
            {"key": "S", "viewer": "v1", "modifiers": ["shift"], "action": "press"},
        )

    async def test_key_hold_forwards_hold_ms_and_extends_timeout(self) -> None:
        with mock.patch("vc3d_mcp.core._call", wraps=core._call) as call:
            await vc3d_key(key="A", hold_ms=2500)
        method, params = call.call_args.args
        self.assertEqual(method, "canvas.key")
        self.assertEqual(params, {"key": "A", "action": "tap", "holdMs": 2500})
        self.assertGreaterEqual(call.call_args.kwargs["timeout"], 2.5 + 10.0)

    async def test_key_press_limit_keeps_default_timeout(self) -> None:
        with mock.patch("vc3d_mcp.core._call", wraps=core._call) as call:
            await vc3d_key(key="S", action="press", hold_ms=30000)
        self.assertEqual(
            call.call_args.args[1], {"key": "S", "action": "press", "holdMs": 30000}
        )
        self.assertIsNone(call.call_args.kwargs["timeout"])

    async def test_key_without_hold_uses_default_timeout(self) -> None:
        with mock.patch("vc3d_mcp.core._call", wraps=core._call) as call:
            await vc3d_key(key="Esc")
        self.assertIsNone(call.call_args.kwargs["timeout"])

    async def test_wheel_volume_point(self) -> None:
        position = {"x": 10.0, "y": 20.0, "z": 30.0}
        result = await vc3d_wheel(position=position, delta_y=120)
        self._assert_wire(
            "canvas.wheel",
            {"position": position, "space": "volume", "deltaY": 120},
        )
        self.assertEqual(result["echoedMethod"], "canvas.wheel")

    async def test_wheel_scene_with_modifiers(self) -> None:
        await vc3d_wheel(
            position={"x": 4.0, "y": 5.0},
            delta_y=-240,
            viewer="seg xz",
            space="scene",
            modifiers=["shift"],
        )
        self._assert_wire(
            "canvas.wheel",
            {
                "position": {"x": 4.0, "y": 5.0},
                "space": "scene",
                "deltaY": -240,
                "viewer": "seg xz",
                "modifiers": ["shift"],
            },
        )


if __name__ == "__main__":
    unittest.main()
