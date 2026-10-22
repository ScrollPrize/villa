"""Wire tests for the viewer display MCP tools (ink detection overlay).

Run:
    python3 -m unittest tests.tools.test_viewer_display -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from tests.support import EchoBridgeServer
from vc3d_mcp import core
from vc3d_mcp.tools.viewer import vc3d_set_overlay, vc3d_set_render_settings
from vc3d_mcp.tools.viewer_display import (
    vc3d_get_ink_overlay,
    vc3d_reset_viewer,
    vc3d_set_ink_overlay,
)


class ViewerDisplayToolsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-viewer-display-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-bridge.sock")
        self.bridge = EchoBridgeServer(self.socket_path)
        await self.bridge.start()
        core.configure_client(self.socket_path, request_timeout=5)

    async def asyncTearDown(self) -> None:
        await core._get_client().close()
        await self.bridge.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_get_ink_overlay_forwards_refresh(self) -> None:
        result = await vc3d_get_ink_overlay(refresh=True)
        self.assertEqual(result["echoedMethod"], "viewer.get_ink_overlay")
        self.assertEqual(result["echoedParams"], {"refresh": True})

    async def test_set_ink_overlay_sends_only_given_fields(self) -> None:
        result = await vc3d_set_ink_overlay(path="", opacity=0.5, flip_vertical=True)
        self.assertEqual(result["echoedMethod"], "viewer.set_ink_overlay")
        self.assertEqual(result["echoedParams"],
                         {"path": "", "opacity": 0.5, "flipVertical": True})


    async def test_reset_viewer_maps_z_offset(self) -> None:
        result = await vc3d_reset_viewer(fit="content", z_offset=-3.0)
        self.assertEqual(result["echoedMethod"], "viewer.reset")
        self.assertEqual(result["echoedParams"], {"fit": "content", "zOffset": -3.0})
        result = await vc3d_reset_viewer()
        self.assertEqual(result["echoedParams"], {"fit": "surface"})

    async def test_render_settings_view_and_composite_fields(self) -> None:
        result = await vc3d_set_render_settings(
            axis_overlays=False,
            raw_points=True,
            focus_bounds={"enabled": True, "min": {"x": 0, "y": 0, "z": 0},
                          "max": {"x": 9, "y": 9, "z": 9}},
            composite={"enabled": True, "method": "alpha",
                       "planes": {"xy": True}},
            volumetric={"lighting": 0.5},
        )
        self.assertEqual(
            result["echoedParams"],
            {
                "axisOverlays": False,
                "rawPoints": True,
                "focusBounds": {"enabled": True, "min": {"x": 0, "y": 0, "z": 0},
                                "max": {"x": 9, "y": 9, "z": 9}},
                "composite": {"enabled": True, "method": "alpha",
                              "planes": {"xy": True}},
                "volumetric": {"lighting": 0.5},
            },
        )

    async def test_overlay_interpolation(self) -> None:
        result = await vc3d_set_overlay(interpolation="trilinear")
        self.assertEqual(result["echoedParams"], {"interpolation": "trilinear"})


if __name__ == "__main__":
    unittest.main()
