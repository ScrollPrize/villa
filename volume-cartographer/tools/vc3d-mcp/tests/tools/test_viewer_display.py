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
from vc3d_mcp.tools.viewer_display import vc3d_get_ink_overlay, vc3d_set_ink_overlay


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


if __name__ == "__main__":
    unittest.main()
