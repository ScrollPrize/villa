#!/usr/bin/env python3
"""Self-test for the vc3d_inspect_ui MCP tool.

An echo bridge records the request so the tests check the method name, the
camelCase params, and that defaults reach the wire.

Run with:
    python3 -m unittest tests.tools.test_ui -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from tests.support import EchoBridgeServer
from vc3d_mcp import core
from vc3d_mcp.tools.session import vc3d_inspect_ui


class InspectToolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-ui-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-bridge.sock")
        self.fake = EchoBridgeServer(self.socket_path)
        await self.fake.start()
        core.configure_client(self.socket_path, request_timeout=5)

    async def asyncTearDown(self) -> None:
        await core._get_client().close()
        await self.fake.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_defaults_inspect_the_window(self) -> None:
        result = await vc3d_inspect_ui()
        self.assertEqual(result["echoedMethod"], "ui.inspect")
        self.assertEqual(
            result["echoedParams"],
            {"root": "window", "depth": 8, "includeHidden": False},
        )

    async def test_dock_root_with_hidden_widgets(self) -> None:
        await vc3d_inspect_ui(root="dock:dockWidgetSegmentation", depth=3, include_hidden=True)
        self.assertEqual(
            self.fake.params_for("ui.inspect"),
            {"root": "dock:dockWidgetSegmentation", "depth": 3, "includeHidden": True},
        )


if __name__ == "__main__":
    unittest.main()
