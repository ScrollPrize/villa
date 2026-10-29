"""Wire tests for the segmentation settings tools.

Run:
    python3 -m unittest tests.tools.test_editing_settings -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from tests.support import EchoBridgeServer
from vc3d_mcp import core
from vc3d_mcp.tools.editing_settings import (
    vc3d_get_segmentation_settings,
    vc3d_set_segmentation_settings,
)


class EditingSettingsToolsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-editing-settings-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-bridge.sock")
        self.bridge = EchoBridgeServer(self.socket_path)
        await self.bridge.start()
        core.configure_client(self.socket_path, request_timeout=5)

    async def asyncTearDown(self) -> None:
        await core._get_client().close()
        await self.bridge.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_set_settings_maps_nested_fields(self) -> None:
        result = await vc3d_set_segmentation_settings(
            drag={"radius": 8.0},
            push_pull={"step": 2.0},
            growth_method="corrections",
            growth_directions=["up", "left"],
            z_range={"enabled": True, "min": 100, "max": 200},
            direction_fields=[{"path": "/fields/a.zarr", "orientation": "vertical"}],
        )
        self.assertEqual(result["echoedMethod"], "segmentation.set_settings")
        self.assertEqual(
            result["echoedParams"],
            {
                "drag": {"radius": 8.0},
                "pushPull": {"step": 2.0},
                "growthMethod": "corrections",
                "growthDirections": ["up", "left"],
                "zRange": {"enabled": True, "min": 100, "max": 200},
                "directionFields": [
                    {"path": "/fields/a.zarr", "orientation": "vertical"}],
            },
        )

    async def test_get_settings_has_no_params(self) -> None:
        result = await vc3d_get_segmentation_settings()
        self.assertEqual(result["echoedMethod"], "segmentation.get_settings")
        self.assertEqual(result["echoedParams"], {})


if __name__ == "__main__":
    unittest.main()
