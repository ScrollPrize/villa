"""Wire tests for the segmentation settings and approval-mask tools.

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
    vc3d_set_approval_mask_mode,
    vc3d_set_segmentation_settings,
    vc3d_undo_approval_stroke,
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
            growth_method="patch_tracer",
            growth_directions=["up", "left"],
            z_range={"enabled": True, "min": 100, "max": 200},
            patch_tracer={"step": 12},
            direction_fields=[{"path": "/fields/a.zarr", "orientation": "vertical"}],
            approval_mask={"show": True, "brushRadius": 30.0, "opacity": 0.4},
        )
        self.assertEqual(result["echoedMethod"], "segmentation.set_settings")
        self.assertEqual(
            result["echoedParams"],
            {
                "drag": {"radius": 8.0},
                "pushPull": {"step": 2.0},
                "growthMethod": "patch_tracer",
                "growthDirections": ["up", "left"],
                "zRange": {"enabled": True, "min": 100, "max": 200},
                "patchTracer": {"step": 12},
                "directionFields": [
                    {"path": "/fields/a.zarr", "orientation": "vertical"}],
                "approvalMask": {"show": True, "brushRadius": 30.0, "opacity": 0.4},
            },
        )

    async def test_reads_and_undo_have_no_params(self) -> None:
        for tool, method in (
            (vc3d_get_segmentation_settings, "segmentation.get_settings"),
            (vc3d_undo_approval_stroke, "segmentation.approval_mask.undo"),
        ):
            result = await tool()
            self.assertEqual(result["echoedMethod"], method)
            self.assertEqual(result["echoedParams"], {})

    async def test_approval_mode_sends_show(self) -> None:
        result = await vc3d_set_approval_mask_mode("approve", show=True)
        self.assertEqual(result["echoedParams"], {"mode": "approve", "show": True})


if __name__ == "__main__":
    unittest.main()
