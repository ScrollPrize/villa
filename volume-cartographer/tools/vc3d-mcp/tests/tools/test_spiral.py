"""Wire-level tests for the thin Spiral and manifest-attachment wrappers."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from tests.support import EchoBridgeServer
from vc3d_mcp import core
from vc3d_mcp.tools.lasagna import vc3d_attach_lasagna_manifest
from vc3d_mcp.tools.spiral import (
    vc3d_spiral_connect,
    vc3d_spiral_load_checkpoint,
    vc3d_spiral_run,
    vc3d_spiral_upload_input,
)


class SpiralToolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-spiral-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-bridge.sock")
        self.fake = EchoBridgeServer(self.socket_path)
        await self.fake.start()
        core.configure_client(self.socket_path, request_timeout=5)

    async def asyncTearDown(self) -> None:
        await core._get_client().close()
        await self.fake.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def assert_wire(self, method: str, params: dict) -> None:
        request = self.fake.received_requests[-1]
        self.assertEqual(request["method"], method)
        self.assertEqual(request["params"], params)

    async def test_attach_manifest_forwards_role_and_selection(self) -> None:
        await vc3d_attach_lasagna_manifest(
            "https://example.test/data.lasagna.json",
            "fiber_inference",
            select=False,
        )
        self.assert_wire(
            "lasagna.attach_manifest",
            {
                "location": "https://example.test/data.lasagna.json",
                "role": "fiber_inference",
                "select": False,
            },
        )

    async def test_connect_uses_saved_profile_id(self) -> None:
        await vc3d_spiral_connect("lab-profile")
        self.assert_wire("spiral.connect", {"profileId": "lab-profile"})

    async def test_run_maps_names_and_omits_absent_objects(self) -> None:
        await vc3d_spiral_run(20, influence_config={"weight": 0.5})
        self.assert_wire(
            "spiral.run",
            {"iterations": 20, "influenceConfig": {"weight": 0.5}},
        )

    async def test_upload_maps_input_fields(self) -> None:
        await vc3d_spiral_upload_input(
            "pcl", "/tmp/input.json", "constraint-1", role="same"
        )
        self.assert_wire(
            "spiral.input_upload",
            {
                "kind": "pcl",
                "localPath": "/tmp/input.json",
                "inputId": "constraint-1",
                "role": "same",
            },
        )

    async def test_checkpoint_load_omits_unused_selector(self) -> None:
        await vc3d_spiral_load_checkpoint(
            host_path="/service/checkpoint.ckpt", allow_rebuild=True
        )
        self.assert_wire(
            "spiral.checkpoint_load",
            {"hostPath": "/service/checkpoint.ckpt", "allowRebuild": True},
        )


if __name__ == "__main__":
    unittest.main()
