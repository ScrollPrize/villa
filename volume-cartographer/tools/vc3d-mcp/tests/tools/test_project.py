"""Wire tests for the project attachment MCP tools.

Run:
    python3 -m unittest tests.tools.test_project -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from typing import Any

from tests.support import FakeBridgeServer
from vc3d_mcp import core
from vc3d_mcp.tools.project import (
    vc3d_attach_to_project,
    vc3d_detach_from_project,
    vc3d_save_project_as,
)


class ProjectBridge(FakeBridgeServer):
    async def handle_message(self, message: dict[str, Any], writer) -> None:
        method = message.get("method")
        params = message.get("params") or {}
        if method == "job.status":
            result = {"jobId": params.get("jobId"), "state": "succeeded",
                      "consoleTail": [], "result": {"attached": True}}
        elif method == "project.attach" and params.get("location") == "/data/open.afv":
            # The open fiber volume answers synchronously.
            result = {"jobId": None, "attached": False, "alreadyAttached": True,
                      "params": params}
        elif method == "project.attach" and params.get("kind") in (
                "lasagna_manifest", "afv"):
            result = {"jobId": "job-3", "source": "project", "params": params}
        else:
            result = {"params": params, "attached": True}
        await self.reply(writer, message.get("id"), result=result)


class ProjectToolsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-project-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-bridge.sock")
        self.bridge = ProjectBridge(self.socket_path)
        await self.bridge.start()
        core.configure_client(self.socket_path, request_timeout=5)
        self._orig_poll = core.POLL_INTERVAL_S
        core.POLL_INTERVAL_S = 0.01

    async def asyncTearDown(self) -> None:
        core.POLL_INTERVAL_S = self._orig_poll
        await core._get_client().close()
        await self.bridge.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_synchronous_attach_has_no_job(self) -> None:
        result = await vc3d_attach_to_project("umbilicus", "/data/umb.json", wait=True)
        self.assertTrue(result["attached"])
        self.assertEqual(self.bridge.params_for("project.attach"),
                         {"kind": "umbilicus", "location": "/data/umb.json"})
        self.assertFalse(any(m.get("method") == "job.status"
                             for m in self.bridge.received))

    async def test_lasagna_attach_waits_for_job(self) -> None:
        result = await vc3d_attach_to_project(
            "lasagna_manifest", "s3://bucket/x.lasagna.json",
            fiber_inference=True, wait=True)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(
            self.bridge.params_for("project.attach"),
            {"kind": "lasagna_manifest", "location": "s3://bucket/x.lasagna.json",
             "fiberInference": True})

    async def test_open_fiber_volume_answers_without_a_job(self) -> None:
        result = await vc3d_attach_to_project("afv", "/data/open.afv", wait=True)
        self.assertTrue(result["alreadyAttached"])
        self.assertFalse(any(m.get("method") == "job.status"
                             for m in self.bridge.received))

    async def test_detach_and_save_map_params(self) -> None:
        await vc3d_detach_from_project("umbilicus", confirm=True)
        self.assertEqual(self.bridge.params_for("project.detach"),
                         {"kind": "umbilicus", "confirm": True})
        await vc3d_detach_from_project("normal_grid", "/data/ng")
        self.assertEqual(self.bridge.params_for("project.detach"),
                         {"kind": "normal_grid", "location": "/data/ng",
                          "confirm": False})
        await vc3d_save_project_as("/tmp/copy", overwrite=True)
        self.assertEqual(self.bridge.params_for("project.save_as"),
                         {"path": "/tmp/copy", "overwrite": True})


if __name__ == "__main__":
    unittest.main()
