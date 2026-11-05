"""Argument handling of the store tools, and waits on failed imports."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import unittest
from typing import Any

from tests.support import FakeBridgeServer
from vc3d_mcp import core
from vc3d_mcp.tools.stores import vc3d_import_objects

PARTIAL_REPORT = {
    "store": "side",
    "ingested": 1,
    "edgesImported": 0,
    "problems": [],
    "uuids": ["11111111-1111-4111-8111-111111111111"],
    "existing": [],
    "cancelled": True,
    "fatalError": None,
}


class _ImportBridge(FakeBridgeServer):
    """objects.import starts a job that is already terminal: failed, with the
    partial report of a cancelled import."""

    async def handle_message(self, message: dict[str, Any], writer: asyncio.StreamWriter) -> None:
        method = message.get("method")
        if method == "objects.import":
            result = {"jobId": "job-7", "kind": "objects.import", "source": "import", "store": "side"}
        elif method == "job.status":
            result = {
                "jobId": "job-7",
                "kind": "objects.import",
                "source": "import",
                "state": "failed",
                "message": "Import cancelled after 1 object(s)",
                "progressHistory": [],
                "result": PARTIAL_REPORT,
            }
        else:
            result = {"echoedMethod": method, "echoedParams": message.get("params") or {}}
        await self.reply(writer, message.get("id"), result=result)


class StoreToolsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-stores-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-bridge.sock")
        self.fake = _ImportBridge(self.socket_path)
        await self.fake.start()
        core.configure_client(self.socket_path, request_timeout=5)

    async def asyncTearDown(self) -> None:
        await core._get_client().close()
        await self.fake.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_import_keeps_explicit_empty_tags_and_renames(self) -> None:
        await vc3d_import_objects(
            "surface", "/data/paths", tags=[], bind={"scale": 2.0},
            uuid_namespace="0b6c2f1e-36f6-4f43-9a52-6f2c7b1d9e01", store="side",
        )
        self.assertEqual(
            self.fake.params_for("objects.import"),
            {
                "kind": "surface",
                "path": "/data/paths",
                "tags": [],
                "bind": {"scale": 2.0},
                "uuidNamespace": "0b6c2f1e-36f6-4f43-9a52-6f2c7b1d9e01",
                "store": "side",
            },
        )
        await vc3d_import_objects("fiber", "/data/fibers")
        self.assertEqual(self.fake.params_for("objects.import"), {"kind": "fiber", "path": "/data/fibers"})

    async def test_wait_returns_failed_partial_report_promptly(self) -> None:
        result = await asyncio.wait_for(
            vc3d_import_objects("surface", "/data/paths", wait=True), timeout=3.0
        )
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["message"], "Import cancelled after 1 object(s)")
        self.assertEqual(result["result"], PARTIAL_REPORT)


if __name__ == "__main__":
    unittest.main()
