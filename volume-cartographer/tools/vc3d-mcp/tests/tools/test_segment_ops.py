"""Wire tests for the segment file-operation MCP tools.

Each tool is called against an in-process fake bridge that records the
request, so the tests pin the RPC method, the camelCase parameter mapping,
omitted optionals, and the job wait behavior.

Run:
    python3 -m unittest tests.tools.test_segment_ops -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from typing import Any

from tests.support import FakeBridgeServer
from vc3d_mcp import core
from vc3d_mcp.tools import segment_ops
from vc3d_mcp.tools.segment_ops import (
    vc3d_export_segment_obj,
    vc3d_merge_segments,
    vc3d_neighbor_copy_segment,
    vc3d_patch_segment,
    vc3d_rasterize_segments,
    vc3d_transform_segment,
)


class SegmentOpsBridge(FakeBridgeServer):
    """Echo every request; answer job.status with a finished job."""

    async def handle_message(self, message: dict[str, Any], writer) -> None:
        method = message.get("method")
        params = message.get("params") or {}
        if method == "job.status":
            result = {"jobId": params.get("jobId"), "state": "succeeded",
                      "outputPath": "/vpkg/paths/final", "consoleTail": [],
                      "result": None}
        else:
            result = {"method": method, "params": params,
                      "jobId": "job-7", "source": "tool"}
        await self.reply(writer, message.get("id"), result=result)


class SegmentOpsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-segmentops-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-bridge.sock")
        self.bridge = SegmentOpsBridge(self.socket_path)
        await self.bridge.start()
        core.configure_client(self.socket_path, request_timeout=5)
        self._orig_poll = core.POLL_INTERVAL_S
        core.POLL_INTERVAL_S = 0.01

    async def asyncTearDown(self) -> None:
        core.POLL_INTERVAL_S = self._orig_poll
        await core._get_client().close()
        await self.bridge.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_transform_maps_op_and_confirm(self) -> None:
        for op in ("flip_u", "flip_v"):
            with self.subTest(op=op):
                result = await vc3d_transform_segment("seg-1", op)
                self.assertEqual(result["method"], "segment.transform")
                self.assertEqual(
                    self.bridge.params_for("segment.transform"),
                    {"segmentId": "seg-1", "op": op, "confirm": False},
                )
        await vc3d_transform_segment("seg-1", "rotate90", confirm=True)
        self.assertEqual(
            self.bridge.params_for("segment.transform"),
            {"segmentId": "seg-1", "op": "rotate90", "confirm": True},
        )

    async def test_export_obj_maps_options(self) -> None:
        result = await vc3d_export_segment_obj(
            "seg-1", output_path="/out/seg.obj", normalize_uv=True, omp_threads=4)
        self.assertEqual(result["jobId"], "job-7")
        self.assertNotIn("state", result)
        self.assertEqual(
            self.bridge.params_for("segment.export_obj"),
            {"segmentId": "seg-1", "outputPath": "/out/seg.obj",
             "overwrite": False, "normalizeUv": True, "alignGrid": False,
             "ompThreads": 4},
        )

    async def test_export_obj_wait_returns_terminal_status(self) -> None:
        result = await vc3d_export_segment_obj("seg-1", overwrite=True, wait=True)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(result["outputPath"], "/vpkg/paths/final")
        self.assertEqual(
            self.bridge.params_for("segment.export_obj"),
            {"segmentId": "seg-1", "overwrite": True, "normalizeUv": False,
             "alignGrid": False},
        )

    async def test_rasterize_maps_options_and_omits_unset(self) -> None:
        result = await vc3d_rasterize_segments(["seg-1", "seg-2"], z_max=20)
        self.assertEqual(result["jobId"], "job-7")
        self.assertEqual(
            self.bridge.params_for("segment.rasterize"),
            {"segmentIds": ["seg-1", "seg-2"], "zMax": 20},
        )

    async def test_neighbor_copy_maps_passes(self) -> None:
        result = await vc3d_neighbor_copy_segment(
            "seg-1", "out", neighbor_max_distance=150, pass2_omp_threads=8,
            pass2_param_overrides={"step": 10})
        self.assertEqual(result["jobId"], "job-7")
        self.assertEqual(
            self.bridge.params_for("segment.neighbor_copy"),
            {"segmentId": "seg-1", "direction": "out",
             "neighborMaxDistance": 150, "pass2OmpThreads": 8,
             "pass2ParamOverrides": {"step": 10}},
        )

    async def test_merge_grid_keeps_holes(self) -> None:
        await vc3d_merge_segments(
            grid=[["a", None], ["b", "c"]], ref_segment_id="b", ransac_iters=500)
        self.assertEqual(
            self.bridge.params_for("segments.merge"),
            {"grid": [["a", None], ["b", "c"]], "refSegmentId": "b",
             "ransacIters": 500},
        )

    async def test_patch_waits_and_forwards_confirm(self) -> None:
        result = await vc3d_patch_segment(
            ["a", "b"], parent_id="a", confirm=True, border_cells=8, wait=True)
        self.assertEqual(result["state"], "succeeded")
        self.assertEqual(
            self.bridge.params_for("segments.patch"),
            {"segmentIds": ["a", "b"], "parentId": "a", "confirm": True,
             "borderCells": 8},
        )

    def test_synchronous_file_operations_outlast_default_timeout(self) -> None:
        self.assertGreater(segment_ops.SEGMENT_FILE_OP_TIMEOUT_S, 30.0)


if __name__ == "__main__":
    unittest.main()
