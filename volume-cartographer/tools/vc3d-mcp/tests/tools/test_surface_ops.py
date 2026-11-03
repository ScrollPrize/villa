"""Standalone tests for the surface_ops MCP tool module.

Exercises the actual @mcp.tool() wrappers in vc3d_mcp.tools.surface_ops against a
small in-process fake Agent Bridge (AF_UNIX, newline-delimited JSON-RPC 2.0),
asserting:
  * method + params land on the wire with None stripped and snake_case mapped to
    the RPC's camelCase keys;
  * the synchronous crop passes its result straight through;
  * async "tool" ops (reoptimize, refine) honor wait vs no-wait, mirroring the
    run_trace / grow idiom (wait=True polls job.status to a terminal state).

Run:
    python3 -m unittest tests.tools.test_surface_ops -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from tests.support import FakeBridgeServer
from vc3d_mcp import core
from vc3d_mcp.tools.surface_ops import (
    vc3d_crop_segment_bounds,
    vc3d_refine_segment_alpha_comp,
    vc3d_reoptimize_segment,
)


class SurfaceOpsBridge(FakeBridgeServer):
    """Minimal AgentBridgeServer stand-in for the surface_ops RPCs."""

    def __init__(self, socket_path: str):
        super().__init__(socket_path)
        # Job records answered by job.status; they start "running" and flip to
        # "succeeded" after the first poll so wait=True exercises the poll loop.
        self._jobs: dict[str, dict] = {}
        self._polls: dict[str, int] = {}

    async def handle_message(self, msg: dict, writer) -> None:
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "segment.crop_bounds":
            # The crop core reports failures through its error out-param.
            if params.get("segmentId") == "seg-bad":
                await self.reply(writer, req_id, error={
                    "code": -32005, "message": "Crop failed",
                    "data": {"detail": "Cannot crop surface: Missing coordinate grid"}})
            else:
                await self.reply(writer, req_id,
                                  result={"cropped": True, "segmentId": params.get("segmentId")})
        elif method == "segment.reoptimize":
            job_id = "job-ro"
            self._jobs[job_id] = {"jobId": job_id, "kind": "segment.reoptimize"}
            await self.reply(writer, req_id, result={
                "jobId": job_id, "kind": "segment.reoptimize", "source": "tool",
                "volumeId": "vol-a"})
        elif method == "segment.refine_alpha_comp":
            job_id = "job-ac"
            self._jobs[job_id] = {"jobId": job_id, "kind": "segment.refine_alpha_comp"}
            await self.reply(writer, req_id, result={
                "jobId": job_id, "kind": "segment.refine_alpha_comp", "source": "tool",
                "segmentId": params.get("segmentId")})
        elif method == "job.status":
            job_id = params.get("jobId")
            rec = self._jobs.get(job_id, {"jobId": job_id, "kind": "segment.reoptimize"})
            self._polls[job_id] = self._polls.get(job_id, 0) + 1
            state = "running" if self._polls[job_id] < 2 else "succeeded"
            await self.reply(writer, req_id, result={
                "jobId": job_id, "kind": rec["kind"], "label": "op",
                "state": state, "message": "finished" if state == "succeeded" else "working",
                "success": state == "succeeded", "outputPath": None,
                "consoleTail": ["line A"] if state == "running" else ["line A", "line B"]})
        else:
            await self.reply(
                writer,
                req_id,
                error={"code": -32601, "message": "Method not found"},
            )


class SurfaceOpsTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-surfaceops-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-bridge.sock")
        self.bridge = SurfaceOpsBridge(self.socket_path)
        await self.bridge.start()
        core.configure_client(self.socket_path, request_timeout=5)
        self._orig_poll = core.POLL_INTERVAL_S
        core.POLL_INTERVAL_S = 0.01

    async def asyncTearDown(self) -> None:
        core.POLL_INTERVAL_S = self._orig_poll
        await core._get_client().close()
        await self.bridge.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # Synchronous operations

    async def test_crop_bounds_passthrough(self) -> None:
        result = await vc3d_crop_segment_bounds("seg-1")
        self.assertTrue(result["cropped"])
        self.assertEqual(result["segmentId"], "seg-1")
        self.assertEqual(self.bridge.params_for("segment.crop_bounds"), {"segmentId": "seg-1"})

    # Async external-tool operations

    async def test_reoptimize_no_wait_returns_job(self) -> None:
        result = await vc3d_reoptimize_segment("seg-1", wait=False)
        self.assertEqual(result["jobId"], "job-ro")
        self.assertEqual(result["source"], "tool")
        self.assertNotIn("state", result)
        self.assertEqual(
            self.bridge.params_for("segment.reoptimize"),
            {"segmentId": "seg-1", "ompThreads": 1},
        )

    async def test_reoptimize_wait_returns_terminal_status(self) -> None:
        result = await vc3d_reoptimize_segment("seg-1", wait=True)
        self.assertEqual(result["jobId"], "job-ro")
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("consoleTail", result)
        # Initial-result fields survive the merge with the terminal status.
        self.assertEqual(result["source"], "tool")

    async def test_reoptimize_optionals_mapped_to_camelcase(self) -> None:
        await vc3d_reoptimize_segment(
            "seg-1", volume_id="vol-b", omp_threads=4,
            param_overrides={"resume_local_max_iters": 500}, wait=False)
        self.assertEqual(
            self.bridge.params_for("segment.reoptimize"),
            {"segmentId": "seg-1", "volumeId": "vol-b", "ompThreads": 4,
             "paramOverrides": {"resume_local_max_iters": 500}},
        )

    async def test_refine_alpha_comp_no_wait_returns_job(self) -> None:
        result = await vc3d_refine_segment_alpha_comp("seg-1", wait=False)
        self.assertEqual(result["jobId"], "job-ac")
        self.assertEqual(result["source"], "tool")
        self.assertNotIn("state", result)
        self.assertEqual(
            self.bridge.params_for("segment.refine_alpha_comp"),
            {
                "segmentId": "seg-1",
                "refine": True,
                "start": -6.0,
                "stop": 30.0,
                "step": 2.0,
                "low": 26,
                "high": 255,
                "borderOff": 1.0,
                "radius": 3,
                "genVertexColor": False,
                "readerScale": 0.5,
                "scaleGroup": "1",
                "inPlace": False,
            },
        )

    async def test_refine_alpha_comp_wait_returns_terminal_status(self) -> None:
        result = await vc3d_refine_segment_alpha_comp("seg-1", wait=True)
        self.assertEqual(result["jobId"], "job-ac")
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("consoleTail", result)

    async def test_refine_alpha_comp_optionals_mapped_to_camelcase(self) -> None:
        await vc3d_refine_segment_alpha_comp(
            "seg-1", refine=False, start=-3.0, stop=20.0, step=1.0, low=10, high=200,
            border_off=2.0, radius=5, gen_vertex_color=True,
            reader_scale=0.25, scale_group="2", omp_threads=8, output_name="refined",
            in_place=True,
            wait=False)
        self.assertEqual(
            self.bridge.params_for("segment.refine_alpha_comp"),
            {"segmentId": "seg-1", "refine": False, "start": -3.0, "stop": 20.0,
             "step": 1.0, "low": 10, "high": 200, "borderOff": 2.0, "radius": 5,
             "genVertexColor": True, "readerScale": 0.25,
             "scaleGroup": "2", "ompThreads": 8, "outputName": "refined", "inPlace": True},
        )

    async def test_crop_bounds_failure_propagates(self) -> None:
        """A crop core failure surfaces as a BridgeError (-32005), not a false
        cropped:true."""
        from vc3d_mcp.bridge_client import BridgeError

        with self.assertRaises(BridgeError) as cm:
            await vc3d_crop_segment_bounds("seg-bad")
        self.assertEqual(cm.exception.code, -32005)


if __name__ == "__main__":
    unittest.main()
