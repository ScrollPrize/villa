#!/usr/bin/env python3
"""Self-test for vc3d-mcp, run without a real VC3D instance.

Stands up a trivial fake JSON-RPC-over-AF_UNIX-socket server (standing in for
apps/VC3D/agent_bridge/AgentBridgeServer, which doesn't exist yet at the time
this was written) and confirms:

  * BridgeClient can connect, send a request, and parse a response
    (including JSON-RPC error replies, per SPEC.md section 2.5);
  * job.progress notifications (SPEC.md section 3.18) are picked up by the
    background reader and folded into JobTracker, including the console
    tail and the "wait until finished" convenience;
  * the actual MCP tool functions in vc3d_mcp.server wire all of the above
    together correctly (configure_client -> tool call -> bridge round trip).

Run directly:
    cd tools/vc3d-mcp && python3 test_mcp_bridge.py -v
or via unittest discovery:
    python3 -m unittest test_mcp_bridge -v
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vc3d_mcp.bridge_client import BridgeClient, BridgeClientConfig, BridgeError  # noqa: E402
from vc3d_mcp import server as server_module  # noqa: E402


class FakeAgentBridgeServer:
    """Minimal stand-in for AgentBridgeServer: AF_UNIX, newline-delimited
    JSON-RPC 2.0, a couple of canned methods, and the ability to simulate a
    segmentation.grow job's job.progress notification sequence."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self._server: asyncio.base_events.Server | None = None
        self._writers: list[asyncio.StreamWriter] = []
        self.received_requests: list[dict] = []

    async def start(self) -> None:
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server = await asyncio.start_unix_server(self._handle_client, path=self.socket_path)

    async def stop(self) -> None:
        for w in list(self._writers):
            w.close()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.append(writer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                await self._handle_line(line, writer)
        finally:
            if writer in self._writers:
                self._writers.remove(writer)

    async def _handle_line(self, raw: bytes, writer: asyncio.StreamWriter) -> None:
        msg = json.loads(raw.decode("utf-8"))
        self.received_requests.append(msg)
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "ping":
            await self._reply(writer, req_id, result={"pong": True, "pid": 4242, "version": "test-0.0"})
        elif method == "state.get":
            await self._reply(writer, req_id, result={"vpkg": None, "volume": None})
        elif method == "segmentation.grow":
            job_id = "job-1"
            await self._reply(writer, req_id, result={"jobId": job_id, "kind": "segmentation.grow"})
            asyncio.create_task(self._simulate_job(job_id))
        elif method == "job.status":
            await self._reply(
                writer,
                req_id,
                result={
                    "jobId": params.get("jobId", "job-1"),
                    "kind": "segmentation.grow",
                    "label": "Grow",
                    "state": "succeeded",
                    "message": "finished",
                    "outputPath": None,
                    "consoleTail": ["line A", "line B"],
                },
            )
        elif method == "will_error":
            await self._reply(
                writer,
                req_id,
                error={
                    "code": -32003,
                    "message": "INVALID_COORDINATES",
                    "data": {"point": {"x": 1.0, "y": 2.0, "z": 3.0}},
                },
            )
        else:
            await self._reply(writer, req_id, error={"code": -32601, "message": "Method not found"})

    async def _simulate_job(self, job_id: str) -> None:
        await asyncio.sleep(0.02)
        await self._broadcast(
            {"jobId": job_id, "kind": "segmentation.grow", "phase": "started", "message": "starting"}
        )
        await asyncio.sleep(0.02)
        await self._broadcast(
            {"jobId": job_id, "kind": "segmentation.grow", "phase": "output", "message": "line A\nline B"}
        )
        await asyncio.sleep(0.02)
        await self._broadcast(
            {
                "jobId": job_id,
                "kind": "segmentation.grow",
                "phase": "finished",
                "success": True,
                "outputPath": None,
            }
        )

    async def _reply(self, writer: asyncio.StreamWriter, req_id, *, result=None, error=None) -> None:
        msg: dict = {"jsonrpc": "2.0", "id": req_id}
        if error is not None:
            msg["error"] = error
        else:
            msg["result"] = result
        await self._send(writer, msg)

    async def _broadcast(self, params: dict) -> None:
        msg = {"jsonrpc": "2.0", "method": "job.progress", "params": params}
        for w in list(self._writers):
            await self._send(w, msg)

    async def _send(self, writer: asyncio.StreamWriter, obj: dict) -> None:
        writer.write((json.dumps(obj) + "\n").encode("utf-8"))
        await writer.drain()


class BridgeClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-mcp-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-agent-bridge.sock")
        self.fake_server = FakeAgentBridgeServer(self.socket_path)
        await self.fake_server.start()
        self.client = BridgeClient(BridgeClientConfig(socket=self.socket_path, request_timeout=5))

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.fake_server.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_connect_send_request_and_parse_response(self) -> None:
        result = await self.client.call("ping")
        self.assertEqual(result, {"pong": True, "pid": 4242, "version": "test-0.0"})

    async def test_jsonrpc_error_reply_becomes_bridge_error(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            await self.client.call("will_error")
        err = ctx.exception
        self.assertEqual(err.code, -32003)
        self.assertEqual(err.message, "INVALID_COORDINATES")
        self.assertEqual(err.data, {"point": {"x": 1.0, "y": 2.0, "z": 3.0}})
        # str(err) must be a JSON object carrying code/message/data, since
        # that's what ends up as the MCP tool-call error text.
        as_json = json.loads(str(err))
        self.assertEqual(as_json["code"], -32003)
        self.assertEqual(as_json["data"]["point"]["x"], 1.0)

    async def test_method_not_found(self) -> None:
        with self.assertRaises(BridgeError) as ctx:
            await self.client.call("nope.not_a_real_method")
        self.assertEqual(ctx.exception.code, -32601)

    async def test_job_progress_notifications_feed_tracker(self) -> None:
        result = await self.client.call(
            "segmentation.grow", {"method": "tracer", "direction": "all", "steps": 1}
        )
        job_id = result["jobId"]
        record = await self.client.jobs.wait_finished(job_id, timeout=5)
        self.assertIsNotNone(record)
        self.assertEqual(record.state, "succeeded")
        self.assertTrue(record.success)
        self.assertTrue(any("line A" in line for line in record.console_tail))

    async def test_resolve_socket_path_absolute_existing_path(self) -> None:
        self.assertEqual(BridgeClient.resolve_socket_path(self.socket_path), self.socket_path)

    def test_handshake_line_parsing(self) -> None:
        parsed = BridgeClient.socket_path_from_handshake(
            "VC3D-AGENT-BRIDGE: listening name=vc3d-agent-1234 path=/tmp/vc3d-agent-1234\n"
        )
        self.assertEqual(parsed, ("vc3d-agent-1234", "/tmp/vc3d-agent-1234"))
        self.assertIsNone(BridgeClient.socket_path_from_handshake("not a handshake line"))


class ToolLayerTest(unittest.IsolatedAsyncioTestCase):
    """Exercises the actual @mcp.tool() functions, not just BridgeClient."""

    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-mcp-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-agent-bridge.sock")
        self.fake_server = FakeAgentBridgeServer(self.socket_path)
        await self.fake_server.start()
        server_module.configure_client(self.socket_path, request_timeout=5)

    async def asyncTearDown(self) -> None:
        await server_module._get_client().close()
        await self.fake_server.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    async def test_vc3d_ping_tool_roundtrip(self) -> None:
        result = await server_module.vc3d_ping()
        self.assertTrue(result["pong"])
        self.assertEqual(result["pid"], 4242)

    async def test_vc3d_grow_segment_wait_returns_terminal_status(self) -> None:
        result = await server_module.vc3d_grow_segment(steps=1, wait=True)
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("consoleTail", result)
        self.assertEqual(result["jobId"], "job-1")

    async def test_vc3d_grow_segment_no_wait_returns_immediately(self) -> None:
        result = await server_module.vc3d_grow_segment(steps=1, wait=False)
        self.assertEqual(result["jobId"], "job-1")
        self.assertNotIn("state", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
