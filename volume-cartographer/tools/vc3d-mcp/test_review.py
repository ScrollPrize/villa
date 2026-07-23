#!/usr/bin/env python3
"""Self-test for the segments.review MCP tool (Workstream 3).

Stands up a tiny purpose-built fake bridge (AF_UNIX, newline-delimited
JSON-RPC 2.0) that records every request and echoes a canned result, so each
assertion can check:

  * the correct bridge method name + params land on the wire
    (``received_requests[-1]``);
  * optional/None args (onlyLoaded, filter) are stripped before send when
    omitted;
  * a "filter" dict is forwarded to the wire verbatim;
  * the bridge's result is passed through unchanged.

Run directly:
    cd tools/vc3d-mcp && python3 test_review.py -v
or via unittest discovery:
    python3 -m unittest test_review -v
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import unittest

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vc3d_mcp import core  # noqa: E402
from vc3d_mcp.tools.review import vc3d_review_segments  # noqa: E402


class FakeReviewBridge:
    """AF_UNIX JSON-RPC 2.0 stand-in: records requests, echoes a canned result
    ({"echoedMethod", "echoedParams"}) for every method so passthrough is
    observable."""

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
            try:
                await asyncio.wait_for(self._server.wait_closed(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    async def _handle_client(self, reader, writer) -> None:
        self._writers.append(writer)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                msg = json.loads(line.decode("utf-8"))
                self.received_requests.append(msg)
                reply = {
                    "jsonrpc": "2.0",
                    "id": msg.get("id"),
                    "result": {
                        "echoedMethod": msg.get("method"),
                        "echoedParams": msg.get("params") or {},
                    },
                }
                writer.write((json.dumps(reply) + "\n").encode("utf-8"))
                await writer.drain()
        finally:
            if writer in self._writers:
                self._writers.remove(writer)


class ReviewToolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-review-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-bridge.sock")
        self.fake = FakeReviewBridge(self.socket_path)
        await self.fake.start()
        core.configure_client(self.socket_path, request_timeout=5)

    async def asyncTearDown(self) -> None:
        await core._get_client().close()
        await self.fake.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _assert_wire(self, method: str, params: dict) -> None:
        req = self.fake.received_requests[-1]
        self.assertEqual(req["method"], method)
        self.assertEqual(req["params"], params)

    async def test_no_args_sends_empty_params(self) -> None:
        # Both onlyLoaded and filter are None by default -> stripped entirely.
        await vc3d_review_segments()
        self._assert_wire("segments.review", {})

    async def test_only_loaded_forwarded(self) -> None:
        await vc3d_review_segments(only_loaded=True)
        self._assert_wire("segments.review", {"onlyLoaded": True})

    async def test_only_loaded_false_is_not_none_so_forwarded(self) -> None:
        # False is a meaningful value (distinct from omitted); _strip_none must
        # only drop None, not falsy values.
        await vc3d_review_segments(only_loaded=False)
        self._assert_wire("segments.review", {"onlyLoaded": False})

    async def test_filter_dict_forwarded_verbatim(self) -> None:
        filt = {"unreviewed": True, "hideDefective": True}
        await vc3d_review_segments(filter=filt)
        self._assert_wire("segments.review", {"filter": filt})

    async def test_only_loaded_and_filter_both_forwarded(self) -> None:
        filt = {"approved": True}
        result = await vc3d_review_segments(only_loaded=True, filter=filt)
        self._assert_wire("segments.review", {"onlyLoaded": True, "filter": filt})
        # Response passthrough: the tool returns the bridge's result unchanged.
        self.assertEqual(
            result,
            {
                "echoedMethod": "segments.review",
                "echoedParams": {"onlyLoaded": True, "filter": filt},
            },
        )


if __name__ == "__main__":
    unittest.main()
