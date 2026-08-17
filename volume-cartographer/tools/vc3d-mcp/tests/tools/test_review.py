#!/usr/bin/env python3
"""Self-test for the segments.review MCP tool.

Stands up a tiny purpose-built fake bridge (AF_UNIX, newline-delimited
JSON-RPC 2.0) that records every request and echoes a canned result, so each
assertion can check:

  * the correct bridge method name + params land on the wire
    (``received_requests[-1]``);
  * the false onlyLoaded default and optional filter reach the wire correctly;
  * a "filter" dict is forwarded to the wire verbatim;
  * the bridge's result is passed through unchanged.

Run with:
    python3 -m unittest tests.tools.test_review -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from tests.support import EchoBridgeServer
from vc3d_mcp import core
from vc3d_mcp.tools.review import vc3d_review_segments


class ReviewToolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-review-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-bridge.sock")
        self.fake = EchoBridgeServer(self.socket_path)
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

    async def test_no_args_sends_default_scope(self) -> None:
        await vc3d_review_segments()
        self._assert_wire("segments.review", {"onlyLoaded": False})

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
        self._assert_wire(
            "segments.review",
            {"onlyLoaded": False, "filter": filt},
        )

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
