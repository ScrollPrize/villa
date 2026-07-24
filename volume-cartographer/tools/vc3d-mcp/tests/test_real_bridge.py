"""Opt-in round trip against a real VC3D binary supplied by CI."""

from __future__ import annotations

import asyncio
import os
import unittest

from vc3d_mcp import server
from vc3d_mcp.bridge_client import BridgeClient, BridgeClientConfig


@unittest.skipUnless(
    os.environ.get("VC3D_MCP_TEST_BINARY"),
    "set VC3D_MCP_TEST_BINARY for the real VC3D bridge check",
)
class RealVC3DBridgeTest(unittest.IsolatedAsyncioTestCase):
    async def test_protocol_and_descriptor_coverage(self) -> None:
        binary = os.environ["VC3D_MCP_TEST_BINARY"]
        endpoint = await asyncio.to_thread(
            server.launch_vc3d,
            binary,
            timeout=60.0,
        )
        client = BridgeClient(
            BridgeClientConfig(
                socket=endpoint,
                connect_timeout=10.0,
                request_timeout=30.0,
            )
        )
        try:
            ping = await client.call("ping")
            self.assertEqual(ping["protocolVersion"], server.BRIDGE_PROTOCOL_VERSION)

            description = await client.call("rpc.describe")
            self.assertTrue(description["coverage"]["complete"])
            self.assertGreater(description["coverage"]["described"], 0)
        finally:
            await client.close()
            await asyncio.to_thread(server._terminate_launched_process)
