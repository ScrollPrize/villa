"""Focused tests for the fiber MCP wrappers."""

from __future__ import annotations

import unittest
from unittest import mock

from vc3d_mcp.tools import fiber


class FiberToolsTest(unittest.IsolatedAsyncioTestCase):
    async def test_create_atlas_passes_init_shell_override(self) -> None:
        call = mock.AsyncMock(return_value={"atlasDir": "/tmp/atlas", "displayed": True})
        with mock.patch.object(fiber, "_call", call):
            result = await fiber.vc3d_fiber_create_atlas("17", "/tmp/init_shells")

        self.assertTrue(result["displayed"])
        call.assert_awaited_once_with(
            "fiber.create_atlas",
            {"fiberId": "17", "initShellDir": "/tmp/init_shells"},
        )

    async def test_create_atlas_omits_unspecified_override(self) -> None:
        call = mock.AsyncMock(return_value={"atlasDir": "/tmp/atlas", "displayed": True})
        with mock.patch.object(fiber, "_call", call):
            await fiber.vc3d_fiber_create_atlas("17")

        call.assert_awaited_once_with("fiber.create_atlas", {"fiberId": "17"})


if __name__ == "__main__":
    unittest.main()
