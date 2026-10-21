from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from vc3d_mcp.tools.fiber import vc3d_fiber_export, vc3d_fiber_import


class FiberTransferTest(unittest.IsolatedAsyncioTestCase):
    async def test_transfer_only_passes_path(self):
        for tool, method in (
            (vc3d_fiber_export, "fiber.export"),
            (vc3d_fiber_import, "fiber.import"),
        ):
            with self.subTest(method=method):
                with patch("vc3d_mcp.tools.fiber._call", new_callable=AsyncMock) as call:
                    call.return_value = {"ok": True}
                    self.assertEqual(await tool("/data/fibers.json"), {"ok": True})
                    call.assert_awaited_once_with(method, {"path": "/data/fibers.json"})

    async def test_scale_is_not_an_api_parameter(self):
        for tool in (vc3d_fiber_export, vc3d_fiber_import):
            with self.assertRaises(TypeError):
                await tool("/data/fibers.json", scale=2.0)
