#!/usr/bin/env python3
"""Self-test for the points.* editing MCP tools.

Stands up a tiny purpose-built fake bridge (AF_UNIX, newline-delimited
JSON-RPC 2.0) that records every request and echoes a canned result for any
points.* method. For each new tool we assert:

  * the correct bridge method name + params land on the wire
    (``received_requests[-1]``);
  * optional/None args are stripped before send;
  * the bridge's result is passed through unchanged.

Run with:
    python3 -m unittest tests.tools.test_points -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from tests.support import EchoBridgeServer
from vc3d_mcp import core
from vc3d_mcp.tools.points import (
    vc3d_add_point_collection,
    vc3d_apply_anchor_offset,
    vc3d_auto_fill_windings,
    vc3d_clear_all_points,
    vc3d_clear_point_collection,
    vc3d_load_points_json,
    vc3d_load_points_segment_path,
    vc3d_remove_point,
    vc3d_remove_point_collection_tag,
    vc3d_rename_point_collection,
    vc3d_reset_windings,
    vc3d_save_points_json,
    vc3d_save_points_segment_path,
    vc3d_set_auto_fill_mode,
    vc3d_set_point_collection_color,
    vc3d_set_point_collection_metadata,
    vc3d_set_point_collection_tag,
    vc3d_set_point_windings_linked,
    vc3d_update_point,
)


class PointsToolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-points-test-")
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

    def _assert_passthrough(self, result, method: str, params: dict) -> None:
        self.assertEqual(result, {"echoedMethod": method, "echoedParams": params})

    async def test_add_collection_named(self) -> None:
        result = await vc3d_add_point_collection(name="corr")
        self._assert_wire("points.add_collection", {"name": "corr"})
        self._assert_passthrough(result, "points.add_collection", {"name": "corr"})

    async def test_add_collection_omitted_name_stripped(self) -> None:
        await vc3d_add_point_collection()
        # None name must be stripped, not sent as null.
        self._assert_wire("points.add_collection", {})

    async def test_update_point_all_fields(self) -> None:
        pos = {"x": 1.0, "y": 2.0, "z": 3.0}
        await vc3d_update_point(point_id=7, position=pos, winding=1.5)
        self._assert_wire(
            "points.update_point", {"pointId": 7, "position": pos, "winding": 1.5}
        )

    async def test_update_point_optionals_stripped(self) -> None:
        await vc3d_update_point(point_id=7)
        self._assert_wire("points.update_point", {"pointId": 7})

    async def test_update_point_can_clear_winding(self) -> None:
        await vc3d_update_point(point_id=7, clear_winding=True)
        self._assert_wire("points.update_point", {"pointId": 7, "winding": None})

    async def test_update_point_rejects_winding_and_clear(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            await vc3d_update_point(point_id=7, winding=1.5, clear_winding=True)

    async def test_remove_point(self) -> None:
        await vc3d_remove_point(point_id=9)
        self._assert_wire("points.remove_point", {"pointId": 9})

    async def test_clear_collection_by_name(self) -> None:
        await vc3d_clear_point_collection(collection="corr")
        self._assert_wire("points.clear_collection", {"collection": "corr"})

    async def test_clear_collection_by_id(self) -> None:
        await vc3d_clear_point_collection(collection_id=3)
        self._assert_wire("points.clear_collection", {"collectionId": 3})

    async def test_clear_all(self) -> None:
        await vc3d_clear_all_points()
        self._assert_wire("points.clear_all", {})

    async def test_rename_collection(self) -> None:
        await vc3d_rename_point_collection(new_name="fresh", collection_id=2)
        self._assert_wire(
            "points.rename_collection", {"collectionId": 2, "newName": "fresh"}
        )

    async def test_set_collection_color(self) -> None:
        await vc3d_set_point_collection_color(color=[0.1, 0.2, 0.3], collection="corr")
        self._assert_wire(
            "points.set_collection_color", {"collection": "corr", "color": [0.1, 0.2, 0.3]}
        )

    async def test_set_collection_metadata(self) -> None:
        await vc3d_set_point_collection_metadata(
            absolute_winding_number=False, collection_id=1
        )
        self._assert_wire(
            "points.set_collection_metadata",
            {"collectionId": 1, "absoluteWindingNumber": False},
        )

    async def test_set_collection_tag(self) -> None:
        await vc3d_set_point_collection_tag(key="k", value="v", collection="corr")
        self._assert_wire(
            "points.set_collection_tag", {"collection": "corr", "key": "k", "value": "v"}
        )

    async def test_remove_collection_tag(self) -> None:
        await vc3d_remove_point_collection_tag(key="k", collection_id=4)
        self._assert_wire(
            "points.remove_collection_tag", {"collectionId": 4, "key": "k"}
        )

    async def test_set_windings_linked(self) -> None:
        await vc3d_set_point_windings_linked(
            linked_collection_ids=[2, 3], collection_id=1
        )
        self._assert_wire(
            "points.set_windings_linked", {"collectionId": 1, "linkedCollectionIds": [2, 3]}
        )

    async def test_auto_fill_windings_with_constant(self) -> None:
        await vc3d_auto_fill_windings(mode="constant", collection="corr", constant=2.0)
        self._assert_wire(
            "points.auto_fill_windings",
            {"collection": "corr", "mode": "constant", "constant": 2.0},
        )

    async def test_auto_fill_windings_constant_stripped(self) -> None:
        await vc3d_auto_fill_windings(mode="incremental", collection_id=1)
        self._assert_wire(
            "points.auto_fill_windings", {"collectionId": 1, "mode": "incremental"}
        )

    async def test_set_auto_fill_mode(self) -> None:
        await vc3d_set_auto_fill_mode(mode="decremental", collection_id=1)
        self._assert_wire(
            "points.set_auto_fill_mode", {"collectionId": 1, "mode": "decremental"}
        )

    async def test_reset_windings(self) -> None:
        await vc3d_reset_windings()
        self._assert_wire("points.reset_windings", {})

    async def test_apply_anchor_offset(self) -> None:
        result = await vc3d_apply_anchor_offset(offset_x=1.5, offset_y=-2.0)
        self._assert_wire("points.apply_anchor_offset", {"offsetX": 1.5, "offsetY": -2.0})
        self._assert_passthrough(
            result, "points.apply_anchor_offset", {"offsetX": 1.5, "offsetY": -2.0}
        )

    async def test_save_json(self) -> None:
        await vc3d_save_points_json(path="/tmp/p.json")
        self._assert_wire("points.save_json", {"path": "/tmp/p.json"})

    async def test_load_json(self) -> None:
        await vc3d_load_points_json(path="/tmp/p.json")
        self._assert_wire("points.load_json", {"path": "/tmp/p.json"})

    async def test_save_segment_path(self) -> None:
        await vc3d_save_points_segment_path(segment_path="/tmp/seg")
        self._assert_wire("points.save_segment_path", {"segmentPath": "/tmp/seg"})

    async def test_load_segment_path(self) -> None:
        await vc3d_load_points_segment_path(segment_path="/tmp/seg")
        self._assert_wire("points.load_segment_path", {"segmentPath": "/tmp/seg"})


if __name__ == "__main__":
    unittest.main()
