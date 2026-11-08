#!/usr/bin/env python3
"""Self-test for the ink_labels.* MCP tools.

A fake bridge echoes every request. For each tool we assert the bridge method
and the exact wire params: snake_case arguments map to the descriptor names,
``label_class`` goes out as ``class``, omitted optional values are stripped,
and the result passes through unchanged.

Run with:
    python3 -m unittest tests.tools.test_ink_labels -v
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from tests.support import EchoBridgeServer
from vc3d_mcp import core
from vc3d_mcp.tools.ink_labels import (
    vc3d_ink_labels_accept_suggestions,
    vc3d_ink_labels_cleanup,
    vc3d_ink_labels_close,
    vc3d_ink_labels_dataset_entry,
    vc3d_ink_labels_discard_suggestions,
    vc3d_ink_labels_fill,
    vc3d_ink_labels_history,
    vc3d_ink_labels_import,
    vc3d_ink_labels_open,
    vc3d_ink_labels_paint,
    vc3d_ink_labels_read,
    vc3d_ink_labels_save,
    vc3d_ink_labels_set_tool_state,
    vc3d_ink_labels_state,
    vc3d_ink_labels_suggest,
    vc3d_ink_labels_versions,
)


class InkLabelsToolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-inklabels-test-")
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

    async def test_state(self) -> None:
        result = await vc3d_ink_labels_state()
        self._assert_wire("ink_labels.state", {})
        self.assertEqual(result, {"echoedMethod": "ink_labels.state", "echoedParams": {}})

    async def test_open_defaults_and_version(self) -> None:
        await vc3d_ink_labels_open()
        self._assert_wire("ink_labels.open", {"recover": "fail", "readOnlyIfLocked": True, "startNewVersion": False})
        await vc3d_ink_labels_open(recover="restore", read_only_if_locked=False, version=2)
        self._assert_wire(
            "ink_labels.open",
            {"recover": "restore", "readOnlyIfLocked": False, "version": 2, "startNewVersion": False},
        )
        await vc3d_ink_labels_open(start_new_version=True)
        self._assert_wire("ink_labels.open", {"recover": "fail", "readOnlyIfLocked": True, "startNewVersion": True})

    async def test_paint_stroke_maps_class(self) -> None:
        stroke = {"points": [{"x": 1.5, "y": 2.5}, {"x": 10.0, "y": 2.5}], "radius": 3.0}
        await vc3d_ink_labels_paint(label_class="ink", stroke=stroke)
        self._assert_wire(
            "ink_labels.paint",
            {"class": "ink", "erase": False, "backgroundKeepsInk": False, "space": "label", "stroke": stroke},
        )

    async def test_paint_mask_with_expect(self) -> None:
        mask = {"path": "/tmp/m.png", "origin": {"x": 4, "y": 5}}
        await vc3d_ink_labels_paint(
            label_class="background",
            erase=True,
            background_keeps_ink=True,
            mask=mask,
            expect={"segment": "w045", "serial": 7},
        )
        self._assert_wire(
            "ink_labels.paint",
            {
                "class": "background",
                "erase": True,
                "backgroundKeepsInk": True,
                "space": "label",
                "mask": mask,
                "expect": {"segment": "w045", "serial": 7},
            },
        )

    async def test_paint_polygon_volume_space(self) -> None:
        polygon = {"points": [{"x": 1, "y": 2, "z": 3}, {"x": 4, "y": 2, "z": 3}, {"x": 4, "y": 6, "z": 3}]}
        await vc3d_ink_labels_paint(label_class="validation", space="volume", polygon=polygon)
        self._assert_wire(
            "ink_labels.paint",
            {"class": "validation", "erase": False, "backgroundKeepsInk": False, "space": "volume", "polygon": polygon},
        )

    async def test_paint_values_mask_without_class(self) -> None:
        mask = {"path": "/tmp/v.png", "origin": {"x": 0, "y": 0}, "format": "values"}
        await vc3d_ink_labels_paint(mask=mask)
        self._assert_wire(
            "ink_labels.paint",
            {"erase": False, "backgroundKeepsInk": False, "space": "label", "mask": mask},
        )

    async def test_read(self) -> None:
        rect = {"x": 0, "y": 0, "width": 64, "height": 32}
        await vc3d_ink_labels_read(rect=rect, path="/tmp/r.png")
        self._assert_wire("ink_labels.read", {"rect": rect, "format": "values", "path": "/tmp/r.png"})
        await vc3d_ink_labels_read(rect=rect, path="/tmp/r.tif", format="ink")
        self._assert_wire("ink_labels.read", {"rect": rect, "format": "ink", "path": "/tmp/r.tif"})

    async def test_history(self) -> None:
        await vc3d_ink_labels_history(action="undo")
        self._assert_wire("ink_labels.history", {"action": "undo", "steps": 1, "force": False})
        await vc3d_ink_labels_history(action="redo", steps=3, force=True)
        self._assert_wire("ink_labels.history", {"action": "redo", "steps": 3, "force": True})

    async def test_save(self) -> None:
        await vc3d_ink_labels_save()
        self._assert_wire("ink_labels.save", {"overwriteExternalEdits": False, "asNewVersion": False})
        await vc3d_ink_labels_save(overwrite_external_edits=True)
        self._assert_wire("ink_labels.save", {"overwriteExternalEdits": True, "asNewVersion": False})
        await vc3d_ink_labels_save(as_new_version=True)
        self._assert_wire("ink_labels.save", {"overwriteExternalEdits": False, "asNewVersion": True})

    async def test_versions_entry_close_discard(self) -> None:
        await vc3d_ink_labels_versions()
        self._assert_wire("ink_labels.versions", {})
        await vc3d_ink_labels_dataset_entry()
        self._assert_wire("ink_labels.dataset_entry", {})
        await vc3d_ink_labels_close()
        self._assert_wire("ink_labels.close", {})
        await vc3d_ink_labels_discard_suggestions()
        self._assert_wire("ink_labels.discard_suggestions", {})

    async def test_set_tool_state(self) -> None:
        await vc3d_ink_labels_set_tool_state()
        self._assert_wire("ink_labels.set_tool_state", {})
        await vc3d_ink_labels_set_tool_state(
            label_mode=True,
            label_class="validation",
            erase=False,
            tool="fill",
            radius=8.0,
            display={"show": True, "opacity": 0.5},
            assist={"guide": "labels", "minArea": 10, "unlabeledOnly": True},
        )
        self._assert_wire(
            "ink_labels.set_tool_state",
            {
                "labelMode": True,
                "class": "validation",
                "erase": False,
                "tool": "fill",
                "radius": 8.0,
                "display": {"show": True, "opacity": 0.5},
                "assist": {"guide": "labels", "minArea": 10, "unlabeledOnly": True},
            },
        )

    async def test_fill(self) -> None:
        await vc3d_ink_labels_fill(point={"x": 10.5, "y": 20.5})
        self._assert_wire(
            "ink_labels.fill",
            {
                "point": {"x": 10.5, "y": 20.5},
                "space": "label",
                "class": "ink",
                "erase": False,
                "backgroundKeepsInk": False,
            },
        )
        bound = {"x": 0, "y": 0, "width": 100, "height": 50}
        await vc3d_ink_labels_fill(
            point={"x": 1, "y": 2, "z": 3},
            space="volume",
            label_class="background",
            guide="prediction",
            threshold=140,
            above=False,
            connectivity=8,
            unlabeled_only=True,
            bound=bound,
        )
        self._assert_wire(
            "ink_labels.fill",
            {
                "point": {"x": 1, "y": 2, "z": 3},
                "space": "volume",
                "class": "background",
                "erase": False,
                "backgroundKeepsInk": False,
                "guide": "prediction",
                "threshold": 140,
                "above": False,
                "connectivity": 8,
                "unlabeledOnly": True,
                "bound": bound,
            },
        )

    async def test_suggest_and_accept(self) -> None:
        await vc3d_ink_labels_suggest()
        self._assert_wire("ink_labels.suggest", {})
        await vc3d_ink_labels_suggest(threshold=100, above=True, median=5, min_area=30)
        self._assert_wire("ink_labels.suggest", {"threshold": 100, "above": True, "median": 5, "minArea": 30})
        await vc3d_ink_labels_accept_suggestions(ids=[1, 3])
        self._assert_wire("ink_labels.accept_suggestions", {"ids": [1, 3], "all": False})
        await vc3d_ink_labels_accept_suggestions(all=True)
        self._assert_wire("ink_labels.accept_suggestions", {"all": True})

    async def test_cleanup(self) -> None:
        await vc3d_ink_labels_cleanup(kind="fill_holes")
        self._assert_wire("ink_labels.cleanup", {"kind": "fill_holes"})
        await vc3d_ink_labels_cleanup(kind="grow", size=4, bound={"x": 1, "y": 2, "width": 3, "height": 4})
        self._assert_wire(
            "ink_labels.cleanup",
            {"kind": "grow", "size": 4, "bound": {"x": 1, "y": 2, "width": 3, "height": 4}},
        )

    async def test_import(self) -> None:
        await vc3d_ink_labels_import(path="/data/w045_inklabels.png")
        self._assert_wire("ink_labels.import", {"path": "/data/w045_inklabels.png", "allowResample": False})
        await vc3d_ink_labels_import(
            path="/data/x.tif", label="validation", allow_resample=True, legacy_mask_as="ignore"
        )
        self._assert_wire(
            "ink_labels.import",
            {"path": "/data/x.tif", "label": "validation", "allowResample": True, "legacyMaskAs": "ignore"},
        )


if __name__ == "__main__":
    unittest.main()
