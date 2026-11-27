#!/usr/bin/env python3
"""Self-test for the MCP observe option and vc3d_observe.

A stateful fake bridge serves state.get with a revision, broadcasts
state.changed when a workspace switch changes state, and answers screenshots,
so the tests check the JSON Patch state diff, the changed-field bookkeeping
whichever side of a reply the notification arrives on, the image return
shape, capture failures, and that the interaction tools keep their output
schemas.

Run with:
    python3 -m unittest tests.tools.test_observe -v
"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
import unittest
from typing import Any

from mcp.server.fastmcp import Image

from tests.support import FakeBridgeServer
from tests.support.agent_bridge import TINY_PNG_B64
from vc3d_mcp import core
from vc3d_mcp.core import state_diff
from vc3d_mcp.tools.session import vc3d_observe, vc3d_switch_workspace
from vc3d_mcp.tools.viewer import vc3d_click


class ObserveBridge(FakeBridgeServer):
    def __init__(self, socket_path: str):
        super().__init__(socket_path)
        self.revision = 3
        self.state: dict[str, Any] = {
            "vpkg": {"path": "/tmp/p.volpkg.json"},
            "workspace": {"current": "main", "available": []},
            "viewers": [{"viewerId": "v1", "surfName": "segmentation", "visible": True}],
            "jobs": [{"jobId": "job-1", "consoleTail": ["a"], "progressHistory": [1]}],
        }
        self.screenshot_error: dict[str, Any] | None = None
        self.screenshot_targets: list[str] = []
        self.change_with_screenshot = False

    async def notify_changed(self, writer, fields: list[str]) -> None:
        self.revision += 1
        await self.send(
            writer,
            {
                "jsonrpc": "2.0",
                "method": "state.changed",
                "params": {"revision": self.revision, "changed": fields},
            },
        )

    async def handle_message(self, message: dict, writer) -> None:
        method = message.get("method")
        req_id = message.get("id")
        params = message.get("params") or {}
        if method == "state.get":
            result = copy.deepcopy(self.state)
            result["revision"] = self.revision
            # Console tails change on every read; observations ignore them.
            result["jobs"][0]["consoleTail"].append("tick")
            await self.reply(writer, req_id, result=result)
        elif method == "workspace.switch":
            previous = self.state["workspace"]["current"]
            self.state["workspace"]["current"] = params["name"]
            self.state["viewers"][0]["visible"] = params["name"] == "main"
            # After the reply: the next state.get must still collect it.
            await self.reply(
                writer, req_id, result={"workspace": params["name"], "previous": previous}
            )
            await self.notify_changed(writer, ["viewers", "workspace"])
        elif method == "canvas.click":
            await self.reply(writer, req_id, result={"clicked": True})
        elif method == "screenshot.capture":
            self.screenshot_targets.append(params.get("target"))
            if self.change_with_screenshot:
                # The bridge sends a request's state.changed before its reply.
                await self.notify_changed(writer, ["overlay"])
            if self.screenshot_error is not None:
                await self.reply(writer, req_id, error=self.screenshot_error)
            else:
                await self.reply(
                    writer,
                    req_id,
                    result={"base64": TINY_PNG_B64, "width": 1, "height": 1,
                            "format": "png", "filePath": None},
                )
        elif method == "test.human_change":
            # Stands in for a person changing the volume in the GUI.
            self.state["volume"] = {"id": "v2"}
            await self.notify_changed(writer, ["volume"])
            await self.reply(writer, req_id, result={})
        else:
            await super().handle_message(message, writer)


class StateDiffTest(unittest.TestCase):
    def test_nested_add_remove_replace(self) -> None:
        before = {"a": 1, "b": {"c": True, "d": [1, 2]}, "gone": None}
        after = {"a": 2, "b": {"c": True, "d": [1, 3]}, "new": "x"}
        self.assertEqual(
            state_diff(before, after),
            [
                {"op": "replace", "path": "/a", "value": 2},
                {"op": "replace", "path": "/b/d/1", "value": 3},
                {"op": "remove", "path": "/gone"},
                {"op": "add", "path": "/new", "value": "x"},
            ],
        )

    def test_resized_list_is_replaced_whole(self) -> None:
        self.assertEqual(
            state_diff({"v": [1]}, {"v": [1, 2]}),
            [{"op": "replace", "path": "/v", "value": [1, 2]}],
        )

    def test_pointer_escaping_and_bool_int(self) -> None:
        self.assertEqual(
            state_diff({"a/b~c": 1}, {"a/b~c": True}),
            [{"op": "replace", "path": "/a~1b~0c", "value": True}],
        )
        self.assertEqual(state_diff({"x": 1}, {"x": 1.0}), [])
        self.assertEqual(state_diff([], []), [])


class ObserveToolTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="vc3d-observe-test-")
        self.socket_path = os.path.join(self.tmp_dir, "fake-bridge.sock")
        self.fake = ObserveBridge(self.socket_path)
        await self.fake.start()
        core.configure_client(self.socket_path, request_timeout=5)
        core._last_observed_state = None

    async def asyncTearDown(self) -> None:
        await core._get_client().close()
        await self.fake.stop()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        core._last_observed_state = None

    async def test_none_returns_plain_result(self) -> None:
        result = await vc3d_switch_workspace(name="lasagna")
        self.assertEqual(result, {"workspace": "lasagna", "previous": "main"})
        methods = [m["method"] for m in self.fake.received_requests]
        self.assertEqual(methods, ["workspace.switch"])

    async def test_state_diff_around_call(self) -> None:
        result = await vc3d_switch_workspace(name="spiral", observe="state")
        observation = result["observation"]
        self.assertEqual(result["workspace"], "spiral")
        self.assertEqual(observation["revision"], 4)
        self.assertEqual(observation["changed"], ["viewers", "workspace"])
        self.assertEqual(
            observation["stateDiff"],
            [
                {"op": "replace", "path": "/viewers/0/visible", "value": False},
                {"op": "replace", "path": "/workspace/current", "value": "spiral"},
            ],
        )
        methods = [m["method"] for m in self.fake.received_requests]
        self.assertEqual(methods, ["state.get", "workspace.switch", "state.get"])

    async def test_state_observation_on_a_click(self) -> None:
        result = await vc3d_click(position={"x": 1, "y": 2, "z": 3}, observe="state")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["clicked"], True)
        self.assertEqual(result["observation"]["stateDiff"], [])
        self.assertNotIn("screenshot.capture", [m["method"] for m in self.fake.received_requests])

    async def test_observe_screenshot_returns_image(self) -> None:
        result = await vc3d_observe(mode="screenshot", target="v1")
        self.assertIsInstance(result, list)
        payload, image = result
        self.assertIsInstance(image, Image)
        self.assertEqual(payload["observation"]["screenshot"]["target"], "v1")
        self.assertEqual(
            self.fake.params_for("screenshot.capture"),
            {"target": "v1", "maxDim": core.OBSERVE_SCREENSHOT_MAX_DIM},
        )
        self.assertNotIn("stateDiff", payload["observation"])

    async def test_observe_screenshot_failure_is_reported_not_raised(self) -> None:
        self.fake.screenshot_error = {"code": -32009, "message": "Capture target not visible"}
        result = await vc3d_observe(mode="screenshot", target="v1")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["observation"]["screenshotError"]["code"], -32009)

    async def test_changes_sent_with_the_screenshot_are_counted(self) -> None:
        await vc3d_observe(mode="state")
        self.fake.change_with_screenshot = True
        result = await vc3d_observe(mode="both")
        self.assertEqual(result[0]["observation"]["changed"], ["overlay"])

    async def test_interaction_tools_keep_output_schemas(self) -> None:
        tools = {tool.name: tool for tool in await core.mcp.list_tools()}
        for name in (
            "vc3d_click",
            "vc3d_shift_click",
            "vc3d_drag",
            "vc3d_key",
            "vc3d_wheel",
            "vc3d_center_viewer",
            "vc3d_zoom_viewer",
            "vc3d_rotate_viewer",
            "vc3d_switch_workspace",
        ):
            with self.subTest(tool=name):
                self.assertIsNotNone(tools[name].outputSchema)
                self.assertEqual(
                    tools[name].inputSchema["properties"]["observe"]["enum"],
                    ["none", "state"],
                )

    async def test_vc3d_observe_reports_changes_made_elsewhere(self) -> None:
        first = await vc3d_observe(mode="state")
        self.assertTrue(first["observation"]["baseline"])
        await core._call("test.human_change")
        second = await vc3d_observe(mode="state")
        observation = second["observation"]
        self.assertEqual(observation["changed"], ["volume"])
        self.assertEqual(
            observation["stateDiff"],
            [{"op": "add", "path": "/volume", "value": {"id": "v2"}}],
        )
        both = await vc3d_observe()
        self.assertIsInstance(both, list)
        self.assertEqual(both[0]["observation"]["stateDiff"], [])
        self.assertEqual(self.fake.screenshot_targets, ["window"])


if __name__ == "__main__":
    unittest.main()
