from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from vc3d_mcp import core
from vc3d_mcp import tools as _tools  # noqa: F401 - registers the MCP tools


VC_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = VC_ROOT / "apps/VC3D/agent_bridge/schema/viewer.json"
SPEC_PATH = VC_ROOT / "apps/VC3D/agent_bridge/SPEC.md"


def _load_contract() -> dict[str, Any]:
    with CONTRACT_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def _resolve_ref(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/$defs/"):
            raise AssertionError(f"unsupported schema reference: {ref}")
        schema = root["$defs"][ref.removeprefix("#/$defs/")]
    return schema


def _without_null(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    schema = _resolve_ref(schema, root)
    if "anyOf" not in schema:
        return schema
    choices = [
        choice
        for choice in schema["anyOf"]
        if _resolve_ref(choice, root).get("type") != "null"
    ]
    if len(choices) != 1:
        raise AssertionError(f"expected one non-null schema choice: {schema}")
    return _resolve_ref(choices[0], root)


class ViewerContractTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = _load_contract()
        cls.methods = cls.contract["methods"]

    def test_contract_shape(self) -> None:
        self.assertEqual(self.contract["contractVersion"], 1)
        self.assertEqual(self.contract["domain"], "viewer")
        for action in self.contract.get("probeSetup", []):
            self.assertTrue(action["method"])
            self.assertIsInstance(action.get("params", {}), dict)

        tools: set[str] = set()
        for method, contract in self.methods.items():
            self.assertTrue(method.startswith("viewer."))
            self.assertEqual(contract["params"]["type"], "object")
            self.assertIsInstance(contract["params"]["properties"], dict)
            self.assertEqual(len(contract["errors"]), len(set(contract["errors"])))
            self.assertTrue(all(isinstance(code, int) and code < 0 for code in contract["errors"]))
            if contract["params"]["properties"]:
                self.assertIn(-32602, contract["errors"])
            self._assert_extensions(contract["params"])

            probes = contract.get("errorProbes", [])
            probed_errors = {probe["code"] for probe in probes}
            if contract["params"]["properties"]:
                probed_errors.add(-32602)
            unprobed_errors = set(contract.get("unprobedErrors", []))
            self.assertFalse(probed_errors & unprobed_errors)
            self.assertEqual(
                set(contract["errors"]),
                probed_errors | unprobed_errors,
            )
            for probe in probes:
                self.assertTrue(probe["name"])
                self.assertTrue(probe["state"])
                self.assertIsInstance(probe["params"], dict)
                self.assertIn(probe["code"], contract["errors"])
                for action in (*probe.get("before", []), *probe.get("after", [])):
                    self.assertTrue(action["method"])
                    self.assertIsInstance(action.get("params", {}), dict)

            tool = contract["mcp"]["tool"]
            self.assertNotIn(tool, tools)
            tools.add(tool)

            properties = contract["params"]["properties"]
            renames = contract["mcp"].get("paramRenames", {})
            self.assertLessEqual(renames.keys(), properties.keys())
            self.assertEqual(len(renames.values()), len(set(renames.values())))

    def _assert_extensions(self, schema: dict[str, Any]) -> None:
        if "x-clamp" in schema:
            bounds = schema["x-clamp"]
            self.assertIn(schema["type"], {"integer", "number"})
            self.assertIsInstance(bounds, list)
            self.assertEqual(len(bounds), 2)
            self.assertTrue(all(bound is None or isinstance(bound, (int, float))
                                for bound in bounds))
            self.assertTrue(any(bound is not None for bound in bounds))
            if bounds[0] is not None and bounds[1] is not None:
                self.assertLess(bounds[0], bounds[1])
        if "x-result" in schema:
            self.assertIn("x-clamp", schema)
            self.assertIsInstance(schema["x-result"], str)
            self.assertTrue(schema["x-result"])
        if "x-ordered-range" in schema:
            order = schema["x-ordered-range"]
            self.assertEqual(schema["type"], "object")
            self.assertGreater(order["minimumGap"], 0)
            self.assertIsInstance(order["ceiling"], (int, float))
            properties = schema["properties"]
            self.assertIn(order["lower"], properties)
            self.assertIn(order["upper"], properties)
            for result_path in order.get("result", {}).values():
                self.assertIsInstance(result_path, str)
                self.assertTrue(result_path)
        for child in schema.get("properties", {}).values():
            self._assert_extensions(child)
        if "items" in schema:
            self._assert_extensions(schema["items"])

    def test_spec_lists_every_mcp_mapping(self) -> None:
        spec = SPEC_PATH.read_text(encoding="utf-8")
        for method, contract in self.methods.items():
            tool = contract["mcp"]["tool"]
            row_start = f"| `{tool}` | `{method}` |"
            self.assertIn(row_start, spec)

    async def test_fastmcp_input_shapes_match_contract(self) -> None:
        registered = {tool.name: tool.inputSchema for tool in await core.mcp.list_tools()}
        expected_tools = {contract["mcp"]["tool"] for contract in self.methods.values()}
        self.assertLessEqual(expected_tools, registered.keys())

        for method, contract in self.methods.items():
            tool_name = contract["mcp"]["tool"]
            actual = registered[tool_name]
            expected = contract["params"]
            renames = contract["mcp"].get("paramRenames", {})
            with self.subTest(method=method, tool=tool_name):
                self._assert_schema_matches(expected, actual, actual, renames)

    def _assert_schema_matches(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
        root: dict[str, Any],
        renames: dict[str, str] | None = None,
    ) -> None:
        actual = _without_null(actual, root)
        expected_types = expected["type"]
        if not isinstance(expected_types, list):
            expected_types = [expected_types]
        non_null_types = [kind for kind in expected_types if kind != "null"]
        self.assertEqual(len(non_null_types), 1)
        expected_type = non_null_types[0]
        self.assertEqual(actual.get("type"), expected_type)

        if "enum" in expected:
            self.assertEqual(actual.get("enum"), expected["enum"])
        if "default" in expected:
            self.assertEqual(actual.get("default"), expected["default"])

        expected_required = set(expected.get("required", []))
        actual_required = set(actual.get("required", []))
        if renames:
            expected_required = {renames.get(name, name) for name in expected_required}
        self.assertEqual(actual_required, expected_required)

        if expected_type == "array":
            self._assert_schema_matches(expected["items"], actual["items"], root)
            return
        if expected_type != "object":
            return

        renames = renames or {}
        expected_properties = {
            renames.get(name, name): schema
            for name, schema in expected.get("properties", {}).items()
        }
        actual_properties = actual.get("properties", {})
        self.assertEqual(set(actual_properties), set(expected_properties))
        for name, property_schema in expected_properties.items():
            self._assert_schema_matches(
                property_schema,
                actual_properties[name],
                root,
            )


if __name__ == "__main__":
    unittest.main()
