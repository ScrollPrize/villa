from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from vc3d_mcp import core
from vc3d_mcp import tools as _tools  # noqa: F401 - registers the MCP tools


VC_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = VC_ROOT / "apps/VC3D/agent_bridge/SPEC.md"
DESCRIPTION_PATH = VC_ROOT / "apps/VC3D/agent_bridge/rpc_description.json"
EXPECTED_RPC_METHODS = 118
MCP_ONLY_TOOLS = {"vc3d_wait_job"}


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


class BridgeContractTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with DESCRIPTION_PATH.open(encoding="utf-8") as stream:
            cls.described_methods = json.load(stream)["methods"]

    def test_live_description_shape(self) -> None:
        self.assertEqual(len(self.described_methods), EXPECTED_RPC_METHODS)
        tools: set[str] = set()
        for method, contract in self.described_methods.items():
            with self.subTest(method=method):
                self.assertEqual(contract["params"]["type"], "object")
                self.assertIsInstance(contract["params"]["properties"], dict)
                self.assertEqual(
                    len(contract["errors"]),
                    len(set(contract["errors"])),
                )
                mcp_contract = contract.get("mcp")
                if mcp_contract is None:
                    continue
                tool = mcp_contract["tool"]
                self.assertNotIn(tool, tools)
                tools.add(tool)
                renames = mcp_contract.get("paramRenames", {})
                self.assertLessEqual(
                    renames.keys(),
                    contract["params"]["properties"].keys(),
                )
                mapped = {
                    renames.get(name, name)
                    for name in contract["params"]["properties"]
                }
                extras = mcp_contract.get("extraParams", [])
                self.assertEqual(len(extras), len(set(extras)))
                self.assertFalse(mapped & set(extras))

    def test_spec_lists_every_mcp_mapping(self) -> None:
        spec = SPEC_PATH.read_text(encoding="utf-8")
        for method, contract in self.described_methods.items():
            if "mcp" not in contract:
                continue
            tool = contract["mcp"]["tool"]
            row_start = f"| `{tool}` | `{method}` |"
            self.assertIn(row_start, spec)

    async def test_fastmcp_input_shapes_match_contract(self) -> None:
        registered = {tool.name: tool.inputSchema for tool in await core.mcp.list_tools()}
        described_tools = {
            contract["mcp"]["tool"]
            for contract in self.described_methods.values()
            if "mcp" in contract
        }
        self.assertEqual(set(registered), described_tools | MCP_ONLY_TOOLS)

        for method, contract in self.described_methods.items():
            if "mcp" not in contract:
                continue
            tool_name = contract["mcp"]["tool"]
            with self.subTest(method=method, tool=tool_name):
                self._assert_mcp_contract(contract, registered)

    async def test_volume_discovery_descriptions_distinguish_catalog_from_project(
        self,
    ) -> None:
        registered = {
            tool.name: tool.description for tool in await core.mcp.list_tools()
        }
        catalog = " ".join(registered["vc3d_list_catalog_samples"].split())
        current = " ".join(registered["vc3d_list_attached_volumes"].split())

        self.assertIn("available to open", catalog)
        self.assertIn("no volume package is currently loaded", catalog)
        self.assertIn("vc3d_open_catalog_sample", catalog)
        self.assertIn("already attached", current)
        self.assertIn("does not discover", current)
        self.assertIn("vc3d_list_catalog_samples", current)

    def _assert_mcp_contract(
        self,
        contract: dict[str, Any],
        registered: dict[str, dict[str, Any]],
    ) -> None:
        mcp_contract = contract["mcp"]
        tool_name = mcp_contract["tool"]
        self.assertIn(tool_name, registered)
        actual = registered[tool_name]
        expected = contract["params"]
        renames = mcp_contract.get("paramRenames", {})
        extra_params = mcp_contract.get("extraParams", {})
        if isinstance(extra_params, dict):
            expected = {
                **expected,
                "properties": {
                    **expected["properties"],
                    **extra_params,
                },
            }
            allowed_extra_properties: set[str] = set()
        else:
            allowed_extra_properties = set(extra_params)
        self._assert_schema_matches(
            expected,
            actual,
            actual,
            renames,
            allowed_extra_properties,
        )

    def _assert_schema_matches(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
        root: dict[str, Any],
        renames: dict[str, str] | None = None,
        allowed_extra_properties: set[str] | None = None,
    ) -> None:
        actual = _without_null(actual, root)
        expected_types = expected["type"]
        if not isinstance(expected_types, list):
            expected_types = [expected_types]
        non_null_types = [kind for kind in expected_types if kind != "null"]
        expected_type = actual.get("type")
        self.assertIn(expected_type, non_null_types)

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
        self.assertEqual(
            set(actual_properties),
            set(expected_properties) | (allowed_extra_properties or set()),
        )
        for name, property_schema in expected_properties.items():
            self._assert_schema_matches(
                property_schema,
                actual_properties[name],
                root,
            )


if __name__ == "__main__":
    unittest.main()
