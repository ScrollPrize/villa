from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from typing import Any

from vc3d_mcp import core
from vc3d_mcp import tools as _tools  # noqa: F401 - registers the MCP tools


VC_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIR = VC_ROOT / "apps/VC3D/agent_bridge/schema"
SPEC_PATH = VC_ROOT / "apps/VC3D/agent_bridge/SPEC.md"
DESCRIPTION_PATH = VC_ROOT / "apps/VC3D/agent_bridge/rpc_description.json"


def _load_contracts() -> list[dict[str, Any]]:
    contracts = []
    for path in sorted(CONTRACT_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            contract = json.load(stream)
        contract["_path"] = path
        contracts.append(contract)
    return contracts


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
        cls.contracts = _load_contracts()
        cls.methods = {
            method: method_contract
            for contract in cls.contracts
            for method, method_contract in contract["methods"].items()
        }
        with DESCRIPTION_PATH.open(encoding="utf-8") as stream:
            cls.described_methods = json.load(stream)["methods"]

    def test_contract_shape(self) -> None:
        self.assertTrue(self.contracts)
        domains: set[str] = set()
        methods: set[str] = set()
        tools: set[str] = set()
        for domain_contract in self.contracts:
            domain = domain_contract["domain"]
            self.assertEqual(domain_contract["contractVersion"], 1)
            self.assertNotIn(domain, domains)
            domains.add(domain)
            self.assertEqual(
                domain_contract["_path"].stem,
                domain.replace(".", "_"),
            )
            for method, contract in domain_contract["methods"].items():
                with self.subTest(domain=domain, method=method):
                    self.assertNotIn(method, methods)
                    methods.add(method)
                    self.assertTrue(
                        method == domain or method.startswith(f"{domain}.")
                    )
                    self.assertEqual(contract["params"]["type"], "object")
                    self.assertIsInstance(contract["params"]["properties"], dict)
                    self.assertEqual(
                        len(contract["errors"]),
                        len(set(contract["errors"])),
                    )
                    self.assertTrue(
                        all(
                            isinstance(code, int) and code < 0
                            for code in contract["errors"]
                        )
                    )
                    if contract["params"]["properties"]:
                        self.assertIn(-32602, contract["errors"])
                    self._assert_extensions(contract["params"])

                    tool = contract["mcp"]["tool"]
                    self.assertNotIn(tool, tools)
                    tools.add(tool)

                    properties = contract["params"]["properties"]
                    mcp_contract = contract["mcp"]
                    renames = mcp_contract.get("paramRenames", {})
                    self.assertLessEqual(renames.keys(), properties.keys())
                    self.assertEqual(
                        len(renames.values()),
                        len(set(renames.values())),
                    )
                    extra_params = mcp_contract.get("extraParams", {})
                    self.assertIsInstance(extra_params, dict)
                    self.assertFalse(
                        set(extra_params)
                        & {renames.get(name, name) for name in properties}
                    )
                    for schema in extra_params.values():
                        self._assert_extensions(schema)

    def test_contracts_are_present_in_live_description(self) -> None:
        remaining = set(self.methods) - set(self.described_methods)
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                VC_ROOT / "apps/VC3D/agent_bridge"
            ).glob("AgentBridge*.cpp")
        )
        registered = set(
            re.findall(r'_handlers\.insert\("([^"]+)"', sources)
        )
        self.assertLessEqual(remaining, registered)

    def test_live_description_shape(self) -> None:
        self.assertTrue(self.described_methods)
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
        contracts = {
            **self.methods,
            **{
                method: contract
                for method, contract in self.described_methods.items()
                if "mcp" in contract
            },
        }
        for method, contract in contracts.items():
            tool = contract["mcp"]["tool"]
            row_start = f"| `{tool}` | `{method}` |"
            self.assertIn(row_start, spec)

    async def test_fastmcp_input_shapes_match_contract(self) -> None:
        registered = {tool.name: tool.inputSchema for tool in await core.mcp.list_tools()}
        expected_tools = {contract["mcp"]["tool"] for contract in self.methods.values()}
        self.assertLessEqual(expected_tools, registered.keys())

        for method, contract in self.methods.items():
            tool_name = contract["mcp"]["tool"]
            with self.subTest(method=method, tool=tool_name):
                self._assert_mcp_contract(contract, registered)

        for method, contract in self.described_methods.items():
            if "mcp" not in contract:
                continue
            tool_name = contract["mcp"]["tool"]
            with self.subTest(method=method, tool=tool_name):
                self._assert_mcp_contract(contract, registered)

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
