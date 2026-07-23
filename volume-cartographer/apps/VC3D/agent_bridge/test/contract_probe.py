from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from bridge_client import BridgeError


@dataclass
class ProbeReport:
    attempted: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def detail(self) -> str:
        if self.ok:
            return f"{self.attempted} probes passed"
        return f"{len(self.failures)}/{self.attempted} failed: {'; '.join(self.failures[:3])}"


def load_contract(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _schema_type(schema: dict[str, Any]) -> str:
    kinds = schema["type"]
    if not isinstance(kinds, list):
        return kinds
    non_null = [kind for kind in kinds if kind != "null"]
    if len(non_null) != 1:
        raise ValueError(f"expected one non-null type: {schema}")
    return non_null[0]


def _sample(schema: dict[str, Any]) -> Any:
    if "enum" in schema:
        return schema["enum"][0]

    kind = _schema_type(schema)
    if kind == "string":
        return "probe"
    if kind == "number":
        return 1.0
    if kind == "integer":
        return 1
    if kind == "boolean":
        return True
    if kind == "array":
        return [_sample(schema["items"])]
    if kind == "object":
        properties = schema.get("properties", {})
        return {
            name: _sample(property_schema)
            for name, property_schema in properties.items()
        }
    raise ValueError(f"unsupported schema type: {kind}")


def _invalid_value(schema: dict[str, Any]) -> Any:
    return {
        "string": 7,
        "number": "not-a-number",
        "integer": 1.5,
        "boolean": "not-a-boolean",
        "array": {},
        "object": [],
    }[_schema_type(schema)]


def _required_params(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema["properties"]
    return {
        name: _sample(properties[name])
        for name in schema.get("required", [])
    }


def _set_path(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    current = value
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = replacement


def _params_with_value(
    params_schema: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> dict[str, Any]:
    params = _required_params(params_schema)
    if len(path) == 1:
        params[path[0]] = value
        return params

    top_schema = params_schema["properties"][path[0]]
    container = _sample(top_schema)
    if not isinstance(container, dict):
        raise ValueError(f"{path[0]} is not an object")
    _set_path(container, path[1:], value)
    params[path[0]] = container
    return params


def _walk(
    schema: dict[str, Any],
    keyword: str,
    path: tuple[str, ...] = (),
) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    if keyword in schema:
        yield path, schema
    for name, child in schema.get("properties", {}).items():
        yield from _walk(child, keyword, (*path, name))


def _invalid_enum_value(schema: dict[str, Any]) -> Any:
    kind = _schema_type(schema)
    if kind == "string":
        return "__vc3d_invalid__"
    if kind in {"integer", "number"}:
        values = schema["enum"]
        return max(values) + 1
    raise ValueError(f"unsupported enum type: {kind}")


def _record_expected_param_error(
    report: ProbeReport,
    client: Any,
    method: str,
    params: dict[str, Any],
    expected_param: str,
) -> None:
    report.attempted += 1
    try:
        result, _ = client.call(method, params, timeout=10.0)
        report.failures.append(
            f"{method}.{expected_param}: expected -32602, got {list(result)[:4]}"
        )
    except BridgeError as error:
        actual_param = error.data.get("param")
        if error.code != -32602 or actual_param != expected_param:
            report.failures.append(
                f"{method}.{expected_param}: got code={error.code} param={actual_param!r}"
            )
    except Exception as error:  # noqa: BLE001
        report.failures.append(
            f"{method}.{expected_param}: {type(error).__name__}: {error}"
        )


def probe_invalid_inputs(client: Any, contract: dict[str, Any]) -> ProbeReport:
    report = ProbeReport()
    for method, method_contract in contract["methods"].items():
        params_schema = method_contract["params"]
        for name in params_schema.get("required", []):
            params = _required_params(params_schema)
            params.pop(name)
            _record_expected_param_error(report, client, method, params, name)

        for name, schema in params_schema["properties"].items():
            params = _required_params(params_schema)
            params[name] = _invalid_value(schema)
            _record_expected_param_error(report, client, method, params, name)

        for path, schema in _walk(params_schema, "enum"):
            if not path:
                continue
            params = _params_with_value(
                params_schema,
                path,
                _invalid_enum_value(schema),
            )
            _record_expected_param_error(
                report,
                client,
                method,
                params,
                ".".join(path),
            )

        for keyword in ("exclusiveMinimum", "minimum", "maximum", "exclusiveMaximum"):
            for path, schema in _walk(params_schema, keyword):
                bound = schema[keyword]
                if keyword == "exclusiveMinimum":
                    value = bound
                elif keyword == "minimum":
                    value = bound - 1
                elif keyword == "maximum":
                    value = bound + 1
                else:
                    value = bound
                params = _params_with_value(params_schema, path, value)
                _record_expected_param_error(
                    report,
                    client,
                    method,
                    params,
                    ".".join(path),
                )
    return report


def _lookup(result: dict[str, Any], path: str) -> Any:
    value: Any = result
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _outside(bound: int | float, lower: bool, integer: bool) -> int | float:
    offset = 7 if integer else 7.5
    return bound - offset if lower else bound + offset


def probe_clamps(client: Any, contract: dict[str, Any]) -> ProbeReport:
    report = ProbeReport()
    for method, method_contract in contract["methods"].items():
        params_schema = method_contract["params"]
        for path, schema in _walk(params_schema, "x-clamp"):
            lower, upper = schema["x-clamp"]
            for is_lower, bound in ((True, lower), (False, upper)):
                if bound is None:
                    continue
                params = _params_with_value(
                    params_schema,
                    path,
                    _outside(bound, is_lower, _schema_type(schema) == "integer"),
                )
                report.attempted += 1
                try:
                    result, _ = client.call(method, params, timeout=10.0)
                    result_path = schema.get("x-result", ".".join(path))
                    actual = _lookup(result, result_path)
                    if (not isinstance(actual, (int, float)) or
                            not math.isclose(actual, bound, abs_tol=1e-6)):
                        report.failures.append(
                            f"{method}.{'.'.join(path)}: expected {bound}, got {actual!r}"
                        )
                except Exception as error:  # noqa: BLE001
                    report.failures.append(
                        f"{method}.{'.'.join(path)}: {type(error).__name__}: {error}"
                    )

        for path, schema in _walk(params_schema, "x-ordered-range"):
            order = schema["x-ordered-range"]
            lower_name = order["lower"]
            upper_name = order["upper"]
            gap = order["minimumGap"]
            ceiling = order["ceiling"]
            result_paths = order.get("result", {})
            cases = [
                ({lower_name: 10, upper_name: 5}, 10, 10 + gap),
                ({lower_name: ceiling, upper_name: 0}, ceiling, ceiling),
            ]
            for value, expected_lower, expected_upper in cases:
                params = _params_with_value(params_schema, path, value)
                report.attempted += 1
                try:
                    result, _ = client.call(method, params, timeout=10.0)
                    default_prefix = ".".join(path)
                    lower_path = result_paths.get(
                        lower_name,
                        f"{default_prefix}.{lower_name}",
                    )
                    upper_path = result_paths.get(
                        upper_name,
                        f"{default_prefix}.{upper_name}",
                    )
                    actual_lower = _lookup(result, lower_path)
                    actual_upper = _lookup(result, upper_path)
                    if (not math.isclose(actual_lower, expected_lower, abs_tol=1e-6) or
                            not math.isclose(actual_upper, expected_upper, abs_tol=1e-6)):
                        report.failures.append(
                            f"{method}.{'.'.join(path)}: expected "
                            f"({expected_lower}, {expected_upper}), got "
                            f"({actual_lower!r}, {actual_upper!r})"
                        )
                except Exception as error:  # noqa: BLE001
                    report.failures.append(
                        f"{method}.{'.'.join(path)}: {type(error).__name__}: {error}"
                    )
    return report
