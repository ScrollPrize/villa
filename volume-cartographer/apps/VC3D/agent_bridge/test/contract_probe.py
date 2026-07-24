from __future__ import annotations

import math

from dataclasses import dataclass, field
from typing import Any, Iterator

from bridge_client import BridgeError


@dataclass
class ProbeReport:
    attempted: int = 0
    skipped: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def detail(self) -> str:
        if self.ok:
            detail = f"{self.attempted - self.skipped} probes passed"
            if self.skipped:
                detail += f", {self.skipped} blocked by declared preconditions"
            return detail
        return f"{len(self.failures)}/{self.attempted} failed: {'; '.join(self.failures[:3])}"


def _schema_type(schema: dict[str, Any]) -> str:
    kinds = schema["type"]
    if not isinstance(kinds, list):
        return kinds
    non_null = [kind for kind in kinds if kind != "null"]
    if not non_null:
        raise ValueError(f"expected a non-null type: {schema}")
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
    invalid = {
        "string": 7,
        "number": "not-a-number",
        "integer": 1.5,
        "boolean": "not-a-boolean",
        "array": {},
        "object": [],
    }[_schema_type(schema)]
    kinds = schema["type"]
    if not isinstance(kinds, list) or not any(
        _value_matches_type(invalid, kind)
        for kind in kinds
        if kind != "null"
    ):
        return invalid
    for candidate in (True, "invalid", 1.5, [], {}):
        if not any(
            _value_matches_type(candidate, kind)
            for kind in kinds
            if kind != "null"
        ):
            return candidate
    raise ValueError(f"could not produce an invalid value: {schema}")


def _value_matches_type(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    return False


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
    declared_errors: set[int],
) -> None:
    report.attempted += 1
    try:
        result, _ = client.call(method, params, timeout=10.0)
        report.failures.append(
            f"{method}.{expected_param}: expected -32602, got {list(result)[:4]}"
        )
    except BridgeError as error:
        if error.code != -32602 and error.code in declared_errors:
            report.skipped += 1
            return
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
        declared_errors = set(method_contract["errors"])
        for name in params_schema.get("required", []):
            params = _required_params(params_schema)
            params.pop(name)
            _record_expected_param_error(
                report,
                client,
                method,
                params,
                name,
                declared_errors,
            )

        for name, schema in params_schema["properties"].items():
            params = _required_params(params_schema)
            params[name] = _invalid_value(schema)
            _record_expected_param_error(
                report,
                client,
                method,
                params,
                name,
                declared_errors,
            )

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
                declared_errors,
            )

        for keyword in ("exclusiveMinimum", "minimum", "maximum", "exclusiveMaximum"):
            for path, schema in _walk(params_schema, keyword):
                bound = schema[keyword]
                if keyword == "exclusiveMinimum":
                    value = bound
                elif keyword == "minimum":
                    value = math.nextafter(bound, -math.inf)
                elif keyword == "maximum":
                    value = math.nextafter(bound, math.inf)
                else:
                    value = bound
                params = _params_with_value(params_schema, path, value)
                _record_expected_param_error(
                    report,
                    client,
                    method,
                    params,
                    ".".join(path),
                    declared_errors,
                )
    return report
