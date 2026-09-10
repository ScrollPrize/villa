"""Draft JSON Schema. State/authorization constraints additionally live in Engine."""

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Villa cooperative GPU manager protocol 1 (draft) envelope",
    "$defs": {
        "allocation": {
            "type": "object", "additionalProperties": False,
            "required": ["cpu_cores", "gpus"],
            "properties": {
                "cpu_cores": {"type": "integer", "minimum": 0},
                "gpus": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}},
            },
        },
        "process": {
            "type": "object", "additionalProperties": False,
            "required": ["pid", "start_us", "boot_us"],
            "properties": {key: {"type": "integer", "minimum": 1}
                           for key in ("pid", "start_us", "boot_us")},
        },
    },
    "type": "object", "additionalProperties": False,
    "required": ["version", "id", "op", "body"],
    "properties": {
        "version": {"const": 1}, "id": {"type": "string", "format": "uuid"},
        "server_epoch": {"type": "string", "format": "uuid"},
        "op": {"enum": ["hello", "register", "set_request", "poll", "report", "set_paused",
                         "set_override", "clear_override", "status", "finish"]},
        "body": {"type": "object"},
    },
}

UUID = {"type": "string", "format": "uuid"}
INTEGER = {"type": "integer", "minimum": 0}
TOKEN = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
REQUEST_PROPERTIES = {
    **{key: {"type": "integer", "minimum": 1} for key in (
        "max_cpu_cores", "max_gpus", "min_vram_bytes_per_gpu", "max_vram_bytes_per_gpu")},
    "allowed_gpu_uuids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
    "priority": {"type": "integer"}, "exclusive": {"type": "boolean"},
}
SCHEMA["$defs"]["request"] = {"type": "object", "additionalProperties": False,
                              "properties": REQUEST_PROPERTIES, "required": list(REQUEST_PROPERTIES)}
TASK = {"task_id": UUID, "token": TOKEN}
FIELDS = {
    "hello": {"versions": {"type": "array", "items": {"type": "integer"}},
              "required_capabilities": {"type": "array", "items": {"type": "string"}}},
    "register": {**TASK, "label": {"type": "string", "minLength": 1, "maxLength": 256},
                 "root": {"$ref": "#/$defs/process"}, "request": {"$ref": "#/$defs/request"},
                 "reconfigure_mode": {"const": "full_release"}},
    "set_request": {**TASK, "request": {"$ref": "#/$defs/request"}, "expected_request_revision": INTEGER},
    "poll": {**TASK, "last_generation": INTEGER},
    "report": {**TASK, "generation": INTEGER, "phase": {"enum": ["applied", "released", "failed"]},
               "held": {"$ref": "#/$defs/allocation"},
               "processes": {"type": "array", "maxItems": 4096, "items": {"$ref": "#/$defs/process"}},
               "detail": {"type": "string", "maxLength": 4096}},
    "set_paused": {**TASK, "paused": {"type": "boolean"}, "scope": {"enum": ["self", "controller"]}},
    "set_override": {"expected_policy_revision": INTEGER,
                     "allocations": {"type": "object", "additionalProperties": {"$ref": "#/$defs/allocation"}},
                     "bound_task_id": {"anyOf": [UUID, {"type": "null"}]}},
    "clear_override": {"expected_policy_revision": INTEGER},
    "status": {"after_order": {"type": "integer", "minimum": -1},
               "limit": {"type": "integer", "minimum": 1, "maximum": 16}},
    "finish": {**TASK, "outcome": {"type": "string", "maxLength": 4096}},
}
SCHEMA["allOf"] = [
    {"if": {"properties": {"op": {"const": op}}},
     "then": {"properties": {"body": {"type": "object", "additionalProperties": False,
                 "properties": properties,
                 "required": [key for key in properties if not (op == "status" or key == "token" and op in ("set_request", "set_paused"))]}},
              **({"required": ["server_epoch"]} if op != "hello" else {})}}
    for op, properties in FIELDS.items()
]

# The response carries no operation name; each operation's body is selected by
# the caller's pending request. Share definitions with the request schema.
TASK_PROPERTIES = {
    "task_id": UUID, "label": {"type": "string"}, "uid": INTEGER,
    "root": {"$ref": "#/$defs/process"}, "request": {"$ref": "#/$defs/request"},
    "request_revision": INTEGER, "order": INTEGER, "generation": INTEGER,
    "phase": {"enum": ["waiting", "initialize", "applied", "release", "finished"]},
    "reason": {"type": "string"},
    "paused": {"type": "object", "additionalProperties": False, "required": ["self", "controller"],
               "properties": {key: {"type": "boolean"} for key in ("self", "controller")}},
    **{key: {"$ref": "#/$defs/allocation"} for key in ("held", "granted", "target")},
    "terminal": {"type": "boolean"}, "last_contact_ms": INTEGER,
}
SCHEMA["$defs"]["task"] = {"type": "object", "additionalProperties": False,
                          "properties": TASK_PROPERTIES, "required": list(TASK_PROPERTIES)}


def response_schema(op):
    """Draft response schema for one known pending request operation."""
    if op not in FIELDS:
        raise ValueError("unknown operation")
    if op == "hello":
        body = {"capabilities": {"type": "array", "items": {"type": "string"}},
                "inventory": {"$ref": "#/$defs/allocation"}, "peer_uid": INTEGER}
    elif op == "status":
        body = {"inventory": {"$ref": "#/$defs/allocation"}, "policy_revision": INTEGER,
                "override": {"type": ["object", "null"]}, "draining": {"type": "boolean"},
                "blocked_gpu_uuids": {"type": "array", "items": {"type": "string"}},
                "tasks": {"type": "array", "maxItems": 16, "items": {"$ref": "#/$defs/task"}},
                "next_after_order": {"type": ["integer", "null"]}}
    elif op in ("set_override", "clear_override"):
        body = {"policy_revision": INTEGER}
    else:
        body = {"task": {"$ref": "#/$defs/task"}}
        if op == "register":
            body["token"] = TOKEN
    envelope = {"version": {"const": 1}, "id": {"anyOf": [UUID, {"type": "null"}]},
                "server_epoch": UUID, "ok": {"type": "boolean"}}
    return {"$schema": SCHEMA["$schema"], "$defs": SCHEMA["$defs"], "oneOf": [
        {"type": "object", "additionalProperties": False,
         "required": [*envelope, "body"], "properties": {**envelope, "ok": {"const": True},
            "body": {"type": "object", "additionalProperties": False, "required": list(body), "properties": body}}},
        {"type": "object", "additionalProperties": False,
         "required": [*envelope, "error"], "properties": {**envelope, "ok": {"const": False},
            "error": {"type": "object", "additionalProperties": False, "required": ["code", "message", "details"],
                      "properties": {"code": {"type": "string"}, "message": {"type": "string"}, "details": {"type": "object"}}}}},
    ]}


if __name__ == "__main__":
    import json
    print(json.dumps(SCHEMA, indent=2))
