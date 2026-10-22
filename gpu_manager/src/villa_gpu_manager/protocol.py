"""Strict, bounded JSON framing shared by clients and server."""

import json
import math
import uuid

VERSION = 1
MAX_FRAME = 1024 * 1024
CAPABILITIES = ["full_release", "vram_shares", "cpu_cores", "poll", "recovery", "task_scheduling"]


class ProtocolError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def as_dict(self):
        return {"code": self.code, "message": str(self), "details": self.details}


def require(condition, message, code="invalid_request"):
    if not condition:
        raise ProtocolError(code, message)


def fields(value, required, optional=()):
    require(type(value) is dict, "expected object")
    require(set(required) <= value.keys(), f"missing fields: {set(required) - value.keys()}")
    require(value.keys() <= set(required) | set(optional), "unknown fields")


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, f"expected integer >= {minimum}")
    return value


def identity(value):
    require(type(value) is str, "expected UUID string")
    try:
        require(str(uuid.UUID(value)) == value, "UUID must be canonical")
    except ValueError as exc:
        raise ProtocolError("invalid_request", "invalid UUID") from exc
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, f"duplicate key: {key}", "invalid_json")
        result[key] = value
    return result


def decode(frame, *, max_frame=MAX_FRAME):
    require(len(frame) <= max_frame, "frame exceeds size bound", "frame_too_large")
    try:
        def finite_float(text):
            number = float(text)
            require(math.isfinite(number), "nonfinite number", "invalid_json")
            return number
        value = json.loads(frame, object_pairs_hook=_pairs, parse_float=finite_float,
                           parse_constant=lambda _: require(False, "nonfinite number", "invalid_json"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError("invalid_json", str(exc)) from exc
    require(type(value) is dict, "frame must be an object", "invalid_json")
    return value


def encode(value, *, max_frame=MAX_FRAME):
    data = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    require(len(data) - 1 <= max_frame, "frame exceeds size bound", "frame_too_large")
    return data
