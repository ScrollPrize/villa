"""Vocabulary shared by every Spiral service component.

The HTTP error type and the safe-path/transfer primitives below are used by
the artifact registry, the upload manager, and the request handler alike, so
they live in one leaf module that none of those import each other through.
"""

from __future__ import annotations

import hashlib
from http import HTTPStatus
from pathlib import Path
import re


TRANSFER_CHUNK_BYTES = 1024 * 1024

SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@ -]{0,127}$")


class ApiError(Exception):
    def __init__(self, status, message, details=None, payload=None):
        super().__init__(message)
        self.status = int(status)
        self.message = message
        self.details = details
        # Extra fields merged into the error response body, for refusals that
        # carry structured facts a client acts on rather than only displays
        # (see ServiceState.load_checkpoint's stage/reasons/refused).
        self.payload = dict(payload or {})


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(TRANSFER_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def is_safe_relative_name(name):
    """Accept forward-slash relative names made of safe components only."""
    if not isinstance(name, str) or not name or len(name) > 1024:
        return False
    if "\\" in name or name.startswith("/"):
        return False
    parts = name.split("/")
    if len(parts) > 8:
        return False
    for part in parts:
        if part in ("", ".", "..") or not SAFE_COMPONENT.match(part):
            return False
    return True


def resolve_inside(root, relative_name):
    """Resolve ``relative_name`` under ``root`` refusing symlink/`..` escapes."""
    root = Path(root).resolve(strict=True)
    candidate = (root / relative_name).resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise ApiError(HTTPStatus.FORBIDDEN, "Path escapes the artifact root")
    if candidate.is_symlink() or not candidate.is_file():
        raise ApiError(HTTPStatus.FORBIDDEN, "Not a regular file")
    return candidate
