"""Download and cache metadata.json.

The object is stored gzip-compressed on S3 and served with
``Content-Encoding: gzip`` regardless of what the client asks for, so the body
is decompressed here when it carries the gzip magic bytes.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_METADATA_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json"


def default_cache_dir() -> Path:
    env = os.environ.get("VC_READER_CACHE_DIR")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "vc_reader"

