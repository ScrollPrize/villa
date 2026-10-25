"""Download and cache metadata.json.

The object is stored gzip-compressed on S3 and served with
``Content-Encoding: gzip`` regardless of what the client asks for, so the body
is decompressed here when it carries the gzip magic bytes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_METADATA_URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/metadata.json"
USER_AGENT = "vc_reader (https://github.com/ScrollPrize/villa)"


def default_cache_dir() -> Path:
    env = os.environ.get("VC_READER_CACHE_DIR")
    if env:
        return Path(env)
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "vc_reader"


def read_metadata_file(path: str | os.PathLike[str]) -> Any:
    from . import _catalog
    return _catalog.read_file(os.fspath(path))


def fetch_metadata(url: str = DEFAULT_METADATA_URL, *, cache_dir=None,
                   use_cache: bool = True, max_age: float = 3600.0,
                   refresh: bool = False, timeout: float = 60.0) -> Any:
    from .catalog import Catalog
    return Catalog.load(url, cache_dir=cache_dir, use_cache=use_cache,
                        max_age=max_age, refresh=refresh, timeout=timeout).raw
