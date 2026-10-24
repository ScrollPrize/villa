"""Download and cache metadata.json.

The object is stored gzip-compressed on S3 and served with
``Content-Encoding: gzip`` regardless of what the client asks for, so the body
is decompressed here when it carries the gzip magic bytes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
import time
import urllib.error
import urllib.request
import warnings
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


def decode_json_bytes(data: bytes) -> Any:
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    return json.loads(data)


def read_metadata_file(path: str | os.PathLike[str]) -> Any:
    return decode_json_bytes(Path(path).read_bytes())


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def fetch_metadata(
    url: str = DEFAULT_METADATA_URL,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    use_cache: bool = True,
    max_age: float = 3600.0,
    refresh: bool = False,
    timeout: float = 60.0,
) -> Any:
    """Return the parsed JSON at ``url``, using an on-disk cache.

    A cached copy younger than ``max_age`` seconds is used without contacting
    the server. Otherwise (or with ``refresh=True``) the server is asked with
    ``If-None-Match``; a 304 reuses the cache. If the request fails and a cached
    copy exists, the cached copy is used and a warning is emitted.
    """
    if not use_cache:
        body, _ = _get(url, {}, timeout)
        return decode_json_bytes(body)

    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    stem = "metadata-" + hashlib.sha256(url.encode()).hexdigest()[:16]
    body_path, info_path = cache / (stem + ".bin"), cache / (stem + ".info.json")

    info: dict[str, Any] = {}
    if body_path.exists() and info_path.exists():
        try:
            info = json.loads(info_path.read_text())
        except (OSError, ValueError):
            info = {}
    have_cache = bool(info)

    if have_cache and not refresh and time.time() - float(info.get("checked_at", 0)) < max_age:
        return decode_json_bytes(body_path.read_bytes())

    headers = {}
    if have_cache and info.get("etag"):
        headers["If-None-Match"] = info["etag"]
    try:
        body, resp_headers = _get(url, headers, timeout)
    except _NotModified:
        info["checked_at"] = time.time()
        _atomic_write(info_path, json.dumps(info).encode())
        return decode_json_bytes(body_path.read_bytes())
    except (urllib.error.URLError, OSError) as exc:
        if not have_cache:
            raise
        warnings.warn(f"could not refresh {url} ({exc}); using cached copy from {body_path}", stacklevel=2)
        return decode_json_bytes(body_path.read_bytes())

    parsed = decode_json_bytes(body)
    _atomic_write(body_path, body)
    info = {
        "url": url,
        "etag": resp_headers.get("ETag"),
        "last_modified": resp_headers.get("Last-Modified"),
        "checked_at": time.time(),
    }
    _atomic_write(info_path, json.dumps(info).encode())
    return parsed


class _NotModified(Exception):
    pass


def _get(url: str, headers: dict[str, str], timeout: float) -> tuple[bytes, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.headers
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            raise _NotModified() from None
        raise
