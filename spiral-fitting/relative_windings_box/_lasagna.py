"""Fetch a published *.lasagna.json manifest and check it the way lasagna/lasagna_volume.py's LasagnaVolume.load
reads it."""

import http.client
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from . import _zarr


REQUIRED_CHANNELS = ("grad_mag", "nx", "ny")


MAX_LEVEL = 32                           # the deepest OME-Zarr level (scaledown) a manifest may name


class LasagnaError(Exception):
    pass


def https(url):
    url = str(url)
    return _zarr.s3_https(url) if url.lower().startswith("s3://") else url


def http_get(url, timeout=60, opener=None):
    """GET with S3's false 404s retried; a server on this machine is reached without the proxy."""
    url = https(url)
    req = urllib.request.Request(url, headers={"User-Agent": _zarr.USER_AGENT})
    try:
        with _zarr.open_url(req, opener or service_opener(url).open, timeout=timeout) as resp:
            return resp.read()
    except http.client.HTTPException as exc:   # a transfer cut off part way (IncompleteRead) is a failed download
        raise OSError(f"{type(exc).__name__}: {exc}") from exc


def fetch(url, what, opener=None):
    """http_get with every failure as a LasagnaError naming what was fetched."""
    try:
        return http_get(url, opener=opener)
    except urllib.error.HTTPError as exc:
        raise LasagnaError(f"{what} {https(url)}: HTTP {exc.code} {exc.reason}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise LasagnaError(f"{what} {https(url)}: {getattr(exc, 'reason', exc)}") from exc


def fetch_json(url, what, opener=None):
    try:
        return json.loads(fetch(url, what, opener))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LasagnaError(f"{what} {https(url)} is not JSON: {exc}") from exc


def list_prefix(url, opener=None):
    """Keys under an S3 prefix (anonymous ListObjectsV2, one level, with continuation)."""
    endpoint, prefix = _zarr.s3_listing(https(url).rstrip("/") + "/")
    if endpoint is None:
        raise LasagnaError(f"{url} is not an S3 URL; give the .lasagna.json URL itself")
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "delimiter": "/"}
        if token:
            q["continuation-token"] = token
        body = fetch(f"{endpoint}/?{urllib.parse.urlencode(q)}", "listing", opener).decode("utf-8", "replace")
        keys += re.findall(r"<Key>([^<]+)</Key>", body)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", body)
        if not m:
            return endpoint, prefix, keys
        token = m.group(1)


def remote_manifest(source, opener=None):
    """(run folder URL ending in /, manifest file name, manifest dict) for a published run folder or manifest URL."""
    src = https(source)
    if src.endswith(".lasagna.json"):
        base, name = src.rsplit("/", 1)
        return base + "/", name, fetch_json(src, "manifest", opener)
    endpoint, prefix, keys = list_prefix(src, opener)
    names = [k[len(prefix):] for k in keys if k.endswith(".lasagna.json") and "/" not in k[len(prefix):]]
    if len(names) != 1:
        raise LasagnaError(f"{source}: expected one .lasagna.json in the folder, found {names or 'none'}")
    base = f"{endpoint}/{prefix}"
    return base, names[0], fetch_json(base + names[0], "manifest", opener)


def check_manifest_doc(doc):
    """Problems LasagnaVolume.load / fit_data would raise on, without reading any zarr."""
    if not isinstance(doc, dict):
        return ["the manifest is not a JSON object"]
    probs = []
    groups = doc.get("groups")
    if not isinstance(groups, dict) or not groups:
        return ["no groups"]
    chans = {}
    for g, spec in groups.items():
        if not isinstance(spec, dict) or not {"zarr", "scaledown", "channels"} <= set(spec):
            probs.append(f"group {g}: needs zarr, scaledown and channels")
            continue
        rel = str(spec["zarr"])
        if rel.startswith("/") or ".." in Path(rel).parts or re.match(r"^[a-z]+://", rel, re.I):
            probs.append(f"group {g}: zarr {rel!r} must be a relative path inside the manifest's folder")
        if (isinstance(spec["scaledown"], bool) or not isinstance(spec["scaledown"], int)
                or not 0 <= spec["scaledown"] <= MAX_LEVEL):
            probs.append(f"group {g}: scaledown {spec['scaledown']!r} is not an OME-Zarr level (0 to {MAX_LEVEL})")
        if not isinstance(spec["channels"], list):
            probs.append(f"group {g}: channels is not a list")
            continue
        for c in spec["channels"]:
            chans.setdefault(c, g)
    miss = [c for c in REQUIRED_CHANNELS if c not in chans]
    if miss:
        probs.append(f"channels {miss} missing (fit.py needs grad_mag, nx and ny; cos for the data terms)")
    shape = doc.get("base_shape_zyx")
    if shape is not None and not (isinstance(shape, list) and len(shape) == 3
                                  and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in shape)):
        probs.append(f"base_shape_zyx {shape!r} is not three positive integers")
    s2b = doc.get("source_to_base", 1.0)
    if isinstance(s2b, bool) or not isinstance(s2b, (int, float)) or not (math.isfinite(s2b) and s2b > 0):
        probs.append(f"source_to_base {s2b!r} is not a positive number")
    elif not probs:
        for g, spec in groups.items():
            if level_spacing(doc, spec) < 1:
                probs.append(f"group {g}: level {spec['scaledown']} with source_to_base {s2b} is under one base "
                             "voxel per voxel")
    return probs


def level_spacing(doc, spec):
    """Base voxels per voxel of a group's level array: 2**scaledown x source_to_base, as lasagna rounds it."""
    return int(round(2 ** int(spec["scaledown"]) * float(doc.get("source_to_base", 1.0))))


def service_opener(url):
    """No proxy for a server on this machine."""
    host = urllib.parse.urlsplit(url).hostname or ""
    if host in ("127.0.0.1", "localhost", "::1") or host.startswith("127."):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()
