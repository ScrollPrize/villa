"""Remote Zarr v2 arrays. Whole-array copies go through vc_reader
(mirror_array). The chunk-level helpers (objects with retries, JSON metadata
and the chunk indices an anonymous virtual-hosted S3 bucket lists in a z
range) serve fiber_direction_samples, which fetches a subset of the chunks;
local paths read the same way, so callers treat both alike."""

from __future__ import annotations

from pathlib import Path
import json
import re
import time
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET


def is_http(location):
    return urlparse(str(location)).scheme in ("http", "https")


def get_json(session, location):
    if not is_http(location):
        return json.loads(Path(location).read_text())
    response = session.get(location, timeout=60)
    response.raise_for_status()
    return response.json()


def chunk_url(array_url, index, metadata):
    separator = metadata.get("dimension_separator", ".")
    return f"{array_url.rstrip('/')}/{separator.join(map(str, index))}"


def fetch(session, location):
    """The bytes of the object at a URL or local path; None when absent."""
    import requests

    if not is_http(location):
        try:
            return Path(location).read_bytes()
        except FileNotFoundError:
            return None
    for attempt in range(6):
        try:
            response = session.get(location, timeout=60)
            if response.status_code < 500:
                break
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 5:
                raise
        if attempt == 5:
            response.raise_for_status()
        time.sleep(min(2 ** attempt, 8))
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.content


def list_s3_chunk_indices(session, array_url, metadata, first_chunk, last_chunk):
    """Return existing chunk indices for an anonymous virtual-hosted S3 URL."""
    parsed = urlparse(array_url)
    if not parsed.hostname or ".s3." not in parsed.hostname:
        return None
    separator = metadata.get("dimension_separator", ".")
    array_key = unquote(parsed.path.lstrip("/").rstrip("/"))
    result = set()
    for cz in range(int(first_chunk[0]), int(last_chunk[0]) + 1):
        prefix = f"{array_key}/{cz}{separator}"
        continuation = None
        while True:
            params = {"list-type": "2", "prefix": prefix}
            if continuation:
                params["continuation-token"] = continuation
            response = session.get(
                f"{parsed.scheme}://{parsed.netloc}/", params=params, timeout=60)
            if response.status_code in (401, 403):
                return None
            response.raise_for_status()
            root = ET.fromstring(response.content)
            keys = [node.text for node in root.iter()
                    if node.tag.rsplit("}", 1)[-1] == "Key"]
            for key in keys:
                suffix = key[len(array_key) + 1:]
                try:
                    index = tuple(int(value) for value in suffix.split(separator))
                except (TypeError, ValueError):
                    continue
                if (len(index) == 3
                        and all(first_chunk[axis] <= index[axis] <= last_chunk[axis]
                                for axis in range(3))):
                    result.add(index)
            truncated = next((node.text == "true" for node in root.iter()
                              if node.tag.rsplit("}", 1)[-1] == "IsTruncated"), False)
            if not truncated:
                break
            continuation = next((node.text for node in root.iter()
                                 if node.tag.rsplit("}", 1)[-1]
                                 == "NextContinuationToken"), None)
            if not continuation:
                raise ValueError("truncated S3 listing omitted continuation token")
    return result


# ---------------------------------------------------------------------------
# Whole-array copies, through vc_reader
# ---------------------------------------------------------------------------

_MIRRORED = ".vc-mirrored"
_VIRTUAL_HOSTED = re.compile(r"^(?P<bucket>[^./]+)\.s3(?:\.(?P<region>[^./]+))?\.amazonaws\.com$")


def listable_location(location):
    """``location`` as vc_reader lists it (its ``s3://`` or ``s3+REGION://``
    form), or None when its objects cannot be listed: an ``s3://`` or
    ``s3+REGION://`` location as given, a virtual-hosted Amazon S3 URL
    (``https://BUCKET.s3[.REGION].amazonaws.com/KEY``) by bucket and region,
    and nothing else (plain HTTP has no listing)."""
    location = str(location).strip().rstrip("/")
    if location.startswith(("s3://", "s3+")):
        return location
    parsed = urlparse(location)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    match = _VIRTUAL_HOSTED.match(parsed.hostname)
    if not match:
        return None
    scheme = f"s3+{match['region']}" if match["region"] else "s3"
    return f"{scheme}://{match['bucket']}/{unquote(parsed.path).lstrip('/')}"


def is_listable(location):
    """Whether vc_reader can list the objects of the array at ``location``,
    and so copy it whole (listable_location)."""
    return listable_location(location) is not None


def access_for(location):
    """How ``location`` is reached (vc_reader.reader.auth.resolve_access,
    "auto"): anonymous, except an ``s3://`` bucket outside the Open Data
    channels when the environment has AWS credentials. Tests replace it."""
    from vc_reader.reader.auth import resolve_access

    return resolve_access(str(location), "auto")


def mirror_array(array_url, cache_directory, *, workers=16, progress=None, access=None):
    """Copy the Zarr array at ``array_url`` (is_listable) byte for byte into
    vc_reader's cache layout under ``cache_directory`` (``_reader.
    download_prefix``: every object the bucket lists, with no byte budget,
    so nothing evicts it; objects already there are kept, so an interrupted
    copy resumes). Returns the array's directory. ``progress(done, total)``
    is called as objects arrive; ``access`` replaces access_for's."""
    from vc_reader import _reader

    location = listable_location(array_url)
    if location is None:
        raise ValueError(f"{array_url}: its objects cannot be listed; only s3:// arrays and "
                         "virtual-hosted Amazon S3 URLs are fetched")
    access = access or access_for(location)
    directory = Path(_reader.decoded_cache_path(str(cache_directory), access.cache_url(location)))
    if (directory / _MIRRORED).is_file():
        return str(directory)
    workers = max(1, int(workers))
    options = _reader.OpenOptions(access, [str(cache_directory)], True, None, False, False,
                                  None, 1, workers, "mmap", False)

    def report(state):
        if progress is not None and state.status == "downloading":
            progress(state.completed_files, state.total_files)

    result = _reader.download_prefix(location, options, workers=workers, progress=report)
    if result.error:
        raise ValueError(f"{array_url}: {result.error}")
    if not (directory / ".zarray").is_file():
        raise ValueError(f"{array_url}: the fetched objects hold no .zarray (not a Zarr v2 array?)")
    (directory / _MIRRORED).write_text(f"{location}\n")
    return str(directory)
