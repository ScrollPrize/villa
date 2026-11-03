"""Reading Zarr v2 arrays over HTTP(S): objects with retries, JSON
metadata, and the chunk indices an anonymous virtual-hosted S3 bucket lists.
Local paths read the same way, so callers treat both alike."""

from __future__ import annotations

from pathlib import Path
import json
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


_MIRRORED = ".vc-mirrored"


def mirror_array(array_url, directory, *, workers=16, progress=None, session_factory=None):
    """Copy the Zarr v2 array at ``array_url`` (an anonymous virtual-hosted
    S3 URL) into the directory Zarr ``directory``, byte for byte: its
    metadata and every chunk its bucket lists. Interrupted copies resume:
    chunks already written whole are kept. ``progress(done, total)`` is
    called as chunks arrive. Returns ``directory``."""
    from concurrent.futures import ThreadPoolExecutor
    import threading

    import requests

    session_factory = session_factory or requests.Session
    directory = Path(directory)
    if (directory / _MIRRORED).is_file():
        return str(directory)
    url = str(array_url).rstrip("/")
    with session_factory() as session:
        metadata = get_json(session, f"{url}/.zarray")
        shape, chunks = metadata["shape"], metadata["chunks"]
        last = tuple(-(-int(s) // int(c)) - 1 for s, c in zip(shape, chunks))
        indices = list_s3_chunk_indices(session, url, metadata, (0, 0, 0), last)
        if indices is None:
            raise ValueError(f"{url}: its chunks cannot be listed; only arrays in an "
                             "anonymous virtual-hosted S3 bucket are fetched")
        directory.mkdir(parents=True, exist_ok=True)
        for name in (".zarray", ".zattrs"):
            content = fetch(session, f"{url}/{name}")
            if content is not None:
                _write_whole(directory / name, content)
    separator = metadata.get("dimension_separator", ".")
    missing = [index for index in sorted(indices)
               if not (directory / separator.join(map(str, index))).is_file()]
    done = len(indices) - len(missing)
    lock = threading.Lock()
    local = threading.local()
    sessions = []
    if progress is not None:
        progress(done, len(indices))

    def copy(index):
        nonlocal done
        session = getattr(local, "session", None)
        if session is None:
            session = local.session = session_factory()
            with lock:
                sessions.append(session)
        content = fetch(session, chunk_url(url, index, metadata))
        if content is None:
            raise ValueError(f"{url}: listed chunk {index} is missing")
        _write_whole(directory / separator.join(map(str, index)), content)
        with lock:
            done += 1
            if progress is not None:
                progress(done, len(indices))

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            for _ in pool.map(copy, missing):
                pass
    finally:
        for session in sessions:
            session.close()
    (directory / _MIRRORED).write_text(f"{url}\n")
    return str(directory)


def _write_whole(path, content):
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes(content)
    os.replace(temporary, path)
