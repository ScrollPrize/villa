from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .catalog import VolumeRecord
from .config import ManagerConfig


PREFETCH_REQUEST_VERSION = 1
_HTTP_USER_AGENT = "las_manager/0.1"
_HTTP_TIMEOUT_SECONDS = 30.0
_HTTP_RETRIES = 4
_HTTP_TRANSIENT_STATUS = {408, 425, 429, 500, 502, 503, 504}
_HTTP_FUTURES_PER_WORKER = 2


def volume_cache_root(config: ManagerConfig, volume: VolumeRecord) -> Path:
    cache = config.resolved_path("cache_dir", required=True)
    assert cache is not None
    return cache / "volumes" / volume.sample_id / volume.long_id


def prefetch_volume(
    config: ManagerConfig,
    volume: VolumeRecord,
    scale: int,
    *,
    workers: int = 64,
    remote_inventory: bool = True,
) -> Path:
    if scale < 0:
        raise ValueError("scale must be a non-negative OME-Zarr group index")
    if workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not volume.prefetch_url:
        raise ValueError(f"volume {volume.selector!r} has no supported public prefetch origin")
    destination = volume_cache_root(config, volume)
    request = build_prefetch_request(
        volume, destination, scale, workers=workers,
        remote_inventory=remote_inventory,
    )
    return execute_prefetch_request(request)


def build_prefetch_request(
    volume: VolumeRecord,
    destination: Path,
    scale: int,
    *,
    workers: int = 64,
    remote_inventory: bool = True,
) -> dict[str, object]:
    if scale < 0:
        raise ValueError("scale must be a non-negative OME-Zarr group index")
    if workers <= 0:
        raise ValueError("workers must be a positive integer")
    if not volume.prefetch_url:
        raise ValueError(f"volume {volume.selector!r} has no supported public prefetch origin")
    return {
        "version": PREFETCH_REQUEST_VERSION,
        "source": volume.prefetch_url,
        "destination": str(destination),
        "scale": int(scale),
        "workers": int(workers),
        "anon": True,
        "remote_inventory": bool(remote_inventory),
    }


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _http_join(root: str, relative: str) -> str:
    return root.rstrip("/") + "/" + relative.lstrip("/")


def _http_read_bytes(url: str, *, allow_missing: bool = False) -> bytes | None:
    last_error: BaseException | None = None
    for attempt in range(_HTTP_RETRIES):
        request = Request(
            url,
            headers={
                "User-Agent": _HTTP_USER_AGENT,
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:
                return response.read()
        except HTTPError as error:
            if allow_missing and error.code == 404:
                return None
            last_error = error
            if error.code not in _HTTP_TRANSIENT_STATUS or attempt == _HTTP_RETRIES - 1:
                raise
        except (URLError, TimeoutError, ConnectionError, OSError) as error:
            last_error = error
            if attempt == _HTTP_RETRIES - 1:
                raise
        time.sleep(0.5 * (2 ** attempt))
    assert last_error is not None
    raise last_error


def _ensure_http_file(
    source_root: str,
    destination: Path,
    relative: str,
    *,
    required: bool,
) -> bytes | None:
    local = destination.joinpath(*relative.split("/"))
    if local.is_file():
        return local.read_bytes()
    raw = _http_read_bytes(_http_join(source_root, relative), allow_missing=not required)
    if raw is None:
        return None
    _atomic_bytes(local, raw)
    return raw


def _chunk_layout(zarray: dict[str, object]) -> tuple[tuple[int, ...], str]:
    shape = zarray.get("shape")
    chunks = zarray.get("chunks")
    if (
        not isinstance(shape, list)
        or not isinstance(chunks, list)
        or len(shape) != len(chunks)
        or not shape
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape)
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in chunks)
    ):
        raise ValueError(
            f"unsupported Zarr v2 geometry: shape={shape!r}, chunks={chunks!r}"
        )
    separator = zarray.get("dimension_separator", ".")
    if separator not in {".", "/"}:
        raise ValueError(f"unsupported Zarr v2 dimension_separator: {separator!r}")
    counts = tuple((size + chunk - 1) // chunk for size, chunk in zip(shape, chunks))
    return counts, str(separator)


def _iter_chunk_keys(counts: tuple[int, ...], separator: str):
    for coordinates in itertools.product(*(range(count) for count in counts)):
        yield separator.join(str(index) for index in coordinates)


def _download_http_chunk(
    source_root: str,
    destination: Path,
    scale: int,
    key: str,
) -> tuple[str, int]:
    relative = f"{scale}/{key}"
    local = destination / str(scale)
    if "/" in key:
        local = local.joinpath(*key.split("/"))
    else:
        local = local / key
    if local.is_file():
        return "cached", local.stat().st_size
    raw = _http_read_bytes(_http_join(source_root, relative), allow_missing=True)
    if raw is None:
        return "missing", 0
    _atomic_bytes(local, raw)
    return "downloaded", len(raw)


def _download_http_omezarr_group(
    *,
    source: str,
    destination: Path,
    scale: int,
    workers: int,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)

    for relative in (".zgroup", ".zattrs", f"{scale}/.zattrs"):
        _ensure_http_file(source, destination, relative, required=False)

    zarray_raw = _ensure_http_file(
        source,
        destination,
        f"{scale}/.zarray",
        required=True,
    )
    assert zarray_raw is not None
    try:
        zarray = json.loads(zarray_raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"remote OME-Zarr level {scale} has invalid .zarray JSON") from error
    if not isinstance(zarray, dict):
        raise ValueError(f"remote OME-Zarr level {scale} .zarray must contain an object")

    counts_by_axis, separator = _chunk_layout(zarray)
    total = math.prod(counts_by_axis)
    counts = {"cached": 0, "downloaded": 0, "missing": 0}
    bytes_downloaded = 0
    chunk_keys = iter(_iter_chunk_keys(counts_by_axis, separator))
    max_pending = max(1, workers * _HTTP_FUTURES_PER_WORKER)

    completed = 0
    progress_step = max(1, min(1000, total // 100 if total >= 100 else total))

    def record(future) -> None:
        nonlocal bytes_downloaded, completed
        status, size = future.result()
        counts[status] += 1
        if status == "downloaded":
            bytes_downloaded += size
        completed += 1
        if completed == total or completed == 1 or completed % progress_step == 0:
            print(
                "[las_manager] HTTP OME-Zarr prefetch progress "
                f"{completed}/{total}",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    key = next(chunk_keys)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(
                    pool.submit(
                        _download_http_chunk,
                        source,
                        destination,
                        scale,
                        key,
                    )
                )
            if not pending:
                continue
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                record(future)

    print(
        "[las_manager] HTTP OME-Zarr prefetch "
        f"level={scale} chunks={total} cached={counts['cached']} "
        f"downloaded={counts['downloaded']} missing={counts['missing']} "
        f"bytes={bytes_downloaded}",
        flush=True,
    )
    return destination / str(scale)


def execute_prefetch_request(request: dict[str, object]) -> Path:
    if request.get("version") != PREFETCH_REQUEST_VERSION:
        raise ValueError(f"unsupported prefetch request version: {request.get('version')!r}")
    source = request.get("source")
    destination_value = request.get("destination")
    scale = request.get("scale")
    workers = request.get("workers")
    anon = request.get("anon")
    remote_inventory = request.get("remote_inventory")
    if not isinstance(source, str):
        raise ValueError("prefetch source must be a URL")
    scheme = urlparse(source).scheme.lower()
    if scheme not in {"s3", "http", "https"}:
        raise ValueError("prefetch source must be an S3 or HTTP(S) URL")
    if not isinstance(destination_value, str) or not destination_value:
        raise ValueError("prefetch destination must be a non-empty path")
    if not isinstance(scale, int) or isinstance(scale, bool) or scale < 0:
        raise ValueError("prefetch scale must be a non-negative integer")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("prefetch workers must be a positive integer")
    if anon is not True:
        raise ValueError("managed open-data prefetch must use anonymous access")
    if not isinstance(remote_inventory, bool):
        raise ValueError("prefetch remote_inventory must be boolean")
    destination = Path(destination_value)

    if scheme == "s3":
        from lasagna.scripts.download_omezarr import download

        result = download(
            source=source,
            dest=str(destination),
            scales=[scale],
            workers=workers,
            anon=anon,
            remote_inventory=remote_inventory,
        )
        if result != 0:
            raise RuntimeError(f"volume download failed with exit status {result}")
        return destination / str(scale)

    return _download_http_omezarr_group(
        source=source,
        destination=destination,
        scale=scale,
        workers=workers,
    )
