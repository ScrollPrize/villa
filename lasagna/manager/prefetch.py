from __future__ import annotations

from pathlib import Path
from typing import Any

from .catalog import VolumeRecord
from .config import ManagerConfig

PREFETCH_REQUEST_VERSION = 2


def input_cache_dir(config: ManagerConfig) -> Path:
    """The vc_reader disk cache every managed inference reads its input through."""
    cache = config.resolved_path("cache_dir", required=True)
    assert cache is not None
    return cache / "volumes"


def volume_input_location(volume: VolumeRecord, scale: int) -> str:
    """The remote OME-Zarr level inference opens: ``<s3 volume url>/<scale>``."""
    if scale < 0:
        raise ValueError("scale must be a non-negative OME-Zarr group index")
    if not volume.s3_url:
        raise ValueError(f"volume {volume.selector!r} has no supported S3 origin")
    return f"{volume.s3_url.rstrip('/')}/{int(scale)}"


def volume_cache_root(config: ManagerConfig, volume: VolumeRecord) -> Path:
    """Where vc_reader keeps this volume's cached chunks: ``<cache>/s3/<bucket>/<key>``."""
    if not volume.s3_url:
        raise ValueError(f"volume {volume.selector!r} has no supported S3 origin")
    bucket, _, key = volume.s3_url.removeprefix("s3://").strip("/").partition("/")
    return input_cache_dir(config) / "s3" / bucket / key


def prefetch_volume(
    config: ManagerConfig,
    volume: VolumeRecord,
    scale: int,
    *,
    connections: int = 64,
) -> Path:
    request = build_prefetch_request(config, volume, scale, connections=connections)
    return execute_prefetch_request(request)


def build_prefetch_request(
    config: ManagerConfig,
    volume: VolumeRecord,
    scale: int,
    *,
    connections: int = 64,
) -> dict[str, Any]:
    if connections <= 0:
        raise ValueError("connections must be a positive integer")
    return {
        "version": PREFETCH_REQUEST_VERSION,
        "location": volume_input_location(volume, scale),
        "cache_dir": str(input_cache_dir(config)),
        "cached_level": str(volume_cache_root(config, volume) / str(int(scale))),
        "connections": int(connections),
    }


def execute_prefetch_request(request: dict[str, Any]) -> Path:
    """Fill the vc_reader disk cache with every chunk of the requested level."""
    if request.get("version") != PREFETCH_REQUEST_VERSION:
        raise ValueError(f"unsupported prefetch request version: {request.get('version')!r}")
    location = request.get("location")
    cache_dir = request.get("cache_dir")
    cached_level = request.get("cached_level")
    connections = request.get("connections")
    if not isinstance(location, str) or not location.startswith("s3://"):
        raise ValueError("prefetch location must be an S3 URL")
    if not isinstance(cache_dir, str) or not cache_dir:
        raise ValueError("prefetch cache_dir must be a non-empty path")
    if not isinstance(cached_level, str) or not cached_level:
        raise ValueError("prefetch cached_level must be a non-empty path")
    if not isinstance(connections, int) or isinstance(connections, bool) or connections <= 0:
        raise ValueError("prefetch connections must be a positive integer")
    from lasagna.scripts.prefill_omezarr import prefill_level

    last = [-1]

    def progress(done: int, total: int) -> None:
        if total and (done == total or done // max(1, total // 100) != last[0]):
            last[0] = done // max(1, total // 100)
            print(f"[las_manager] prefetch {done}/{total} chunks", flush=True)

    result = prefill_level(location, cache_dir, anon=True, connections=connections, progress=progress)
    print(
        f"[las_manager] prefetch fetched {result.fetched_chunks} chunks, "
        f"{result.empty_chunks} empty, of {result.total_chunks}",
        flush=True,
    )
    return Path(cached_level)
