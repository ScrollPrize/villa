from __future__ import annotations

from pathlib import Path

from .catalog import VolumeRecord
from .config import ManagerConfig


PREFETCH_REQUEST_VERSION = 1


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
    if not volume.s3_url:
        raise ValueError(f"volume {volume.selector!r} has no supported S3 origin")
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
    if not volume.s3_url:
        raise ValueError(f"volume {volume.selector!r} has no supported S3 origin")
    return {
        "version": PREFETCH_REQUEST_VERSION,
        "source": volume.s3_url,
        "destination": str(destination),
        "scale": int(scale),
        "workers": int(workers),
        "anon": True,
        "remote_inventory": bool(remote_inventory),
    }


def execute_prefetch_request(request: dict[str, object]) -> Path:
    if request.get("version") != PREFETCH_REQUEST_VERSION:
        raise ValueError(f"unsupported prefetch request version: {request.get('version')!r}")
    source = request.get("source")
    destination_value = request.get("destination")
    scale = request.get("scale")
    workers = request.get("workers")
    anon = request.get("anon")
    remote_inventory = request.get("remote_inventory")
    if not isinstance(source, str) or not source.startswith("s3://"):
        raise ValueError("prefetch source must be an S3 URL")
    if not isinstance(destination_value, str) or not destination_value:
        raise ValueError("prefetch destination must be a non-empty path")
    if not isinstance(scale, int) or isinstance(scale, bool) or scale < 0:
        raise ValueError("prefetch scale must be a non-negative integer")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("prefetch workers must be a positive integer")
    if anon is not True:
        raise ValueError("managed open-data prefetch must use anonymous S3 access")
    if not isinstance(remote_inventory, bool):
        raise ValueError("prefetch remote_inventory must be boolean")
    destination = Path(destination_value)
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
