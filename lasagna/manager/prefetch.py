from __future__ import annotations

from pathlib import Path

from .catalog import VolumeRecord
from .config import ManagerConfig


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
    from lasagna.scripts.download_omezarr import download

    result = download(
        source=volume.s3_url,
        dest=str(destination),
        scales=[scale],
        workers=workers,
        anon=True,
        remote_inventory=remote_inventory,
    )
    if result != 0:
        raise RuntimeError(f"volume download failed with exit status {result}")
    return destination / str(scale)
