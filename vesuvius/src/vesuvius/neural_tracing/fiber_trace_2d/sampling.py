from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vesuvius.neural_tracing.fiber_trace_2d.loader_support import (
    ZarrChunkRequest,
    chunk_requests_for_coords,
    sample_array_trilinear,
)


_REMOTE_PREFIXES = ("http://", "https://", "s3://")


@dataclass(frozen=True)
class CoordinateSampleResult:
    image: np.ndarray
    valid_mask: np.ndarray
    stats: dict[str, Any]


class CoordinateSampler:
    def sample_coords(self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray) -> CoordinateSampleResult:
        raise NotImplementedError

    def sample_coord_batch(
        self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray
    ) -> CoordinateSampleResult:
        return _sample_coord_batch_flattened(self, coords_zyx_base, valid_mask)

    def prefetch_coords(self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
        result = self.sample_coords(coords_zyx_base, valid_mask)
        stats = dict(result.stats)
        stats.setdefault("prefetch_mode", "sample_coords")
        stats.setdefault("valid_pixels", int(np.count_nonzero(valid_mask)))
        return stats

    def chunk_requests_for_coords(
        self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray
    ) -> list[ZarrChunkRequest]:
        raise NotImplementedError


def _sample_coord_batch_flattened(
    sampler: CoordinateSampler,
    coords_zyx_base: np.ndarray,
    valid_mask: np.ndarray,
) -> CoordinateSampleResult:
    coords = np.asarray(coords_zyx_base, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError("coords_zyx_base must have shape B,H,W,3")
    if valid.shape != coords.shape[:3]:
        raise ValueError("valid_mask must have shape B,H,W")
    batch, height, width = (int(v) for v in coords.shape[:3])
    flat_coords = np.ascontiguousarray(coords.reshape(batch * height, width, 3))
    flat_valid = np.ascontiguousarray(valid.reshape(batch * height, width))
    result = sampler.sample_coords(flat_coords, flat_valid)
    image = np.asarray(result.image, dtype=np.float32)
    sampled_valid = np.asarray(result.valid_mask, dtype=bool)
    if image.shape != (batch * height, width):
        raise ValueError(
            "batched coordinate sampler returned incompatible image shape: "
            f"shape={image.shape} expected={(batch * height, width)}"
        )
    if sampled_valid.shape != (batch * height, width):
        raise ValueError(
            "batched coordinate sampler returned incompatible valid-mask shape: "
            f"shape={sampled_valid.shape} expected={(batch * height, width)}"
        )
    stats = dict(result.stats)
    stats["batch_patches"] = batch
    stats["batch_mode_flattened"] = 1
    return CoordinateSampleResult(
        image=image.reshape(batch, height, width),
        valid_mask=sampled_valid.reshape(batch, height, width),
        stats=stats,
    )


def _coords_zyx_base_to_xyz(coords_zyx_base: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords_zyx_base, dtype=np.float32)
    return np.ascontiguousarray(coords[..., (2, 1, 0)])


class Vc3dCoordinateSampler(CoordinateSampler):
    def __init__(
        self,
        volume_path: str,
        *,
        level: int,
        cache_root: str | None,
        sampling: str = "trilinear",
        blocking: bool = True,
    ) -> None:
        from vc.volume import Volume

        self.level = int(level)
        self.sampling = str(sampling)
        self.blocking = bool(blocking)
        path = str(volume_path)
        if path.startswith("s3://"):
            # VC3D remote volume loading is HTTP-backed. The public Vesuvius S3
            # zarrs are also exposed through the standard virtual-hosted URL.
            without_scheme = path[len("s3://") :]
            bucket, _, key = without_scheme.partition("/")
            path = f"https://{bucket}.s3.amazonaws.com/{key}"
        if path.startswith(_REMOTE_PREFIXES):
            self.volume = Volume.open_url(path, "" if cache_root is None else str(cache_root))
        else:
            self.volume = Volume.open(path)

    @property
    def store_identity(self) -> str:
        volume_id = getattr(self.volume, "remote_url", "") or getattr(self.volume, "path", "")
        return f"vc3d:{volume_id}:level={self.level}"

    def sample_coords(self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray) -> CoordinateSampleResult:
        image, sampled_valid, stats = self.volume.sample_coords(
            _coords_zyx_base_to_xyz(coords_zyx_base),
            np.ascontiguousarray(valid_mask, dtype=np.bool_),
            self.level,
            self.sampling,
            32,
            self.blocking,
        )
        image_arr = np.asarray(image, dtype=np.float32)
        valid_arr = np.asarray(sampled_valid, dtype=np.uint8)
        if image_arr.ndim == 3 and image_arr.shape[-1] == 1:
            image_arr = image_arr[..., 0]
        if valid_arr.ndim == 3 and valid_arr.shape[-1] == 1:
            valid_arr = valid_arr[..., 0]
        return CoordinateSampleResult(
            image=image_arr.astype(np.float32, copy=False),
            valid_mask=valid_arr.astype(bool, copy=False),
            stats=dict(stats),
        )

    def chunk_requests_for_coords(
        self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray
    ) -> list[ZarrChunkRequest]:
        dependencies = self.volume.collect_coords_dependencies(
            _coords_zyx_base_to_xyz(coords_zyx_base),
            np.ascontiguousarray(valid_mask, dtype=np.bool_),
            self.level,
            self.sampling,
            32,
        )
        store = _Vc3dChunkStore(self.volume)
        requests: list[ZarrChunkRequest] = []
        for dependency in dependencies:
            if isinstance(dependency, dict):
                def _path_or_none(name: str) -> Path | None:
                    value = str(dependency.get(name, "") or "")
                    return Path(value) if value else None

                key = str(dependency.get("key", ""))
                if not key:
                    level = int(dependency["level"])
                    iz = int(dependency["iz"])
                    iy = int(dependency["iy"])
                    ix = int(dependency["ix"])
                    key = f"{level}/{iz}/{iy}/{ix}"
                requests.append(
                    ZarrChunkRequest(
                        store=store,
                        store_identity=self.store_identity,
                        key=key,
                        cache_path=_path_or_none("cache_path"),
                        empty_path=_path_or_none("empty_path"),
                        remote_url=str(dependency.get("remote_url", "")) or None,
                        remote_chunk_key=str(dependency.get("remote_chunk_key", "")) or None,
                        cache_payload_format=str(dependency.get("cache_payload_format", "")) or None,
                        persistent_extension=str(dependency.get("persistent_extension", "")) or None,
                    )
                )
                continue
            raise RuntimeError(
                "VC3D collect_coords_dependencies returned legacy tuple dependencies. "
                "Rebuild/use the updated volume-cartographer Python bindings so chunk "
                "metadata includes remote_url, cache_path, empty_path, and cache_payload_format."
            )
        return requests


class _Vc3dChunkStore:
    def __init__(self, volume: Any) -> None:
        self.volume = volume


class NumpyZarrCoordinateSampler(CoordinateSampler):
    def __init__(self, array: Any, *, level_spacing_base: float) -> None:
        self.array = array
        self.level_spacing_base = float(level_spacing_base)

    def sample_coords(self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray) -> CoordinateSampleResult:
        coords_zyx_level = np.asarray(coords_zyx_base, dtype=np.float32) / self.level_spacing_base
        image = sample_array_trilinear(self.array, coords_zyx_level, valid_mask)
        return CoordinateSampleResult(
            image=image,
            valid_mask=np.asarray(valid_mask, dtype=bool),
            stats={"covered_pixels": int(np.count_nonzero(valid_mask)), "requested_chunks": 0, "error_chunks": 0},
        )

    def chunk_requests_for_coords(
        self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray
    ) -> list[ZarrChunkRequest]:
        coords_zyx_level = np.asarray(coords_zyx_base, dtype=np.float32) / self.level_spacing_base
        return chunk_requests_for_coords(self.array, coords_zyx_level, valid_mask)

    def prefetch_coords(self, coords_zyx_base: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
        requests = self.chunk_requests_for_coords(coords_zyx_base, valid_mask)
        unique: dict[tuple[str, str], ZarrChunkRequest] = {
            (request.store_identity, request.key): request for request in requests
        }
        return {
            "prefetch_mode": "chunk_requests",
            "generated": len(requests),
            "unique_chunks": len(unique),
            "downloaded": 0,
            "bytes": 0,
            "errors": 0,
            "valid_pixels": int(np.count_nonzero(valid_mask)),
        }


def make_coordinate_sampler(
    *,
    volume_path: str,
    array: Any,
    level: int,
    level_spacing_base: float,
    cache_root: str | None,
) -> CoordinateSampler:
    del array, level_spacing_base
    return Vc3dCoordinateSampler(volume_path, level=level, cache_root=cache_root)
