"""Zarr opening through vc_reader, resolution selection, and padded crops."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vc_reader.reader import load_basic_auth, open_node


_PUBLIC_S3_VOLUME_SUBSTRING = "vesuvius-challenge-open-data"


def open_volume_root(
    path: str | Path,
    auth_json_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    cache_max_gb: float | None = None,
):
    """Open a Zarr root with vc_reader's remote transport and optional disk cache."""
    path_text = str(path)
    auth: Any = "auto"
    if auth_json_path is not None and path_text.startswith("https://") and _PUBLIC_S3_VOLUME_SUBSTRING not in path_text:
        auth = load_basic_auth(auth_json_path)
    if cache_max_gb is not None and cache_max_gb < 0:
        raise ValueError("cache_max_gb must be nonnegative or None")
    return open_node(
        path_text,
        auth=auth,
        cache_dir=None if cache_dir is None else str(cache_dir),
        cache_max_bytes=None if cache_max_gb is None else int(float(cache_max_gb) * 1e9),
    )


def select_volume_level(
    root: Any,
    resolution: int | str,
    *,
    source: str,
    root_array_is_requested_level: bool = False,
) -> Any:
    """Select one resolution from an already opened array or group root."""

    if hasattr(root, "shape"):
        if not root_array_is_requested_level and str(resolution) not in {"0", ""}:
            raise FileNotFoundError(
                f"{source.rstrip('/')}/{resolution} (resolution {str(resolution)!r} "
                f"in zarr array {source!r})"
            )
        return root
    try:
        return root[str(resolution)]
    except KeyError as exc:
        message = (
            f"{source.rstrip('/')}/{resolution} (resolution {str(resolution)!r} "
            f"in zarr store {source!r})"
        )
        try:
            available = sorted(root.keys())
        except Exception:
            available = ()
        if available:
            message += "; available top-level keys: " + ", ".join(available[:20])
        raise FileNotFoundError(message) from exc


def open_volume(
    path: str | Path,
    resolution: int | str,
    auth_json_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    cache_max_gb: float | None = None,
    root_array_is_requested_level: bool = False,
):
    """Open one Zarr pyramid level through the shared root boundary."""

    root = open_volume_root(
        path,
        auth_json_path,
        cache_dir=cache_dir,
        cache_max_gb=cache_max_gb,
    )
    return select_volume_level(
        root,
        resolution,
        source=str(path),
        root_array_is_requested_level=root_array_is_requested_level,
    )


def read_bbox_with_padding(
    volume: Any,
    bbox_zyx: tuple[int, int, int, int, int, int],
    *,
    fill_value: int | float = 0,
) -> tuple[np.ndarray, tuple[slice, slice, slice] | None]:
    """Read a positive ZYX bbox, padding only outside the array bounds."""
    z0, y0, x0, z1, y1, x1 = (int(value) for value in bbox_zyx)
    expected_shape = z1 - z0, y1 - y0, x1 - x0
    if any(size <= 0 for size in expected_shape):
        raise ValueError(f"bbox must define a positive crop, got {bbox_zyx!r}")
    shape = tuple(int(value) for value in volume.shape[:3])
    starts = max(0, z0), max(0, y0), max(0, x0)
    stops = min(shape[0], z1), min(shape[1], y1), min(shape[2], x1)
    output = np.full(expected_shape, fill_value, dtype=np.dtype(volume.dtype))
    if any(stop <= start for start, stop in zip(starts, stops)):
        return output, None
    crop = np.asarray(
        volume[
            starts[0] : stops[0],
            starts[1] : stops[1],
            starts[2] : stops[2],
        ]
    )
    destination_starts = starts[0] - z0, starts[1] - y0, starts[2] - x0
    destination = tuple(
        slice(start, start + size)
        for start, size in zip(destination_starts, crop.shape)
    )
    output[destination] = crop
    return output, destination
