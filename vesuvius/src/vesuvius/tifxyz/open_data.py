"""Open-data helpers: ``original_volume_downscale`` for ``tifxyz_normalized`` meshes.

In the Vesuvius Challenge open-data bucket every segment ships a normalized mesh at
``<sample>/segments/<segment>/mesh/intermediate/tifxyz_normalized/``. For some segments
those coordinates are stored at ``1 / original_volume_downscale`` of the segment's
original volume (factor 4 or 2 on the 2026-10-07 catalog). The factor is recorded in
the bucket catalog (``metadata.json``)::

    samples[<sample>]["segments"][<segment id>]["properties"]["original_volume_downscale"]

The mesh's own ``meta.json`` does not record it. A caller that uses those
coordinates as voxels of the original volume, or applies a catalog transform to
them, is off by a factor of ``original_volume_downscale``. That is centimetres
on the published scrolls.

``read_tifxyz(path, downscale=...)`` applies the factor explicitly. The helpers
here look it up from a local copy of the catalog. Nothing here downloads
anything.
"""

from __future__ import annotations

import gzip
import json
import math
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, Union

DOWNSCALE_PROPERTY = "original_volume_downscale"
DOWNSCALE_APPLIED_KEY = "original_volume_downscale_applied"
# VC3D's Open Data cache writes this meta.json key after it has scaled a cached segment
# (apps/VC3D/OpenDataSegmentCache.cpp, applyOriginalVolumeDownscale). Either key means the
# coordinates are already original-volume voxels.
VC3D_SCALED_KEY = "vc_open_data_coordinates_scaled_to_original_volume"
ALREADY_SCALED_KEYS = (DOWNSCALE_APPLIED_KEY, VC3D_SCALED_KEY)
NORMALIZED_MESH_PARTS = ("mesh", "intermediate", "tifxyz_normalized")

CatalogLike = Union[Mapping[str, Any], str, Path]


class OpenDataDownscaleWarning(UserWarning):
    """A ``tifxyz_normalized`` mesh was read without an explicit ``downscale``."""


def load_open_data_catalog(path: Union[str, Path]) -> dict:
    """Load a local copy of the open-data ``metadata.json`` (plain or gzip-compressed)."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _catalog(catalog: CatalogLike) -> Mapping[str, Any]:
    if isinstance(catalog, (str, Path)):
        return load_open_data_catalog(catalog)
    return catalog


def _find_segment(catalog: Mapping[str, Any], sample: str, segment: str) -> Mapping[str, Any]:
    try:
        segments = catalog["samples"][sample]["segments"]
    except KeyError:
        raise KeyError(f"sample {sample!r} not found in the open-data catalog") from None
    if segment in segments:
        return segments[segment]
    # The bucket directory name is the segment's long_id (e.g. "20250108000004-w029_2025010827").
    hits = [g for g in segments.values() if g.get("long_id") == segment]
    if len(hits) != 1:
        raise KeyError(f"segment {segment!r} not found in sample {sample!r} of the open-data catalog")
    return hits[0]


def original_volume_downscale(catalog: CatalogLike, sample: str, segment: str) -> float:
    """Return ``properties.original_volume_downscale`` for a segment (1.0 if absent).

    ``catalog`` is the parsed ``metadata.json`` or a path to a local copy.
    ``segment`` may be the catalog segment id or its ``long_id`` (the bucket directory name).
    """
    seg = _find_segment(_catalog(catalog), sample, segment)
    value = (seg.get("properties") or {}).get(DOWNSCALE_PROPERTY)
    factor = 1.0 if value is None else float(value)
    _check_factor(factor)
    return factor


def parse_open_data_normalized_path(path: Union[str, Path]) -> Optional[Tuple[str, str]]:
    """Return ``(sample, segment_dir)`` if ``path`` follows the bucket layout
    ``<sample>/segments/<segment_dir>/mesh/intermediate/tifxyz_normalized``, else None."""
    parts = Path(path).parts
    n = len(NORMALIZED_MESH_PARTS)
    if len(parts) < n + 3 or tuple(parts[-n:]) != NORMALIZED_MESH_PARTS:
        return None
    if parts[-n - 2] != "segments":
        return None
    return parts[-n - 3], parts[-n - 1]


def is_open_data_normalized_path(path: Union[str, Path]) -> bool:
    """True if the directory is named like an open-data ``.../mesh/intermediate/tifxyz_normalized``."""
    return tuple(Path(path).parts[-len(NORMALIZED_MESH_PARTS):]) == NORMALIZED_MESH_PARTS


def _check_factor(factor: float) -> None:
    if not (isinstance(factor, (int, float)) and math.isfinite(factor) and factor > 0):
        raise ValueError(f"downscale must be a finite number > 0, got {factor!r}")


def applied_downscale_marker(extra: Mapping[str, Any]) -> Optional[Tuple[str, Any]]:
    """Return ``(key, value)`` of the first already-scaled marker in a mesh's ``meta.json`` extras, or None."""
    for key in ALREADY_SCALED_KEYS:
        if extra.get(key) is not None:
            return key, extra[key]
    return None


def warn_missing_downscale(path: Union[str, Path]) -> None:
    """Warn that a ``tifxyz_normalized`` mesh was read without an explicit ``downscale``."""
    warnings.warn(
        f"{path}: open-data tifxyz_normalized coordinates may be stored at "
        f"1/{DOWNSCALE_PROPERTY} of the segment's original volume (e.g. x4 or x2), and "
        "meta.json does not record the factor. Pass downscale=<catalog "
        f"properties.{DOWNSCALE_PROPERTY}> (or downscale=1 if you know it is 1), or use "
        "vesuvius.tifxyz.read_open_data_normalized(path, catalog).",
        OpenDataDownscaleWarning,
        stacklevel=3,
    )


CATALOG_ENV = "VESUVIUS_OPEN_DATA_CATALOG"


def warn_known_downscale(path: Union[str, Path], factor: float, source: str) -> None:
    """Warn that a mesh with a known factor != 1 was read without applying it, and say how."""
    warnings.warn(
        f"{path}: {source} gives {DOWNSCALE_PROPERTY}={factor:g}, so these coordinates are "
        f"1/{factor:g} of the segment's original volume voxels. Pass downscale={factor:g} "
        "(or downscale='catalog') to read_tifxyz to apply it.",
        OpenDataDownscaleWarning,
        stacklevel=3,
    )


def lookup_downscale(path: Union[str, Path], extra: Mapping[str, Any],
                     catalog: Optional[CatalogLike]) -> Tuple[Optional[float], Optional[str]]:
    """Find the factor for a mesh: its own meta.json field first, then the catalog.

    Returns ``(factor, source)`` or ``(None, None)`` when neither is available.
    ``catalog`` defaults to the path in ``$VESUVIUS_OPEN_DATA_CATALOG``.
    """
    import os

    if extra.get(DOWNSCALE_PROPERTY) is not None:
        factor = float(extra[DOWNSCALE_PROPERTY])
        _check_factor(factor)
        return factor, "meta.json"
    if catalog is None:
        catalog = os.environ.get(CATALOG_ENV) or None
    parsed = parse_open_data_normalized_path(path)
    if catalog is None or parsed is None:
        return None, None
    return original_volume_downscale(catalog, *parsed), "the open-data catalog"


def read_open_data_normalized(
    path: Union[str, Path],
    catalog: CatalogLike,
    *,
    sample: Optional[str] = None,
    segment: Optional[str] = None,
    **read_kwargs: Any,
):
    """Read an open-data ``tifxyz_normalized`` mesh in original-volume voxel coordinates.

    Looks up ``original_volume_downscale`` in ``catalog`` (parsed ``metadata.json`` or a
    path to a local copy) and passes it to ``read_tifxyz(..., downscale=...)``. ``sample``
    and ``segment`` default to the names in a bucket-layout ``path``.
    """
    from .reader import read_tifxyz

    if sample is None or segment is None:
        parsed = parse_open_data_normalized_path(path)
        if parsed is None:
            raise ValueError(
                f"{path}: cannot infer sample/segment from the path; pass sample= and segment="
            )
        sample = parsed[0] if sample is None else sample
        segment = parsed[1] if segment is None else segment
    factor = original_volume_downscale(catalog, sample, segment)
    return read_tifxyz(path, downscale=factor, **read_kwargs)
