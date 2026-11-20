"""Zarr opening through vc_reader, resolution selection, and padded crops."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from vc_reader.reader import load_basic_auth, open_node


_PUBLIC_S3_VOLUME_SUBSTRING = "vesuvius-challenge-open-data"


def normalized_source_path(volume_path: str | Path) -> str:
    """Resolve local path aliases while preserving remote source URLs."""
    volume_path = str(volume_path)
    if "://" in volume_path:
        return volume_path.rstrip("/")
    return str(Path(volume_path).expanduser().resolve())


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


class PyramidDepthScaleError(ValueError):
    """A pyramid level's depth disagrees with its declared Z scale."""


def _multiscale_entry(root: Any) -> dict[str, Any] | None:
    attrs = getattr(root, "attrs", None)
    if not isinstance(attrs, dict):
        return None
    multiscales = attrs.get("multiscales") or []
    if not multiscales or not isinstance(multiscales[0], dict):
        return None
    return multiscales[0]


def _dataset_by_path(entry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for dataset in entry.get("datasets") or []:
        if isinstance(dataset, dict) and dataset.get("path") is not None:
            found[str(dataset["path"])] = dataset
    return found


def _depth_axis(entry: dict[str, Any], rank: int) -> int:
    for index, axis in enumerate(entry.get("axes") or []):
        if isinstance(axis, dict) and str(axis.get("name", "")).lower() == "z" and index < rank:
            return index
    return 0


def _positive_scale_component(dataset: dict[str, Any], axis: int) -> float | None:
    for transform in dataset.get("coordinateTransformations") or []:
        if not isinstance(transform, dict):
            continue
        scale = transform.get("scale")
        if not isinstance(scale, (list, tuple)) or axis >= len(scale):
            continue
        try:
            value = float(scale[axis])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value
    return None


def require_pyramid_depth_scale(
    root: Any,
    resolution: int | str,
    *,
    source: str = "",
) -> None:
    """Reject a level whose depth does not match its declared Z scale.

    A Z-scale ratio near 1 says the level kept every depth slice and only
    changed the in-plane sampling. A larger ratio says depth was pooled by
    about that factor. Published ink volumes that declare a Z scale of 1
    while storing only a prefix of the stack (issue 2107) are refused here.
    Missing multiscale metadata is left unchanged, and this does not rebuild
    a published pyramid.
    """

    if hasattr(root, "shape"):
        return
    entry = _multiscale_entry(root)
    if entry is None:
        return
    datasets = _dataset_by_path(entry)
    level = str(resolution)
    reference_name = "0" if "0" in datasets else next(iter(datasets), "")
    if not reference_name or level == reference_name or level not in datasets:
        return
    try:
        reference = root[reference_name]
        selected = root[level]
    except KeyError:
        return
    reference_shape = tuple(int(value) for value in reference.shape)
    selected_shape = tuple(int(value) for value in selected.shape)
    if len(reference_shape) < 1 or len(reference_shape) != len(selected_shape):
        return
    axis = _depth_axis(entry, len(reference_shape))
    reference_scale = _positive_scale_component(datasets[reference_name], axis)
    selected_scale = _positive_scale_component(datasets[level], axis)
    if reference_scale is None or selected_scale is None:
        return
    ratio = selected_scale / reference_scale
    reference_depth = reference_shape[axis]
    selected_depth = selected_shape[axis]
    where = f" of {source}" if source else ""
    if abs(ratio - 1.0) <= 1e-3:
        if selected_depth != reference_depth:
            raise PyramidDepthScaleError(
                f"Pyramid level {level}{where} has depth {selected_depth}, "
                f"but level {reference_name} has depth {reference_depth} and "
                f"the declared Z scale ratio is {ratio:.6g}. A ratio near 1 "
                "keeps every depth slice. Rebuild this level from the full stack."
            )
        return
    expected = reference_depth / ratio
    tolerance = max(1.0, 0.08 * expected)
    if abs(selected_depth - expected) > tolerance:
        raise PyramidDepthScaleError(
            f"Pyramid level {level}{where} has depth {selected_depth}; the "
            f"declared Z scale ratio {ratio:.6g} against level {reference_name} "
            f"depth {reference_depth} expects about {expected:.4g}. Rebuild "
            "this level from the full stack, or correct its Z scale."
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
    require_pyramid_depth_scale(root, resolution, source=source)
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
    output, destination, region = _padded_bbox(volume, bbox_zyx, fill_value)
    if region is not None:
        start, size = region
        selection = tuple(slice(a, a + n) for a, n in zip(start, size))
        output[destination] = np.asarray(volume[selection])
    return output, destination


def read_bboxes_with_padding(volume, bboxes_zyx, *, fill_value=0):
    """Read padded crops together through vc_reader's shared-chunk batch API."""
    plans = [_padded_bbox(volume, bbox, fill_value) for bbox in bboxes_zyx]
    regions = [region for _, _, region in plans if region is not None]
    outputs = [
        output[destination] for output, destination, region in plans
        if region is not None
    ]
    if regions:
        volume.read_regions(regions, out=outputs)
    return [(output, destination) for output, destination, _ in plans]


def _padded_bbox(volume, bbox_zyx, fill_value):
    """Allocate a crop and plan its in-bounds destination and source region."""
    z0, y0, x0, z1, y1, x1 = (int(value) for value in bbox_zyx)
    expected_shape = z1 - z0, y1 - y0, x1 - x0
    if any(size <= 0 for size in expected_shape):
        raise ValueError(f"bbox must define a positive crop, got {bbox_zyx!r}")
    shape = tuple(int(value) for value in volume.shape[:3])
    starts = max(0, z0), max(0, y0), max(0, x0)
    stops = min(shape[0], z1), min(shape[1], y1), min(shape[2], x1)
    output = np.full(expected_shape, fill_value, dtype=np.dtype(volume.dtype))
    if any(stop <= start for start, stop in zip(starts, stops)):
        return output, None, None
    size = tuple(stop - start for start, stop in zip(starts, stops))
    destination_starts = starts[0] - z0, starts[1] - y0, starts[2] - x0
    destination = tuple(
        slice(start, start + length)
        for start, length in zip(destination_starts, size)
    )
    return output, destination, (starts, size)
