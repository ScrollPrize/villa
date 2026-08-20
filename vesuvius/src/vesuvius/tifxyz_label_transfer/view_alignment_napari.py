#!/usr/bin/env python3
"""Inspect TIFXYZ label transfer and volume registration in one Napari viewer.

Blue always means the earlier/source representation and red means the
updated/target representation. With additive blending, matching content is
purple. The layer list contains togglable groups for:

- HF-source versus updated-2.399um labels and supervision;
- updated-2.399um versus projected-to-9.362um labels and supervision;
- middle-three-layer max composites from the actual surface-volume Zarrs;
- raw stage-one and affine-registered 3D TIFXYZ point overlays.

The 9.362um label layers are projections of the original 2.399um annotations,
not independent native annotations.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.spatial import cKDTree
import tifffile

from vesuvius.utils.cli import HyphenUnderscoreParser

from .core import transfer_array as transfer_surface_array
from .estimate_canvas_offset import measure_render_shift
from .io import load_surface as load_full_surface
from .prepare_canvas_offset_evidence import (
    DEFAULT_INK_ROOT,
    DEFAULT_OPEN_DATA_ROOT,
    extract_composite as extract_rclone_composite,
    inspect_zarr as inspect_rclone_zarr,
)


INITIAL_ZOOM_FACTOR = 1.6


@dataclass
class Surface:
    points_xyz: np.ndarray
    flat_indices: np.ndarray
    shape: tuple[int, int]


@dataclass
class Case:
    root: Path
    name: str
    source_tifxyz: Path
    updated_tifxyz: Path
    target_tifxyz: Path | None
    affine_path: Path | None
    source_label: Path
    source_supervision: Path
    source_surface_volume_url: str | None
    surface_volume_urls: dict[float, str]
    results: Path
    updated_resolution: float
    target_resolution: float | None
    source_validation: Path | None = None

    @property
    def updated_stage(self) -> str:
        resolution_stage = f"{self.updated_resolution:g}um"
        if (
            self.results / f"inklabels-{resolution_stage}.report.json"
        ).is_file():
            return resolution_stage

        # Single-destination batches use a stable semantic name so 2.399 um
        # and 2.4 um targets can be consumed uniformly.  Keep the historical
        # resolution-derived name as the default for older case directories.
        semantic_stage = "updated-2um"
        if (
            self.results / f"inklabels-{semantic_stage}.report.json"
        ).is_file():
            return semantic_stage
        return resolution_stage

    @property
    def target_stage(self) -> str | None:
        if self.target_resolution is None:
            return None
        return f"{self.target_resolution:g}um"


@dataclass
class PreparedCase:
    case: Case
    summary: dict[str, Any]
    surfaces: dict[str, Surface]
    affine: np.ndarray | None
    source_render: np.ndarray | None
    renders: dict[str, np.ndarray]
    registered_hf_to_updated_render: np.ndarray | None
    registered_source_render: np.ndarray | None
    diagnostics: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    registered_hf_label_to_updated: np.ndarray | None = None
    registered_hf_supervision_to_updated: np.ndarray | None = None
    stage_one_affine: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )


@dataclass(frozen=True)
class ComparisonPreset:
    key: str
    label: str
    description: str
    ndisplay: int
    section: str


COMPARISON_PRESETS = (
    ComparisonPreset(
        "render_same",
        "2 µm CT alignment",
        "Original HF projected → updated 2.399 µm",
        2,
        "2D",
    ),
    ComparisonPreset(
        "render_registered",
        "Final CT alignment",
        "Updated 2.4 projected → native 9 µm",
        2,
        "2D",
    ),
    ComparisonPreset(
        "self_center",
        "Self-render · center",
        "Raw CT sampled on source and target TIFXYZ, compared on target canvas",
        2,
        "2D evidence",
    ),
    ComparisonPreset(
        "self_max",
        "Self-render · matched max",
        "Raw CT matched slabs sampled on both TIFXYZ surfaces",
        2,
        "2D evidence",
    ),
    ComparisonPreset(
        "evidence_center",
        "Cross evidence · center",
        "Published exact-center renders after source→target projection",
        2,
        "2D evidence",
    ),
    ComparisonPreset(
        "evidence_max",
        "Cross evidence · matched max",
        "Published annotation slab versus physically matched target slab",
        2,
        "2D evidence",
    ),
    ComparisonPreset(
        "aligned_original",
        "Stage · original 2 µm",
        "Native original HF CT with native annotations",
        2,
        "2D",
    ),
    ComparisonPreset(
        "aligned_updated",
        "Stage · updated 2 µm",
        "Native updated CT with transferred annotations",
        2,
        "2D",
    ),
    ComparisonPreset(
        "aligned_final",
        "Stage · final 9 µm",
        "Native final CT and projected annotations",
        2,
        "2D",
    ),
    ComparisonPreset(
        "render_label",
        "Final 9 µm + annotations",
        "Native CT with projected ink and mask",
        2,
        "2D",
    ),
    ComparisonPreset(
        "label_same",
        "Updated 2 µm + ink",
        "Updated CT with the transferred HF ink annotation",
        2,
        "2D",
    ),
    ComparisonPreset(
        "supervision_same",
        "Updated 2 µm + mask",
        "Updated CT with the transferred HF supervision mask",
        2,
        "2D",
    ),
    ComparisonPreset(
        "validation_same",
        "Updated 2 µm + validation",
        "Updated CT with the transferred HF validation mask",
        2,
        "2D",
    ),
    ComparisonPreset(
        "label_cross",
        "Final 9 µm + ink",
        "Native final CT with the projected ink annotation",
        2,
        "2D",
    ),
    ComparisonPreset(
        "supervision_cross",
        "Final 9 µm + mask",
        "Native final CT with the projected supervision mask",
        2,
        "2D",
    ),
    ComparisonPreset(
        "validation_cross",
        "Final 9 µm + validation",
        "Native final CT with the projected validation mask",
        2,
        "2D",
    ),
    ComparisonPreset(
        "validity_same",
        "Coverage · updated 2 µm",
        "Old → updated acceptance on the updated CT canvas",
        2,
        "2D",
    ),
    ComparisonPreset(
        "validity_cross",
        "Coverage · final 9 µm",
        "Composed two-stage acceptance on the final CT canvas",
        2,
        "2D",
    ),
    ComparisonPreset(
        "surface_same",
        "Surfaces · HF / updated",
        "Stage-one-registered TIFXYZ points",
        3,
        "3D",
    ),
    ComparisonPreset(
        "surface_affine",
        "Surfaces · 2.4 / 9 µm",
        "Affine TIFXYZ plus residuals",
        3,
        "3D",
    ),
    ComparisonPreset(
        "label3d_same",
        "Ink points · HF / transferred",
        "Stage-one-registered annotation points",
        3,
        "3D",
    ),
    ComparisonPreset(
        "label3d_affine",
        "Ink points · 2.4 / 9 µm",
        "Affine annotation points",
        3,
        "3D",
    ),
    ComparisonPreset(
        "supervision3d_same",
        "Mask points · HF / transferred",
        "Stage-one-registered supervision points",
        3,
        "3D",
    ),
    ComparisonPreset(
        "supervision3d_affine",
        "Mask points · 2.4 / 9 µm",
        "Affine supervision points",
        3,
        "3D",
    ),
)

PRESET_LAYER_LABELS = {
    "render_same": (
        "Blue original",
        "Red updated",
        "Ink transferred",
        "Mask transferred",
        "Ink HF",
        "Mask HF",
    ),
    "render_registered": ("Blue 2.4", "Red 9"),
    "self_center": ("Blue source", "Red target"),
    "self_max": ("Blue source", "Red target"),
    "evidence_center": ("Blue source", "Red target"),
    "evidence_max": ("Blue source", "Red target"),
    "aligned_original": ("CT", "Ink", "Supervision", "Validation"),
    "aligned_updated": ("CT", "Ink", "Supervision", "Validation"),
    "aligned_final": ("CT", "Ink", "Supervision", "Validation"),
    "render_label": ("9 µm CT", "Ink", "Mask"),
    "label_same": ("Updated CT", "Transferred ink"),
    "supervision_same": ("Updated CT", "Transferred mask"),
    "validation_same": ("Updated CT", "Transferred validation"),
    "label_cross": ("Final CT", "Projected ink"),
    "supervision_cross": ("Final CT", "Projected mask"),
    "validation_cross": ("Final CT", "Projected validation"),
    "validity_same": ("Updated CT", "Accepted"),
    "validity_cross": ("Final CT", "Accepted"),
    "surface_same": ("Blue HF", "Red updated"),
    "surface_affine": ("Blue 2.4", "Red 9", "Residuals"),
    "label3d_same": ("Blue HF", "Red transferred"),
    "label3d_affine": ("Blue 2.4", "Red 9"),
    "supervision3d_same": ("Blue HF", "Red transferred"),
    "supervision3d_affine": ("Blue 2.4", "Red 9"),
}


def should_refit_camera(
    previous_ndisplay: int,
    current_ndisplay: int,
    force_frame: bool,
) -> bool:
    """Refit only for a new case or a 2D/3D display-mode transition."""

    return force_frame or previous_ndisplay != current_ndisplay


def calculate_2d_camera(
    extents: list[np.ndarray],
    canvas_size: np.ndarray,
) -> tuple[tuple[float, float], float]:
    """Return center and zoom for only the active 2D layer extents."""

    lower = np.min([extent[0] for extent in extents], axis=0)
    upper = np.max([extent[1] for extent in extents], axis=0)
    scene_size = np.maximum(upper - lower, 1e-9)
    center = tuple(float(value) for value in (lower + upper) / 2)
    zoom = (
        0.92
        * float(np.min(canvas_size / scene_size))
        * INITIAL_ZOOM_FACTOR
    )
    return center, zoom


def valid_coordinates(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(z)
        & (x != -1)
        & (y != -1)
        & (z > 0)
    )


def load_surface(path: Path, max_points: int) -> Surface:
    x = tifffile.imread(path / "x.tif")
    y = tifffile.imread(path / "y.tif")
    z = tifffile.imread(path / "z.tif")
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(f"Coordinate shapes differ in {path}")
    valid = valid_coordinates(x, y, z).ravel()
    indices = np.flatnonzero(valid)
    if indices.size > max_points:
        selection = np.linspace(
            0, indices.size - 1, max_points, dtype=np.int64
        )
        indices = indices[selection]
    points = np.column_stack(
        (x.ravel()[indices], y.ravel()[indices], z.ravel()[indices])
    ).astype(np.float64, copy=False)
    return Surface(
        points_xyz=points,
        flat_indices=indices,
        shape=(int(x.shape[0]), int(x.shape[1])),
    )


def apply_affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def nearest_summary(
    source_points: np.ndarray, target_points: np.ndarray
) -> dict[str, float]:
    distances, _ = cKDTree(target_points).query(source_points, workers=-1)
    return {
        "median": float(np.median(distances)),
        "p95": float(np.percentile(distances, 95)),
        "max": float(np.max(distances)),
    }


def _tifxyz_resolution(path: Path) -> float:
    prefix = path.name.split("um-", 1)[0]
    try:
        return float(prefix)
    except ValueError as error:
        raise ValueError(
            f"Cannot parse resolution from TIFXYZ name {path.name}; "
            "expected '<resolution>um-<volume_id>.tifxyz'"
        ) from error


def _remote_join(root: str, path: str) -> str:
    return root.rstrip("/") + "/" + path.strip("/")


def preferred_annotation_image(paths: list[Path]) -> Path:
    """Choose the highest explicit ``_vN`` annotation revision."""

    if not paths:
        raise ValueError("no annotation images supplied")

    def key(path: Path) -> tuple[int, str]:
        match = re.search(r"_v(\d+)\.(?:tiff?|zarr)$", path.name.lower())
        return (int(match.group(1)) if match else 1, path.name)

    return max(paths, key=key)


def discover_case(
    root: Path,
    ink_rclone_root: str | None = DEFAULT_INK_ROOT,
    open_data_rclone_root: str = DEFAULT_OPEN_DATA_ROOT,
) -> Case:
    selection_path = root / "selection.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source_volume = selection["segment"]["original_volume_id"]

    tifxyz_matches = sorted((root / "open-data").glob("*um-*.tifxyz"))
    if len(tifxyz_matches) not in {1, 2}:
        raise ValueError(
            f"{root} must contain one (updated only) or two (updated and "
            f"native target) open-data TIFXYZ, found {len(tifxyz_matches)}"
        )
    tifxyz_matches.sort(key=_tifxyz_resolution)
    updated_tifxyz = tifxyz_matches[0]
    updated_resolution = _tifxyz_resolution(updated_tifxyz)
    target_tifxyz: Path | None = None
    target_resolution: float | None = None
    affine_path: Path | None = None
    if len(tifxyz_matches) == 2:
        target_tifxyz = tifxyz_matches[1]
        target_resolution = _tifxyz_resolution(target_tifxyz)
        target_volume = (
            target_tifxyz.name.split("um-", 1)[1].split(".", 1)[0]
        )
        affine_path = (
            root / "affines" / f"{source_volume}-to-{target_volume}.json"
        )
    labels = sorted((root / "hf" / "labels").glob("*.tif*"))
    labels.extend(
        sorted((root / "hf" / "v2").rglob("*_inklabels_v*.tif*"))
    )
    labels.extend(
        sorted((root / "hf" / "v2").rglob("*_inklabels_v*.zarr"))
    )
    masks = sorted((root / "hf" / "supervision-masks").glob("*.tif*"))
    masks.extend(
        sorted((root / "hf" / "v2").rglob("*_supervision_mask_v*.tif*"))
    )
    masks.extend(
        sorted((root / "hf" / "v2").rglob("*_supervision_mask_v*.zarr"))
    )
    validation_masks = sorted(
        (root / "hf" / "validation-masks").glob("*.tif*")
    )
    validation_masks.extend(
        sorted((root / "hf" / "v2").rglob("*_validation_mask_v*.tif*"))
    )
    validation_masks.extend(
        sorted((root / "hf" / "v2").rglob("*_validation_mask_v*.zarr"))
    )
    if not labels or not masks:
        raise ValueError(f"Missing source label or supervision mask in {root}")

    surface_volumes = selection.get("surface_volumes")
    if surface_volumes is None:
        raise ValueError(
            f"{selection_path} predates surface-volume support; rerun "
            "download_ink_app_inputs.py to refresh it"
        )
    # The transferred target can exist without a matching native CT surface
    # volume (currently PHerc1667 and PHercParis4 at 7.91 um).  Keep the target
    # TIFXYZ/report available for geometric inspection, but require CT only for
    # the updated stage so validation can still measure old -> updated.
    requested = {updated_resolution} | (
        {target_resolution} if target_resolution is not None else set()
    )
    surface_urls = {
        float(volume["resolution_um"]): _remote_join(
            open_data_rclone_root, volume["path"]
        )
        for volume in surface_volumes
        if float(volume.get("resolution_um") or -1.0) in requested
    }
    if updated_resolution not in surface_urls:
        raise ValueError(
            f"Missing {updated_resolution:g}um surface Zarr for {root}"
        )
    source_zarrs = selection.get("source_surface_zarrs")
    if not source_zarrs or len(source_zarrs) != 1:
        raise ValueError(
            f"{selection_path} lacks one HF source surface Zarr; rerun "
            "download_ink_app_inputs.py to refresh it"
        )
    source_path = source_zarrs[0]["path"].strip("/")
    if source_path.startswith("ink/"):
        source_path = source_path[len("ink/") :]
    # The ink dataset is private; without an explicit mirror the viewer can
    # still run from cached/prepared renders and only fails on a remote fetch.
    source_surface_url = (
        _remote_join(ink_rclone_root, source_path)
        if ink_rclone_root
        else None
    )

    return Case(
        root=root,
        name=root.name,
        source_tifxyz=root / "hf" / "source.tifxyz",
        updated_tifxyz=updated_tifxyz,
        target_tifxyz=target_tifxyz,
        affine_path=affine_path,
        source_label=preferred_annotation_image(labels),
        source_supervision=preferred_annotation_image(masks),
        source_surface_volume_url=source_surface_url,
        surface_volume_urls=surface_urls,
        results=root / "results",
        updated_resolution=updated_resolution,
        target_resolution=target_resolution,
        source_validation=(
            preferred_annotation_image(validation_masks)
            if validation_masks
            else None
        ),
    )


def resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.minimum(
        ((np.arange(shape[0]) + 0.5) * image.shape[0] / shape[0]).astype(int),
        image.shape[0] - 1,
    )
    columns = np.minimum(
        ((np.arange(shape[1]) + 0.5) * image.shape[1] / shape[1]).astype(int),
        image.shape[1] - 1,
    )
    return image[np.ix_(rows, columns)]


def preview_render(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a render for visual review without changing geometry data."""

    if factor <= 0:
        raise ValueError("preview factor must be positive")
    if factor == 1:
        return image
    shape = tuple(max(1, math.ceil(size / factor)) for size in image.shape)
    return resize_nearest(image, shape)


def preview_cache_name(name: str, factor: int) -> str:
    """Keep approximate preview projections separate from full-resolution caches."""

    if factor <= 0:
        raise ValueError("preview factor must be positive")
    if factor == 1:
        return name
    path = Path(name)
    return f"{path.stem}-preview{factor}{path.suffix}"


def read_render_tiff(path: Path) -> np.ndarray:
    """Open a cached render without copying the complete raster into RAM."""

    try:
        return tifffile.memmap(path, mode="r")
    except (OSError, ValueError):
        # Externally supplied evidence may be tiled or compressed and therefore
        # cannot be memory-mapped.  Keep compatibility with those files.
        return tifffile.imread(path)


def read_tiff_nearest(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Read a 2D TIFF directly at ``shape`` without materialising it whole.

    Transfer TIFFs can decode to several gigabytes even though the viewer only
    needs a render-sized raster. Decode just the strips or tiles touched by the
    nearest-neighbour output grid and discard each compressed segment
    immediately.
    """

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        if len(series.pages) != 1 or len(series.shape) != 2:
            return resize_nearest(tif.asarray(), shape)
        page = series.pages[0]
        source_shape = tuple(int(value) for value in series.shape)
        if source_shape == shape:
            return tif.asarray()
        rows = np.minimum(
            (
                (np.arange(shape[0], dtype=np.float64) + 0.5)
                * source_shape[0]
                / shape[0]
            ).astype(np.int64),
            source_shape[0] - 1,
        )
        columns = np.minimum(
            (
                (np.arange(shape[1], dtype=np.float64) + 0.5)
                * source_shape[1]
                / shape[1]
            ).astype(np.int64),
            source_shape[1] - 1,
        )
        output = np.empty(shape, dtype=series.dtype)
        handle = tif.filehandle

        if page.is_tiled:
            tile_height = int(page.tilelength)
            tile_width = int(page.tilewidth)
            tiles_across = math.ceil(source_shape[1] / tile_width)
            row_tiles = np.unique(rows // tile_height)
            column_tiles = np.unique(columns // tile_width)
            for tile_row in row_tiles:
                output_rows = np.flatnonzero(rows // tile_height == tile_row)
                source_rows = rows[output_rows] - tile_row * tile_height
                for tile_column in column_tiles:
                    output_columns = np.flatnonzero(
                        columns // tile_width == tile_column
                    )
                    source_columns = (
                        columns[output_columns] - tile_column * tile_width
                    )
                    index = int(tile_row * tiles_across + tile_column)
                    handle.seek(page.dataoffsets[index])
                    encoded = handle.read(page.databytecounts[index])
                    decoded, _, _ = page.decode(encoded, index)
                    tile = np.asarray(decoded).squeeze()
                    output[np.ix_(output_rows, output_columns)] = tile[
                        np.ix_(source_rows, source_columns)
                    ]
            return output

        rows_per_strip = int(page.rowsperstrip)
        if rows_per_strip > 0:
            for strip in np.unique(rows // rows_per_strip):
                output_rows = np.flatnonzero(rows // rows_per_strip == strip)
                source_rows = rows[output_rows] - strip * rows_per_strip
                index = int(strip)
                handle.seek(page.dataoffsets[index])
                encoded = handle.read(page.databytecounts[index])
                decoded, _, _ = page.decode(encoded, index)
                segment = np.asarray(decoded).squeeze()
                if segment.ndim == 1:
                    segment = segment[None, :]
                output[output_rows] = segment[np.ix_(source_rows, columns)]
            return output

        return resize_nearest(tif.asarray(), shape)


def read_label_display(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Read a transferred label from its OME-Zarr pyramid when available."""

    zarr_path = path if path.suffix == ".zarr" else path.with_suffix(".zarr")
    if zarr_path.is_dir():
        import zarr

        group = zarr.open_group(str(zarr_path), mode="r")
        levels = sorted((int(key) for key in group.array_keys()))
        candidates = [
            level
            for level in levels
            if group[str(level)].shape[1] >= shape[0]
            and group[str(level)].shape[2] >= shape[1]
        ]
        level = max(candidates) if candidates else min(levels)
        array = group[str(level)]
        image = np.asarray(array[array.shape[0] // 2])
        return image if image.shape == shape else resize_nearest(image, shape)
    return read_tiff_nearest(path, shape)


def transferred_result_path(
    results: Path,
    kind: str,
    stage: str,
    suffix: str = ".tif",
) -> Path:
    """Resolve historical and semantic transferred-output names."""

    names = [f"{kind}-{stage}{suffix}"]
    if kind in {"supervision", "validation"}:
        names.append(f"{kind}-mask-{stage}{suffix}")
    for name in names:
        path = results / name
        if path.is_file():
            return path
    # Preserve the historical path in the eventual FileNotFoundError when no
    # supported spelling exists.
    return results / names[0]


def load_middle_three_max(
    root_url: str,
    cache_path: Path,
    preferred_level: int,
    workers: int = 12,
    plane_count: int = 3,
) -> np.ndarray:
    """Load an odd-sized centered Z maximum without syncing the Zarr."""
    if plane_count <= 0 or plane_count % 2 == 0:
        raise ValueError("plane_count must be a positive odd integer")
    if cache_path.is_file():
        print(f"Using cached middle-{plane_count} max: {cache_path}")
        return read_render_tiff(cache_path)
    if "://" not in root_url:
        info = inspect_rclone_zarr(root_url, preferred_level)
        middle = info.shape[0] // 2
        radius = plane_count // 2
        extract_rclone_composite(
            info,
            list(
                range(
                    max(0, middle - radius),
                    min(info.shape[0], middle + radius + 1),
                )
            ),
            cache_path,
            workers=workers,
            overwrite=False,
        )
        print(
            f"Using rclone-backed middle-{plane_count} max: {cache_path}"
        )
        return read_render_tiff(cache_path)

    raise ValueError(
        "HTTP/URL zarr access is disabled; data must be read with "
        f"rclone. Pass an rclone remote path (remote:bucket/path) "
        f"instead of {root_url!r}"
    )


def evidence_comparison_render(
    case: Case, comparison_name: str, key: str
) -> np.ndarray | None:
    """Load one locally prepared evidence raster by comparison name."""

    manifest_path = (
        case.root / "renders" / "offset-evidence" / "manifest.json"
    )
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if comparison_name == "exact-center":
        source_center = manifest.get("source_center") or {}
        selected_shape = source_center.get("selected_level_shape_zyx")
        full_shape = source_center.get("full_resolution_shape_zyx")
        if (
            isinstance(selected_shape, list)
            and isinstance(full_shape, list)
            and len(selected_shape) == 3
            and len(full_shape) == 3
            and int(selected_shape[0]) != int(full_shape[0])
        ):
            return None
    matches = [
        item
        for item in manifest.get("comparisons", [])
        if item.get("name") == comparison_name
    ]
    if len(matches) != 1:
        return None
    value = matches[0].get(key)
    if not value:
        return None
    path = Path(value)
    if not path.is_file():
        # Evidence prepared on the cluster records absolute cluster paths.
        # Allow the self-contained evidence directory to be copied to another
        # workstation without rewriting its audit manifest.
        copied_path = manifest_path.parent / path.name
        if not copied_path.is_file():
            return None
        path = copied_path
    print(f"Using prepared {comparison_name} render: {path}")
    return read_render_tiff(path)


def evidence_center_render(case: Case, key: str) -> np.ndarray | None:
    """Load a prepared exact-center raster without another remote read."""

    return evidence_comparison_render(case, "exact-center", key)


def load_reports(case: Case) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    stages = [case.updated_stage]
    if case.target_stage is not None:
        stages.append(case.target_stage)
    for stage in stages:
        path = case.results / f"inklabels-{stage}.report.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing pipeline report {path}")
        result[stage] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _registered_render_cache_spec(
    source_tifxyz: Path,
    target_tifxyz: Path,
    source_render_shape: tuple[int, ...],
    target_shape: tuple[int, int],
    matrix: np.ndarray,
    report: dict[str, Any],
    label_offset_render_yx: tuple[float, float],
    fill_seams: bool,
) -> dict[str, Any]:
    return {
        "cache_version": 4,
        "source": str(source_tifxyz),
        "target": str(target_tifxyz),
        "matrix": np.asarray(matrix, dtype=np.float64).tolist(),
        "fill_seams": bool(fill_seams),
        "label_offset_render_yx": [
            float(label_offset_render_yx[0]),
            float(label_offset_render_yx[1]),
        ],
        "source_render_shape": list(source_render_shape),
        "output_shape": list(target_shape),
        "max_distance": float(report["max_distance"]),
        "nearest_vertices": int(report["nearest_vertices"]),
        "tile_size": int(report["tile_size"]),
        "query_batch_size": int(report["query_batch_size"]),
    }


def _load_registered_render_cache(
    case: Case,
    cache_name: str,
    expected: dict[str, Any],
) -> np.ndarray | None:
    cache = case.root / "renders" / cache_name
    cache_report_path = cache.with_suffix(".json")
    if not cache.exists() or not cache_report_path.exists():
        return None
    try:
        cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
        comparable = {key: cache_report.get(key) for key in expected}
        if comparable == expected:
            return read_render_tiff(cache)
    except (OSError, ValueError, TypeError):
        pass
    return None


def register_render(
    case: Case,
    source_tifxyz: Path,
    target_tifxyz: Path,
    source_render: np.ndarray,
    target_shape: tuple[int, int],
    matrix: np.ndarray,
    report: dict[str, Any],
    cache_name: str,
    label_offset_render_yx: tuple[float, float] = (0.0, 0.0),
    fill_seams: bool = False,
) -> np.ndarray:
    """Project a render onto another TIFXYZ canvas.

    ``fill_seams`` continues the mapping across fold-seam rejections the
    same way the transfer outputs do. Use it only for comparison overlays,
    never for projections that feed residual measurements.
    """
    cache = case.root / "renders" / cache_name
    cache_report_path = cache.with_suffix(".json")
    expected_cache_report = _registered_render_cache_spec(
        source_tifxyz,
        target_tifxyz,
        source_render.shape,
        target_shape,
        matrix,
        report,
        label_offset_render_yx,
        fill_seams,
    )
    cached = _load_registered_render_cache(case, cache_name, expected_cache_report)
    if cached is not None:
        return cached
    source_surface = load_full_surface(source_tifxyz)
    target_surface = load_full_surface(target_tifxyz)
    registered, valid, _, stats = transfer_surface_array(
        source_surface,
        target_surface,
        source_render,
        output_shape=target_shape,
        affine=matrix,
        label_offset_yx=(
            float(label_offset_render_yx[0]),
            float(label_offset_render_yx[1]),
        ),
        max_distance=float(report["max_distance"]),
        nearest_vertices=int(report["nearest_vertices"]),
        tile_size=int(report["tile_size"]),
        query_batch_size=int(report["query_batch_size"]),
        fill_seams=fill_seams,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(cache, registered)
    tifffile.imwrite(cache.with_name(f"{cache.stem}.valid.tif"), valid)
    cache_report_path.write_text(
        json.dumps(
            {
                **expected_cache_report,
                "affine": (
                    "pipeline-report-selected matrix"
                    if not np.allclose(matrix, np.eye(4))
                    else "identity"
                ),
                "mapping": stats.as_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Cached TIFXYZ-registered surface render -> {cache}")
    return read_render_tiff(cache)


def report_affine_matrix(
    reports: dict[str, dict[str, Any]], stage: str, case_name: str
) -> np.ndarray:
    matrix = np.asarray(
        reports[stage]["affine"]["matrix"], dtype=np.float64
    )
    if (
        matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])
        or abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-15
    ):
        raise ValueError(
            f"Invalid selected affine matrix in {case_name} {stage} report"
        )
    return matrix


def load_diagnostic_renders(
    case: Case,
    reports: dict[str, dict[str, Any]],
    stage_one_matrix: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load optional cross-evidence and raw self-render comparison pairs."""

    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    manifest_path = case.root / "renders" / "offset-evidence" / "manifest.json"
    if not manifest_path.is_file():
        return pairs
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence_path = case.root / "affines" / "hf-render-canvas-offset-evidence.json"
    evidence_offsets: dict[str, np.ndarray] = {}
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("approved"):
            evidence_offsets = {
                str(item["name"]): np.asarray(
                    item["offset_yx_full_resolution_px"],
                    dtype=np.float64,
                )
                for item in evidence.get("evidence") or []
            }
    shape_value = reports[case.updated_stage].get(
        "source_full_resolution_shape"
    )
    source_full: np.ndarray | None = None
    if shape_value is not None:
        candidate = np.asarray(shape_value, dtype=np.float64)
        if (
            candidate.shape == (2,)
            and np.all(np.isfinite(candidate))
            and np.all(candidate > 0)
        ):
            source_full = candidate
    for comparison in manifest.get("comparisons") or []:
        key = {
            "exact-center": "evidence_center",
            "annotation-matched-slab": "evidence_max",
        }.get(comparison.get("name"))
        if key is None:
            continue
        source_image = evidence_comparison_render(
            case, str(comparison["name"]), "source_render"
        )
        target_image = evidence_comparison_render(
            case, str(comparison["name"]), "target_render"
        )
        if source_image is None or target_image is None:
            continue
        offset_full = evidence_offsets.get(
            str(comparison.get("name")), np.zeros(2, dtype=np.float64)
        )
        if np.any(offset_full):
            if source_full is None:
                print(
                    f"Skipping {comparison.get('name')} evidence pair: the "
                    f"{case.updated_stage} report lacks a valid "
                    "source_full_resolution_shape needed to scale the "
                    "evidence offset"
                )
                continue
            offset_render = (
                offset_full * np.asarray(source_image.shape) / source_full
            )
        else:
            offset_render = np.zeros(2, dtype=np.float64)
        projected = register_render(
            case,
            case.source_tifxyz,
            case.updated_tifxyz,
            source_image,
            target_image.shape,
            stage_one_matrix,
            reports[case.updated_stage],
            f"{key}-source-on-updated.tif",
            label_offset_render_yx=(
                float(offset_render[0]),
                float(offset_render[1]),
            ),
        )
        pairs[key] = (projected, target_image)

    self_report_value = manifest.get("self_render_report")
    if self_report_value:
        self_report_path = Path(self_report_value)
        if self_report_path.is_file():
            self_report = json.loads(
                self_report_path.read_text(encoding="utf-8")
            )
            for key, source_key, target_key in (
                (
                    "self_center",
                    "source_on_target_center",
                    "target_self_center",
                ),
                (
                    "self_max",
                    "source_on_target_matched_max",
                    "target_self_matched_max",
                ),
            ):
                source_path = Path(self_report[source_key])
                target_path = Path(self_report[target_key])
                if source_path.is_file() and target_path.is_file():
                    source_image = read_render_tiff(source_path)
                    target_image = read_render_tiff(target_path)
                    plan_path = self_report_path.with_name("plan.json")
                    plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    bounds = [
                        item["bounds_yxyx"]
                        for item in plan["target_overlap_tiles"]
                    ]
                    pairs[key] = diagnostic_contact_sheet(
                        source_image, target_image, bounds
                    )
    return pairs


def diagnostic_contact_sheet(
    source: np.ndarray,
    target: np.ndarray,
    bounds: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pack sparse evidence tiles so Napari opens on useful content."""

    if not bounds:
        return source, target
    tile_height = max(item[1] - item[0] for item in bounds)
    tile_width = max(item[3] - item[2] for item in bounds)
    columns = int(math.ceil(math.sqrt(len(bounds))))
    rows = int(math.ceil(len(bounds) / columns))
    source_sheet = np.zeros(
        (rows * tile_height, columns * tile_width), dtype=source.dtype
    )
    target_sheet = np.zeros_like(source_sheet, dtype=target.dtype)
    for index, (row0, row1, col0, col1) in enumerate(bounds):
        sheet_row = (index // columns) * tile_height
        sheet_col = (index % columns) * tile_width
        height = row1 - row0
        width = col1 - col0
        source_sheet[
            sheet_row : sheet_row + height,
            sheet_col : sheet_col + width,
        ] = source[row0:row1, col0:col1]
        target_sheet[
            sheet_row : sheet_row + height,
            sheet_col : sheet_col + width,
        ] = target[row0:row1, col0:col1]
    return source_sheet, target_sheet


def prepare_case(
    case: Case,
    max_points: int,
    zarr_level: int,
    include_renders: bool,
    preview_factor: int = 1,
) -> PreparedCase:
    if preview_factor <= 0:
        raise ValueError("preview_factor must be positive")
    source = load_surface(case.source_tifxyz, max_points)
    updated = load_surface(case.updated_tifxyz, max_points)
    target = (
        load_surface(case.target_tifxyz, max_points)
        if case.target_tifxyz is not None
        else None
    )
    reports = load_reports(case)
    updated_stage = case.updated_stage
    target_stage = case.target_stage
    matrix = (
        report_affine_matrix(reports, target_stage, case.name)
        if target_stage is not None
        else None
    )
    stage_one_matrix = report_affine_matrix(
        reports, updated_stage, case.name
    )
    stage_one_offset_full = tuple(
        float(value)
        for value in reports[updated_stage].get(
            "label_canvas_offset_full_resolution_px", (0.0, 0.0)
        )
    )
    source_render_offset_full = stage_one_offset_full
    self_render_validation: dict[str, Any] | None = None
    self_manifest_path = (
        case.root / "renders" / "offset-evidence" / "manifest.json"
    )
    if self_manifest_path.is_file():
        self_manifest = json.loads(
            self_manifest_path.read_text(encoding="utf-8")
        )
        self_report_value = self_manifest.get("self_render_report")
        if self_report_value and Path(self_report_value).is_file():
            self_report = json.loads(
                Path(self_report_value).read_text(encoding="utf-8")
            )
            self_render_validation = {
                "source_raw_volume": self_report.get("source_raw_volume"),
                "target_raw_volume": self_report.get("target_raw_volume"),
                "annotation_canvas_offset": self_report.get(
                    "annotation_canvas_offset"
                ),
                "approval": self_report.get("approval"),
            }
            approval = self_report.get("approval") or {}
            if approval.get("approved"):
                annotation = (
                    self_report.get("annotation_canvas_offset") or {}
                )
                center = self_report.get("center_canvas_check") or {}
                offset_value = (
                    annotation.get("canvas_offset_yx_full_resolution_px")
                    if annotation.get("authoritative")
                    else None
                )
                if offset_value is None:
                    offset_value = center.get(
                        "canvas_offset_yx_full_resolution_px"
                    )
                if (
                    offset_value is not None
                    and len(offset_value) == 2
                    and all(
                        math.isfinite(float(value))
                        for value in offset_value
                    )
                ):
                    source_render_offset_full = tuple(
                        float(value) for value in offset_value
                    )
                else:
                    print(
                        f"{case.name}: approved self-render report lacks "
                        "a usable canvas offset; keeping the stage "
                        "report offset"
                    )
    source_full_shape = reports[updated_stage].get(
        "source_full_resolution_shape"
    )
    if source_full_shape is None and any(stage_one_offset_full):
        raise ValueError(
            f"{case.name} {updated_stage} report declares a label canvas "
            "offset but no source_full_resolution_shape"
        )
    summary = {
        "case": case.name,
        "label_provenance": (
            f"HF {updated_stage} annotation projected to updated "
            f"{updated_stage}"
            + (
                f" and {target_stage}; the {target_stage} output is not "
                "a native annotation"
                if target_stage is not None
                else "; no native-resolution target in this case"
            )
        ),
        "old_to_updated_nearest_vertex": nearest_summary(
            apply_affine(source.points_xyz, stage_one_matrix),
            updated.points_xyz,
        ),
        "old_to_updated_mapping": reports[updated_stage]["mapping"],
        "stage_one_affine": reports[updated_stage]["affine"],
        "label_canvas_offset_full_resolution_px": list(
            stage_one_offset_full
        ),
        "source_render_canvas_offset_full_resolution_px": list(
            source_render_offset_full
        ),
    }
    if self_render_validation is not None:
        summary["self_render_validation"] = self_render_validation
    if target is not None and matrix is not None and target_stage:
        transformed = apply_affine(updated.points_xyz, matrix)
        summary["updated_to_target_nearest_vertex"] = nearest_summary(
            transformed, target.points_xyz
        )
        summary["updated_to_target_mapping"] = reports[target_stage][
            "mapping"
        ]
        summary["affine_direction"] = reports[target_stage]["affine"]
    source_render = None
    renders: dict[str, np.ndarray] = {}
    registered_hf_to_updated_render = None
    registered_source_render = None
    registered_hf_label_to_updated = None
    registered_hf_supervision_to_updated = None
    diagnostics: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if include_renders:
        matched_source_render = evidence_comparison_render(
            case, "annotation-matched-slab", "source_render"
        )
        matched_updated_render = evidence_comparison_render(
            case, "annotation-matched-slab", "target_render"
        )
        if matched_source_render is None or matched_updated_render is None:
            matched_source_render = None
            matched_updated_render = None
        source_render_kind = "matched-max"
        source_cache = (
            case.root
            / "renders"
            / (
                f"hf-original-{updated_stage}-level{zarr_level}"
                "-middle3-max.tif"
            )
        )
        source_render = matched_source_render
        if source_render is None:
            if source_cache.is_file():
                source_render = read_render_tiff(source_cache)
                source_render_kind = "middle3-max"
            else:
                source_render = evidence_center_render(
                    case, "source_render"
                )
                source_render_kind = "center"
        if source_render is None:
            if case.source_surface_volume_url is None:
                raise ValueError(
                    f"{case.name}: no cached source render and no ink "
                    "dataset mirror configured; pass --ink-rclone-root "
                    "to fetch it"
                )
            source_render = load_middle_three_max(
                case.source_surface_volume_url,
                source_cache,
                zarr_level,
            )
            source_render_kind = "middle3-max"
        source_render = preview_render(source_render, preview_factor)
        stage_resolutions = {"updated": case.updated_resolution}
        if (
            case.target_resolution is not None
            and case.target_resolution in case.surface_volume_urls
        ):
            stage_resolutions["target"] = case.target_resolution
        elif case.target_resolution is not None:
            print(
                f"{case.name}: no matching {case.target_resolution:g}um "
                "surface-volume render; updated-to-target CT residual is "
                "unavailable"
            )
        for stage_name, resolution in stage_resolutions.items():
            cache = (
                case.root
                / "renders"
                / f"{resolution:g}um-level{zarr_level}-middle3-max.tif"
            )
            prepared_render = (
                (
                    matched_updated_render
                    if matched_updated_render is not None
                    else evidence_center_render(case, "target_render")
                )
                if stage_name == "updated" and not cache.is_file()
                else None
            )
            renders[stage_name] = preview_render(
                prepared_render
                if prepared_render is not None
                else load_middle_three_max(
                    case.surface_volume_urls[resolution],
                    cache,
                    zarr_level,
                ),
                preview_factor,
            )
        if matched_updated_render is not None:
            # The shipped annotation canvas and this target composite use the
            # same physical slab thickness.  Keep it separate from the
            # updated render used by the updated→final comparison.
            renders["updated_stage_one"] = preview_render(
                matched_updated_render, preview_factor
            )
        if (
            "target" in renders
            and case.target_tifxyz is not None
            and matrix is not None
            and target_stage is not None
        ):
            registered_source_render = register_render(
                case,
                case.updated_tifxyz,
                case.target_tifxyz,
                renders["updated"],
                renders["target"].shape,
                matrix,
                reports[target_stage],
                preview_cache_name(
                    f"{updated_stage}-affine-to-{target_stage}-"
                    f"level{zarr_level}-middle3-max.tif",
                    preview_factor,
                ),
            )
        # The stage-one canvas offset is declared in full-resolution source
        # canvas pixels; scale it to the source render actually projected.
        source_offset_render = (0.0, 0.0)
        if any(source_render_offset_full):
            source_offset_render = (
                source_render_offset_full[0]
                * source_render.shape[0]
                / float(source_full_shape[0]),
                source_render_offset_full[1]
                * source_render.shape[1]
                / float(source_full_shape[1]),
            )
        registered_hf_to_updated_render = register_render(
            case,
            case.source_tifxyz,
            case.updated_tifxyz,
            source_render,
            renders.get("updated_stage_one", renders["updated"]).shape,
            stage_one_matrix,
            reports[updated_stage],
            preview_cache_name(
                "".join(
                    (
                        f"hf-original-{updated_stage}-to-updated-",
                        f"level{zarr_level}-",
                        f"{source_render_kind}.tif",
                    )
                ),
                preview_factor,
            ),
            label_offset_render_yx=source_offset_render,
        )
        # Project the native HF annotations with the same geometry so the
        # CT alignment preset can compare original versus transferred
        # annotations on one canvas.
        for annotation_path, annotation_cache in (
            (case.source_label, "label"),
            (case.source_supervision, "supervision"),
        ):
            if not annotation_path.is_file():
                continue
            cache_name = preview_cache_name(
                f"hf-{annotation_cache}-{updated_stage}-to-updated-"
                f"level{zarr_level}.tif",
                preview_factor,
            )
            cache_spec = _registered_render_cache_spec(
                case.source_tifxyz,
                case.updated_tifxyz,
                source_render.shape,
                renders.get("updated_stage_one", renders["updated"]).shape,
                stage_one_matrix,
                reports[updated_stage],
                source_offset_render,
                True,
            )
            registered_annotation = _load_registered_render_cache(
                case,
                cache_name,
                cache_spec,
            )
            if registered_annotation is None:
                annotation_render = (
                    binary_image(
                        read_tiff_nearest(annotation_path, source_render.shape)
                    )
                    * 255
                )
                registered_annotation = register_render(
                    case,
                    case.source_tifxyz,
                    case.updated_tifxyz,
                    annotation_render,
                    renders.get(
                        "updated_stage_one", renders["updated"]
                    ).shape,
                    stage_one_matrix,
                    reports[updated_stage],
                    cache_name,
                    label_offset_render_yx=source_offset_render,
                    # Comparison overlay: match the seam-filled transfer
                    # outputs so both sides cover the folds.
                    fill_seams=True,
                )
            if annotation_cache == "label":
                registered_hf_label_to_updated = registered_annotation
            else:
                registered_hf_supervision_to_updated = registered_annotation
        # Point-to-surface statistics cannot see tangential frame offsets,
        # so also measure the residual image shift of both projections.
        residual_checks = [
            (
                "old_to_updated",
                renders.get("updated_stage_one", renders["updated"]),
                registered_hf_to_updated_render,
            ),
        ]
        if registered_source_render is not None:
            residual_checks.append(
                (
                    "updated_to_target",
                    renders["target"],
                    registered_source_render,
                )
            )
        if preview_factor == 1:
            summary["render_residual_shift_px"] = {}
            for name, reference, projected in residual_checks:
                try:
                    summary["render_residual_shift_px"][name] = (
                        measure_render_shift(reference, projected)
                    )
                except ValueError as error:
                    summary["render_residual_shift_px"][name] = {
                        "error": str(error)
                    }
            if target is not None and "target" not in renders:
                summary["render_residual_shift_px"][
                    "updated_to_target"
                ] = {
                    "unavailable": (
                        f"no matching {case.target_resolution:g}um native "
                        "surface-volume render"
                    )
                }
            diagnostics = load_diagnostic_renders(
                case, reports, stage_one_matrix
            )
        else:
            summary["preview"] = {
                "downsample_factor": preview_factor,
                "residual_measurement": "skipped",
                "diagnostic_contact_sheets": "skipped",
            }
    surfaces = {"source": source, "updated": updated}
    if target is not None:
        surfaces["target"] = target
    return PreparedCase(
        case,
        summary,
        surfaces,
        matrix,
        source_render,
        renders,
        registered_hf_to_updated_render,
        registered_source_render,
        diagnostics,
        registered_hf_label_to_updated=registered_hf_label_to_updated,
        registered_hf_supervision_to_updated=(
            registered_hf_supervision_to_updated
        ),
        stage_one_affine=stage_one_matrix,
    )


def binary_image(image: np.ndarray) -> np.ndarray:
    return np.asarray(image > 0, dtype=np.uint8)


def masked_surface_points(surface: Surface, mask: np.ndarray) -> np.ndarray:
    if mask.shape != surface.shape:
        mask = resize_nearest(mask, surface.shape)
    selected = np.asarray(mask > 0).ravel()[surface.flat_indices]
    return surface.points_xyz[selected]


def add_image_pair(
    viewer: Any,
    blue_colormap: Any,
    red_colormap: Any,
    prefix: str,
    blue: np.ndarray,
    red: np.ndarray,
    scale: tuple[float, float],
    translate: tuple[float, float],
    visible: bool,
) -> tuple[Any, Any]:
    if blue.shape != red.shape:
        raise ValueError(
            f"Common-canvas comparison {prefix!r} has mismatched shapes: "
            f"source={blue.shape}, target={red.shape}. Project the source "
            "through both TIFXYZ surfaces (and the applicable affine) "
            "before constructing the overlay; a plain resize is not a "
            "geometric registration."
        )
    blue_layer = viewer.add_image(
        blue,
        name=f"{prefix} / BLUE source",
        colormap=blue_colormap,
        blending="additive",
        contrast_limits=(0.0, float(max(1.0, np.max(blue)))),
        scale=scale,
        translate=translate,
        visible=visible,
    )
    red_layer = viewer.add_image(
        red,
        name=f"{prefix} / RED target",
        colormap=red_colormap,
        blending="additive",
        contrast_limits=(0.0, float(max(1.0, np.max(red)))),
        scale=scale,
        translate=translate,
        visible=visible,
    )
    return blue_layer, red_layer


def add_case_layers(
    viewer: Any,
    prepared: PreparedCase,
    _case_index: int,
    vector_count: int,
    colormaps: dict[str, Any],
) -> dict[str, list[Any]]:
    case = prepared.case
    surfaces = prepared.surfaces
    groups: dict[str, list[Any]] = {}
    updated_stage = case.updated_stage
    target_stage = case.target_stage
    source_display_shape = (
        prepared.source_render.shape
        if prepared.source_render is not None
        else surfaces["source"].shape
    )
    updated_display_shape = (
        prepared.renders["updated"].shape
        if "updated" in prepared.renders
        else surfaces["updated"].shape
    )
    updated_label = read_label_display(
        case.results / f"inklabels-{updated_stage}.tif",
        updated_display_shape,
    )
    updated_supervision = read_label_display(
        transferred_result_path(
            case.results, "supervision", updated_stage
        ),
        updated_display_shape,
    )
    updated_validation_path = transferred_result_path(
        case.results, "validation", updated_stage
    )
    updated_validation = (
        read_label_display(updated_validation_path, updated_display_shape)
        if updated_validation_path.is_file()
        else None
    )
    updated_valid = (
        read_tiff_nearest(
            case.results / f"inklabels-{updated_stage}.valid.tif",
            updated_display_shape,
        )
        if prepared.renders
        else None
    )
    target_label = None
    target_supervision = None
    target_validation = None
    target_valid = None
    if target_stage is not None:
        target_display_shape = (
            prepared.renders["target"].shape
            if "target" in prepared.renders
            else surfaces["target"].shape
        )
        target_label = read_label_display(
            case.results / f"inklabels-{target_stage}.tif",
            target_display_shape,
        )
        target_supervision = read_label_display(
            transferred_result_path(
                case.results, "supervision", target_stage
            ),
            target_display_shape,
        )
        target_validation_path = transferred_result_path(
            case.results, "validation", target_stage
        )
        if target_validation_path.is_file():
            target_validation = read_label_display(
                target_validation_path, target_display_shape
            )
        target_valid = (
            read_tiff_nearest(
                case.results / f"inklabels-{target_stage}.valid.tif",
                target_display_shape,
            )
            if "target" in prepared.renders
            else None
        )
    source_label_display = read_label_display(
        case.source_label, source_display_shape
    )
    source_label_surface = resize_nearest(
        source_label_display,
        surfaces["source"].shape,
    )
    source_label_render = (
        source_label_display
        if prepared.source_render is not None
        else None
    )

    source_supervision_display = read_label_display(
        case.source_supervision, source_display_shape
    )
    source_supervision_surface = resize_nearest(
        source_supervision_display,
        surfaces["source"].shape,
    )
    source_supervision_render = (
        source_supervision_display
        if prepared.source_render is not None
        else None
    )
    source_validation_render = None
    if case.source_validation is not None:
        source_validation_display = read_label_display(
            case.source_validation, source_display_shape
        )
        if prepared.source_render is not None:
            source_validation_render = source_validation_display

    display_height = 1_000.0
    # The sidebar shows only one case at a time, so all 2D layers can share
    # one origin. This lets reset_view center every preset consistently.
    translate = (0.0, 0.0)
    if prepared.renders:
        source_render = prepared.source_render
        assert source_render is not None
        assert source_label_render is not None
        assert source_supervision_render is not None
        updated_render = prepared.renders.get(
            "updated_stage_one", prepared.renders["updated"]
        )
        target_render = prepared.renders.get("target")
        assert prepared.registered_hf_to_updated_render is not None
        registered_hf_to_updated = prepared.registered_hf_to_updated_render
        registered_render = prepared.registered_source_render
        source_render_scale = display_height / source_render.shape[0]
        updated_render_scale = display_height / updated_render.shape[0]

        # Add every grayscale CT layer before annotation layers. Napari draws
        # later image layers above earlier ones, so this guarantees that ink
        # and supervision remain visible in every stage preset.
        source_native_layer = viewer.add_image(
            source_render,
            name=f"{case.name} RENDER grayscale native HF {updated_stage}",
            colormap="gray",
            scale=(source_render_scale, source_render_scale),
            translate=translate,
            visible=False,
        )
        updated_native_layer = viewer.add_image(
            updated_render,
            name=f"{case.name} RENDER grayscale updated {updated_stage}",
            colormap="gray",
            scale=(updated_render_scale, updated_render_scale),
            translate=translate,
            visible=False,
        )
        target_render_layer = None
        render_scale = None
        if target_render is not None:
            render_scale = display_height / target_render.shape[0]
            target_render_layer = viewer.add_image(
                target_render,
                name=(
                    f"{case.name} RENDER grayscale native {target_stage}"
                ),
                colormap="gray",
                scale=(render_scale, render_scale),
                translate=translate,
                visible=False,
            )
        groups["render_same"] = list(
            add_image_pair(
                viewer,
                colormaps["blue"],
                colormaps["red"],
                (
                    f"{case.name} RENDER TIFXYZ original→updated "
                    f"{updated_stage}"
                ),
                registered_hf_to_updated,
                updated_render,
                (updated_render_scale, updated_render_scale),
                translate,
                visible=False,
            )
        )
        if target_render is not None and registered_render is not None:
            groups["render_registered"] = list(
                add_image_pair(
                    viewer,
                    colormaps["blue"],
                    colormaps["red"],
                    (
                        f"{case.name} RENDER affine middle3-max "
                        f"{updated_stage}→{target_stage}"
                    ),
                    registered_render,
                    target_render,
                    (render_scale, render_scale),
                    translate,
                    visible=False,
                )
            )
        for diagnostic_key, (blue, red) in prepared.diagnostics.items():
            diagnostic_scale = display_height / red.shape[0]
            groups[diagnostic_key] = list(
                add_image_pair(
                    viewer,
                    colormaps["blue"],
                    colormaps["red"],
                    f"{case.name} {diagnostic_key.replace('_', ' ')}",
                    blue,
                    red,
                    (diagnostic_scale, diagnostic_scale),
                    translate,
                    visible=False,
                )
            )

        source_label_layer = viewer.add_image(
            binary_image(source_label_render),
            name=f"{case.name} LABEL native HF {updated_stage}",
            colormap=colormaps["ink"],
            blending="additive",
            contrast_limits=(0, 1),
            scale=(source_render_scale, source_render_scale),
            translate=translate,
            visible=False,
        )
        source_supervision_layer = viewer.add_image(
            binary_image(source_supervision_render),
            name=f"{case.name} SUPERVISION native HF {updated_stage}",
            colormap=colormaps["supervision"],
            blending="additive",
            contrast_limits=(0, 1),
            opacity=0.45,
            scale=(source_render_scale, source_render_scale),
            translate=translate,
            visible=False,
        )
        source_validation_layer = None
        if source_validation_render is not None:
            source_validation_layer = viewer.add_image(
                binary_image(source_validation_render),
                name=f"{case.name} VALIDATION native HF {updated_stage}",
                colormap=colormaps["orange"],
                blending="additive",
                contrast_limits=(0, 1),
                opacity=0.45,
                scale=(source_render_scale, source_render_scale),
                translate=translate,
                visible=False,
            )
        updated_label_layer = viewer.add_image(
            binary_image(resize_nearest(updated_label, updated_render.shape)),
            name=(
                f"{case.name} LABEL transferred onto updated {updated_stage}"
            ),
            colormap=colormaps["ink"],
            blending="additive",
            contrast_limits=(0, 1),
            scale=(updated_render_scale, updated_render_scale),
            translate=translate,
            visible=False,
        )
        updated_supervision_layer = viewer.add_image(
            binary_image(
                resize_nearest(updated_supervision, updated_render.shape)
            ),
            name=(
                f"{case.name} SUPERVISION transferred onto updated "
                f"{updated_stage}"
            ),
            colormap=colormaps["supervision"],
            blending="additive",
            contrast_limits=(0, 1),
            opacity=0.45,
            scale=(updated_render_scale, updated_render_scale),
            translate=translate,
            visible=False,
        )
        updated_validation_layer = None
        if updated_validation is not None:
            updated_validation_layer = viewer.add_image(
                binary_image(
                    resize_nearest(updated_validation, updated_render.shape)
                ),
                name=(
                    f"{case.name} VALIDATION transferred onto updated "
                    f"{updated_stage}"
                ),
                colormap=colormaps["orange"],
                blending="additive",
                contrast_limits=(0, 1),
                opacity=0.45,
                scale=(updated_render_scale, updated_render_scale),
                translate=translate,
                visible=False,
            )
        # Raw validity values: 255 = measured, 128 = seam-filled (mid
        # brightness), 0 = unmapped. Binarising would hide the distinction.
        updated_valid_layer = viewer.add_image(
            # Loaded directly at display resolution; the native validity TIFF
            # can contain billions of pixels.
            np.asarray(updated_valid),
            name=f"{case.name} VALIDITY old→updated {updated_stage}",
            colormap=colormaps["green"],
            blending="additive",
            contrast_limits=(0, 255),
            opacity=0.45,
            scale=(updated_render_scale, updated_render_scale),
            translate=translate,
            visible=False,
        )
        # Slots stay index-aligned with PRESET_LAYER_LABELS["render_same"];
        # a missing annotation leaves None so the other keeps its button.
        render_same_extra: list[Any] = [None, None]
        if prepared.registered_hf_label_to_updated is not None:
            render_same_extra[0] = (
                viewer.add_image(
                    binary_image(prepared.registered_hf_label_to_updated),
                    name=(
                        f"{case.name} LABEL original HF projected onto "
                        f"updated {updated_stage}"
                    ),
                    colormap=colormaps["green"],
                    blending="additive",
                    contrast_limits=(0, 1),
                    scale=(updated_render_scale, updated_render_scale),
                    translate=translate,
                    visible=False,
                )
            )
        if prepared.registered_hf_supervision_to_updated is not None:
            render_same_extra[1] = (
                viewer.add_image(
                    binary_image(
                        prepared.registered_hf_supervision_to_updated
                    ),
                    name=(
                        f"{case.name} SUPERVISION original HF projected "
                        f"onto updated {updated_stage}"
                    ),
                    colormap=colormaps["orange"],
                    blending="additive",
                    contrast_limits=(0, 1),
                    opacity=0.45,
                    scale=(updated_render_scale, updated_render_scale),
                    translate=translate,
                    visible=False,
                )
            )
        groups["render_same"].extend(
            [updated_label_layer, updated_supervision_layer]
        )
        while render_same_extra and render_same_extra[-1] is None:
            render_same_extra.pop()
        groups["render_same"].extend(render_same_extra)
        groups["aligned_original"] = [
            source_native_layer,
            source_label_layer,
            source_supervision_layer,
        ]
        if source_validation_layer is not None:
            groups["aligned_original"].append(source_validation_layer)
        groups["aligned_updated"] = [
            updated_native_layer,
            updated_label_layer,
            updated_supervision_layer,
        ]
        if updated_validation_layer is not None:
            groups["aligned_updated"].append(updated_validation_layer)
        groups["label_same"] = [updated_native_layer, updated_label_layer]
        groups["supervision_same"] = [
            updated_native_layer,
            updated_supervision_layer,
        ]
        if updated_validation_layer is not None:
            groups["validation_same"] = [
                updated_native_layer,
                updated_validation_layer,
            ]
        groups["validity_same"] = [
            updated_native_layer,
            updated_valid_layer,
        ]
        if target_render_layer is not None:
            target_label_layer = viewer.add_image(
                binary_image(
                    resize_nearest(target_label, target_render.shape)
                ),
                name=(
                    f"{case.name} LABEL projected from {updated_stage} "
                    f"on {target_stage} render"
                ),
                colormap=colormaps["ink"],
                blending="additive",
                contrast_limits=(0, 1),
                scale=(render_scale, render_scale),
                translate=translate,
                visible=False,
            )
            target_supervision_layer = viewer.add_image(
                binary_image(
                    resize_nearest(target_supervision, target_render.shape)
                ),
                name=(
                    f"{case.name} SUPERVISION projected from "
                    f"{updated_stage} on {target_stage} render"
                ),
                colormap=colormaps["supervision"],
                blending="additive",
                contrast_limits=(0, 1),
                opacity=0.45,
                scale=(render_scale, render_scale),
                translate=translate,
                visible=False,
            )
            target_validation_layer = None
            if target_validation is not None:
                target_validation_layer = viewer.add_image(
                    binary_image(
                        resize_nearest(target_validation, target_render.shape)
                    ),
                    name=(
                        f"{case.name} VALIDATION projected from "
                        f"{updated_stage} on {target_stage} render"
                    ),
                    colormap=colormaps["orange"],
                    blending="additive",
                    contrast_limits=(0, 1),
                    opacity=0.45,
                    scale=(render_scale, render_scale),
                    translate=translate,
                    visible=False,
                )
            target_valid_layer = viewer.add_image(
                np.asarray(target_valid),
                name=(
                    f"{case.name} VALIDITY composed "
                    f"old→updated→{target_stage}"
                ),
                colormap=colormaps["green"],
                blending="additive",
                contrast_limits=(0, 255),
                opacity=0.45,
                scale=(render_scale, render_scale),
                translate=translate,
                visible=False,
            )
            groups["aligned_final"] = [
                target_render_layer,
                target_label_layer,
                target_supervision_layer,
            ]
            if target_validation_layer is not None:
                groups["aligned_final"].append(target_validation_layer)
            groups["render_label"] = [
                target_render_layer,
                target_label_layer,
                target_supervision_layer,
            ]
            groups["label_cross"] = [
                target_render_layer,
                target_label_layer,
            ]
            groups["supervision_cross"] = [
                target_render_layer,
                target_supervision_layer,
            ]
            if target_validation_layer is not None:
                groups["validation_cross"] = [
                    target_render_layer,
                    target_validation_layer,
                ]
            groups["validity_cross"] = [
                target_render_layer,
                target_valid_layer,
            ]

    source = surfaces["source"].points_xyz
    registered_source = apply_affine(source, prepared.stage_one_affine)
    updated = surfaces["updated"].points_xyz
    target = (
        surfaces["target"].points_xyz if "target" in surfaces else None
    )
    point_size = 4
    source_surface_layer = viewer.add_points(
        registered_source[:, ::-1],
        name=f"{case.name} 3D REGISTERED / BLUE HF source",
        face_color="blue",
        size=point_size,
        blending="additive",
        visible=False,
    )
    updated_surface_layer = viewer.add_points(
        updated[:, ::-1],
        name=f"{case.name} 3D REGISTERED / RED updated {updated_stage}",
        face_color="red",
        size=point_size,
        blending="additive",
        visible=False,
    )
    groups["surface_same"] = [source_surface_layer, updated_surface_layer]
    transformed = None
    if target is not None and prepared.affine is not None:
        transformed = apply_affine(updated, prepared.affine)
        transformed_surface_layer = viewer.add_points(
            transformed[:, ::-1],
            name=(
                f"{case.name} 3D AFFINE / BLUE transformed {updated_stage}"
            ),
            face_color="blue",
            size=point_size,
            blending="additive",
            visible=False,
        )
        target_surface_layer = viewer.add_points(
            target[:, ::-1],
            name=f"{case.name} 3D AFFINE / RED native {target_stage}",
            face_color="red",
            size=point_size,
            blending="additive",
            visible=False,
        )
        count = min(vector_count, transformed.shape[0])
        indices = np.linspace(
            0, transformed.shape[0] - 1, count, dtype=np.int64
        )
        starts = transformed[indices]
        _, nearest = cKDTree(target).query(starts, workers=-1)
        displacements = target[nearest] - starts
        vectors = np.stack(
            (starts[:, ::-1], displacements[:, ::-1]), axis=1
        )
        residual_layer = viewer.add_vectors(
            vectors,
            name=f"{case.name} 3D AFFINE / residual vectors",
            edge_color="yellow",
            edge_width=0.4,
            visible=False,
        )
        groups["surface_affine"] = [
            transformed_surface_layer,
            target_surface_layer,
            residual_layer,
        ]

    source_label_points = masked_surface_points(
        surfaces["source"], source_label_surface
    )
    source_label_points = apply_affine(
        source_label_points, prepared.stage_one_affine
    )
    updated_label_points = masked_surface_points(
        surfaces["updated"], updated_label
    )
    source_supervision_points = masked_surface_points(
        surfaces["source"], source_supervision_surface
    )
    source_supervision_points = apply_affine(
        source_supervision_points, prepared.stage_one_affine
    )
    updated_supervision_points = masked_surface_points(
        surfaces["updated"], updated_supervision
    )
    point_pairs = [
        (
            "label3d_same",
            "3D LABEL REGISTERED",
            source_label_points,
            updated_label_points,
            f"HF {updated_stage}",
            f"updated {updated_stage}",
        ),
        (
            "supervision3d_same",
            "3D SUPERVISION REGISTERED",
            source_supervision_points,
            updated_supervision_points,
            f"HF {updated_stage}",
            f"updated {updated_stage}",
        ),
    ]
    if "target" in surfaces and prepared.affine is not None:
        target_label_points = masked_surface_points(
            surfaces["target"], target_label
        )
        target_supervision_points = masked_surface_points(
            surfaces["target"], target_supervision
        )
        transformed_label_points = apply_affine(
            updated_label_points, prepared.affine
        )
        transformed_supervision_points = apply_affine(
            updated_supervision_points, prepared.affine
        )
        point_pairs.extend(
            [
                (
                    "label3d_affine",
                    "3D LABEL AFFINE",
                    transformed_label_points,
                    target_label_points,
                    f"transformed {updated_stage}",
                    f"projected {target_stage}",
                ),
                (
                    "supervision3d_affine",
                    "3D SUPERVISION AFFINE",
                    transformed_supervision_points,
                    target_supervision_points,
                    f"transformed {updated_stage}",
                    f"projected {target_stage}",
                ),
            ]
        )
    for (
        preset,
        group,
        blue_points,
        red_points,
        blue_name,
        red_name,
    ) in point_pairs:
        blue_layer = viewer.add_points(
            blue_points[:, ::-1],
            name=f"{case.name} {group} / BLUE {blue_name}",
            face_color="blue",
            size=6,
            blending="additive",
            visible=False,
        )
        red_layer = viewer.add_points(
            red_points[:, ::-1],
            name=f"{case.name} {group} / RED {red_name}",
            face_color="red",
            size=6,
            blending="additive",
            visible=False,
        )
        groups[preset] = [blue_layer, red_layer]
    return groups


def release_case_layers(
    viewer: Any,
    case_groups: dict[str, dict[str, list[Any]]],
) -> None:
    """Release the currently materialised case before preparing another."""

    case_groups.clear()
    viewer.layers.clear()
    # Removing a Napari layer releases Python references immediately, while
    # Qt/Vispy may defer deletion of the corresponding image and GPU texture
    # until the event queue runs.  Cross that release barrier before allocating
    # the next case so only one complete segment is resident at a time.
    try:
        from qtpy.QtWidgets import QApplication
    except ImportError:
        pass
    else:
        application = QApplication.instance()
        if application is not None:
            application.processEvents()
    gc.collect()


def replace_case_layers(
    viewer: Any,
    case_groups: dict[str, dict[str, list[Any]]],
    case: Case,
    prepare: Callable[[Case], PreparedCase],
    add_layers: Callable[[PreparedCase], dict[str, list[Any]]],
) -> dict[str, list[Any]]:
    """Unload the current segment and materialise exactly one selected case."""

    release_case_layers(viewer, case_groups)
    prepared = prepare(case)
    groups = add_layers(prepared)
    case_groups[case.name] = groups
    return groups


def start_case_worker(
    worker: Any,
    on_returned: Callable[[PreparedCase], None],
    on_errored: Callable[[BaseException], None],
) -> Any:
    """Connect a Napari worker before starting asynchronous preparation."""

    worker.returned.connect(on_returned)
    worker.errored.connect(on_errored)
    worker.start()
    return worker


def load_case_in_background(
    viewer: Any,
    case_groups: dict[str, dict[str, list[Any]]],
    case: Case,
    worker_factory: Callable[[Case], Any],
    materialise: Callable[[PreparedCase], dict[str, list[Any]]],
    on_loaded: Callable[[str, dict[str, list[Any]]], None],
    on_errored: Callable[[str, BaseException], None],
    active_workers: dict[str, Any],
) -> Any:
    """Prepare one case off-thread and materialise it from the GUI callback."""

    # This function is called by Qt on the GUI thread. Layer/Vispy cleanup is
    # therefore safe here; the worker only performs prepare_case's I/O and
    # numerical work.
    release_case_layers(viewer, case_groups)
    worker = worker_factory(case)
    active_workers[case.name] = worker

    def returned(prepared: PreparedCase) -> None:
        try:
            groups = materialise(prepared)
        except BaseException as error:
            on_errored(case.name, error)
        else:
            on_loaded(case.name, groups)
        finally:
            active_workers.pop(case.name, None)

    def errored(error: BaseException) -> None:
        active_workers.pop(case.name, None)
        on_errored(case.name, error)

    return start_case_worker(worker, returned, errored)


def add_comparison_panel(
    viewer: Any,
    case_groups: dict[str, dict[str, list[Any]]],
    *,
    case_names: list[str] | None = None,
    case_loader: Callable[
        [
            str,
            Callable[[str, dict[str, list[Any]]], None],
            Callable[[str, BaseException], None],
        ],
        None,
    ]
    | None = None,
) -> Any:
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import (
        QButtonGroup,
        QComboBox,
        QGridLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )

    panel = QWidget()
    panel.setObjectName("comparison_panel")
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(4)

    case_label = QLabel("Segment")
    case_label.setStyleSheet("font-weight: 600;")
    layout.addWidget(case_label)
    case_picker = QComboBox()
    case_picker.addItems(case_names or list(case_groups))
    layout.addWidget(case_picker)

    legend = QLabel(
        "<span style='color:#5a9cff'>Blue source</span> · "
        "<span style='color:#ff5a5a'>Red target</span> · "
        "<span style='color:#d67cff'>Purple match</span>"
    )
    legend.setWordWrap(True)
    layout.addWidget(legend)
    description = QLabel()
    description.setWordWrap(True)
    description.setStyleSheet("font-size: 11px; color: #aeb4be;")
    layout.addWidget(description)

    buttons: dict[str, QPushButton] = {}
    layer_buttons: dict[str, list[QPushButton]] = {}
    layer_rows: dict[str, QWidget] = {}
    button_group = QButtonGroup(panel)
    button_group.setExclusive(True)
    current_section = ""
    for preset in COMPARISON_PRESETS:
        if preset.section != current_section:
            current_section = preset.section
            heading = QLabel(current_section)
            heading.setStyleSheet(
                "font-size: 11px; font-weight: 600; margin-top: 7px;"
            )
            layout.addWidget(heading)
        button = QPushButton(preset.label)
        button.setObjectName(f"preset_{preset.key}")
        button.setCheckable(True)
        button.setToolTip(preset.description)
        button.setMinimumHeight(28)
        layout.addWidget(button)
        button_group.addButton(button)
        buttons[preset.key] = button

        layer_row = QWidget()
        layer_layout = QGridLayout(layer_row)
        layer_layout.setContentsMargins(8, 0, 0, 2)
        layer_layout.setSpacing(4)
        layer_buttons[preset.key] = []
        for layer_index, layer_label in enumerate(
            PRESET_LAYER_LABELS[preset.key]
        ):
            layer_button = QPushButton(layer_label)
            layer_button.setObjectName(
                f"layer_{preset.key}_{layer_index}"
            )
            layer_button.setCheckable(True)
            layer_button.setChecked(True)
            layer_button.setMinimumHeight(23)
            layer_button.setStyleSheet("font-size: 10px; padding: 2px 5px;")
            layer_button.clicked.connect(
                lambda checked,
                selected=preset.key,
                index=layer_index: toggle_layer(
                    selected,
                    index,
                    checked,
                )
            )
            layer_layout.addWidget(
                layer_button,
                layer_index // 3,
                layer_index % 3,
            )
            layer_buttons[preset.key].append(layer_button)
        layer_rows[preset.key] = layer_row
        layer_row.setVisible(False)
        layout.addWidget(layer_row)
    layout.addStretch(1)

    preset_by_key = {preset.key: preset for preset in COMPARISON_PRESETS}
    state = {"preset": "render_registered", "loading": False}

    def center_data(layers: list[Any]) -> None:
        if viewer.dims.ndisplay != 2:
            viewer.fit_to_view(margin=0.08)
            return
        present = [layer for layer in layers if layer is not None]
        extent_layers = [layer for layer in present if layer.visible] or present
        extents = [
            np.asarray(layer.extent.world, dtype=np.float64)[:, -2:]
            for layer in extent_layers
        ]
        # Napari has no public canvas-size accessor. Its own fit_to_view uses
        # this model value, but also includes hidden layers in this viewer.
        canvas_size = np.asarray(viewer._canvas_size, dtype=np.float64)
        if canvas_size.shape != (2,) or np.any(canvas_size <= 0):
            return
        active_center, active_zoom = calculate_2d_camera(
            extents, canvas_size
        )
        camera_center = tuple(viewer.camera.center)
        viewer.camera.center = (*camera_center[:-2], *active_center)
        viewer.camera.zoom = active_zoom

    def toggle_layer(key: str, index: int, checked: bool) -> None:
        if state["preset"] != key:
            show_preset(key)
        layers = case_groups[case_picker.currentText()][key]
        if index < len(layers) and layers[index] is not None:
            layers[index].visible = checked

    def show_preset(key: str, force_frame: bool = False) -> None:
        case_name = case_picker.currentText()
        available = case_groups[case_name]
        if key not in available:
            key = next(
                preset.key
                for preset in COMPARISON_PRESETS
                if preset.key in available
            )
        previous_key = state["preset"]
        state["preset"] = key
        for layer in viewer.layers:
            layer.visible = False
        for row in layer_rows.values():
            row.setVisible(False)
        layer_rows[key].setVisible(True)
        preset_layers = available[key]
        if len(preset_layers) > len(layer_buttons[key]):
            raise ValueError(
                f"{key} has {len(preset_layers)} layers but only "
                f"{len(layer_buttons[key])} configured buttons"
            )
        for index, layer_button in enumerate(layer_buttons[key]):
            has_layer = (
                index < len(preset_layers)
                and preset_layers[index] is not None
            )
            layer_button.setVisible(has_layer)
            if has_layer:
                preset_layers[index].visible = layer_button.isChecked()
        buttons[key].setChecked(True)
        preset = preset_by_key[key]
        previous_preset = preset_by_key[previous_key]
        description.setText(preset.description)
        if key == "render_same":
            legend.setText(
                "<span style='color:#5a9cff'>Blue original CT</span> · "
                "<span style='color:#ff5a5a'>Red updated CT</span> · "
                "<span style='color:#ff5aff'>Magenta transferred ink</span> · "
                "<span style='color:#5affff'>Cyan transferred mask</span> · "
                "<span style='color:#5aff5a'>Green HF ink</span> · "
                "<span style='color:#ffb45a'>Orange HF mask</span>"
            )
        elif key in {
            "render_label",
            "aligned_original",
            "aligned_updated",
            "aligned_final",
            "label_same",
            "supervision_same",
            "validation_same",
            "label_cross",
            "supervision_cross",
            "validation_cross",
        }:
            legend.setText(
                "<span style='color:#d8d8d8'>Gray CT</span> · "
                "<span style='color:#ff5aff'>Magenta ink</span> · "
                "<span style='color:#5affff'>Cyan supervision</span> · "
                "<span style='color:#ffb45a'>Orange validation</span>"
            )
        elif key in {"validity_same", "validity_cross"}:
            legend.setText(
                "<span style='color:#d8d8d8'>Gray CT</span> · "
                "<span style='color:#5aff5a'>Green accepted mapping</span>"
            )
        else:
            legend.setText(
                "<span style='color:#5a9cff'>Blue source</span> · "
                "<span style='color:#ff5a5a'>Red target</span> · "
                "<span style='color:#d67cff'>Purple match</span>"
            )
        if viewer.dims.ndisplay != preset.ndisplay:
            viewer.dims.ndisplay = preset.ndisplay
        if should_refit_camera(
            previous_preset.ndisplay,
            preset.ndisplay,
            force_frame,
        ):
            center_data(available[key])

    def set_loading(loading: bool, case_name: str = "") -> None:
        state["loading"] = loading
        case_picker.setEnabled(not loading)
        for button in buttons.values():
            button.setEnabled(not loading)
        if loading:
            description.setText(f"Loading {case_name}…")

    def frame_initial_view() -> None:
        case_name = case_picker.currentText()
        if case_name not in case_groups:
            return
        active_layers = case_groups[case_name][state["preset"]]
        center_data(active_layers)

    def activate_case(case_name: str) -> None:
        available = case_groups[case_name]
        for key, button in buttons.items():
            button.setEnabled(key in available)
        show_preset(state["preset"], force_frame=True)
        # Let Qt/Napari finish slicing the newly added layers before fitting.
        QTimer.singleShot(100, frame_initial_view)
        QTimer.singleShot(350, frame_initial_view)

    def case_loaded(
        case_name: str,
        _available: dict[str, list[Any]],
    ) -> None:
        set_loading(False)
        if case_name == case_picker.currentText():
            activate_case(case_name)

    def case_failed(case_name: str, error: BaseException) -> None:
        set_loading(False)
        description.setText(f"Failed to load {case_name}: {error}")
        print(f"Failed to load {case_name}: {error}", file=sys.stderr)

    def update_case(_case_name: str) -> None:
        if state["loading"]:
            return
        case_name = case_picker.currentText()
        if case_name not in case_groups:
            if case_loader is None:
                raise KeyError(f"Case has not been loaded: {case_name}")
            set_loading(True, case_name)
            case_loader(case_name, case_loaded, case_failed)
            return
        activate_case(case_name)

    for key, button in buttons.items():
        button.clicked.connect(
            lambda _checked=False, selected=key: show_preset(selected)
        )
    case_picker.currentTextChanged.connect(update_case)

    dock = viewer.window.add_dock_widget(
        panel,
        name="Comparisons",
        area="right",
    )
    dock.setMinimumWidth(230)
    dock.setMaximumWidth(300)
    update_case(case_picker.currentText())

    def size_and_center_window() -> None:
        window = panel.window()
        screen = window.screen()
        if screen is None:
            return
        available = screen.availableGeometry()
        width = max(1_100, round(available.width() * 0.9))
        height = max(700, round(available.height() * 0.9))
        window.resize(
            min(width, available.width()),
            min(height, available.height()),
        )
        frame = window.frameGeometry()
        frame.moveCenter(available.center())
        window.move(frame.topLeft())

    QTimer.singleShot(0, size_and_center_window)
    if case_picker.currentText() in case_groups:
        # Let Qt/Napari process the resized canvas and layer slicing before
        # fitting. An immediate fit uses the old canvas size and can make the
        # data tiny.
        QTimer.singleShot(100, frame_initial_view)
        QTimer.singleShot(350, frame_initial_view)
    return panel


def build_parser() -> argparse.ArgumentParser:
    parser = HyphenUnderscoreParser(description=__doc__)
    parser.add_argument(
        "--case-dir",
        type=Path,
        action="append",
        required=True,
        help="Downloaded case directory (repeatable)",
    )
    parser.add_argument("--max-points", type=int, default=100_000)
    parser.add_argument("--vectors", type=int, default=1_000)
    parser.add_argument(
        "--zarr-level",
        type=int,
        default=2,
        help="Preferred surface-volume pyramid level (default: 2)",
    )
    parser.add_argument(
        "--preview-factor",
        type=int,
        default=1,
        help=(
            "Downsample CT comparisons by this integer factor and skip "
            "residual/contact-sheet diagnostics for a faster visual pass "
            "(default: 1, full resolution)"
        ),
    )
    parser.add_argument(
        "--ink-rclone-root",
        default=DEFAULT_INK_ROOT,
        help=(
            "rclone remote:path of your mirror of the private ink dataset; "
            "needed only when a source CT render must be fetched remotely"
        ),
    )
    parser.add_argument(
        "--open-data-rclone-root",
        default=DEFAULT_OPEN_DATA_ROOT,
        help=(
            "rclone remote:path of the public Vesuvius open-data bucket "
            "(default: anonymous inline S3 remote, no rclone config needed)"
        ),
    )
    parser.add_argument(
        "--skip-renders",
        action="store_true",
        help="Do not fetch/cache surface-volume middle-layer composites",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Print summaries and validate render access without importing Napari",
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Render one PNG and exit instead of starting the event loop",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preview_factor <= 0:
        raise ValueError("--preview-factor must be positive")
    cases: list[Case] = []
    for path in args.case_dir:
        case = discover_case(
            path.resolve(),
            ink_rclone_root=args.ink_rclone_root,
            open_data_rclone_root=args.open_data_rclone_root,
        )
        if any(existing.name == case.name for existing in cases):
            raise ValueError(f"Duplicate case name: {case.name}")
        cases.append(case)
    if args.validate_only:
        for case in cases:
            item = prepare_case(
                case,
                args.max_points,
                args.zarr_level,
                include_renders=not args.skip_renders,
                preview_factor=args.preview_factor,
            )
            print(json.dumps(item.summary, indent=2))
        return 0

    try:
        import napari
        from napari.qt.threading import thread_worker
        from napari.utils.colormaps import Colormap
    except ImportError as error:
        raise RuntimeError(
            "Napari is not installed. Install it separately as documented "
            "in the vesuvius.tifxyz_label_transfer README."
        ) from error

    colormaps = {
        "blue": Colormap(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
            name="source-blue",
        ),
        "red": Colormap(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]],
            name="target-red",
        ),
        "green": Colormap(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
            name="valid-green",
        ),
        "ink": Colormap(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0]],
            name="ink-magenta",
        ),
        "supervision": Colormap(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
            name="supervision-cyan",
        ),
        "orange": Colormap(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.6, 0.0, 1.0]],
            name="original-orange",
        ),
    }
    viewer = napari.Viewer(title="TIFXYZ alignment: blue + red = purple")
    case_groups: dict[str, dict[str, list[Any]]] = {}
    cases_by_name = {case.name: case for case in cases}

    def prepare(item: Case) -> PreparedCase:
        return prepare_case(
            item,
            args.max_points,
            args.zarr_level,
            include_renders=not args.skip_renders,
            preview_factor=args.preview_factor,
        )

    def materialise(prepared: PreparedCase) -> dict[str, list[Any]]:
        print(json.dumps(prepared.summary, indent=2))
        groups = add_case_layers(
            viewer,
            prepared,
            0,
            args.vectors,
            colormaps,
        )
        case_groups[prepared.case.name] = groups
        return groups

    def load_case_synchronously(case_name: str) -> dict[str, list[Any]]:
        return replace_case_layers(
            viewer,
            case_groups,
            cases_by_name[case_name],
            prepare,
            materialise,
        )

    @thread_worker
    def prepare_in_background(item: Case) -> PreparedCase:
        return prepare(item)

    active_workers: dict[str, Any] = {}

    def load_case_asynchronously(
        case_name: str,
        on_loaded: Callable[[str, dict[str, list[Any]]], None],
        on_errored: Callable[[str, BaseException], None],
    ) -> None:
        load_case_in_background(
            viewer,
            case_groups,
            cases_by_name[case_name],
            prepare_in_background,
            materialise,
            on_loaded,
            on_errored,
            active_workers,
        )

    viewer.dims.axis_labels = ("z", "y", "x")
    if args.screenshot is not None:
        load_case_synchronously(cases[0].name)
    add_comparison_panel(
        viewer,
        case_groups,
        case_names=[case.name for case in cases],
        case_loader=(
            None if args.screenshot is not None else load_case_asynchronously
        ),
    )

    @viewer.bind_key("2")
    def show_2d(bound_viewer: Any) -> None:
        bound_viewer.dims.ndisplay = 2
        bound_viewer.reset_view()

    @viewer.bind_key("3")
    def show_3d(bound_viewer: Any) -> None:
        bound_viewer.dims.ndisplay = 3
        bound_viewer.reset_view()

    if args.screenshot is not None:
        from qtpy.QtTest import QTest
        from qtpy.QtWidgets import QApplication

        QTest.qWait(450)
        QApplication.processEvents()
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        viewer.screenshot(path=args.screenshot, canvas_only=False)
        print(f"Wrote {args.screenshot}")
        viewer.close()
        return 0
    print(
        "One viewer opened. Toggle paired BLUE/RED layers together; purple "
        "indicates overlap. Press 2 for 2D and 3 for 3D."
    )
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
