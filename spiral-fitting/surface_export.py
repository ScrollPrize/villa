"""New surfaces as the fit exports them: its metadata on a ``vc_reader.tifxyz.Surface``."""

import json
import os
import shutil
import tempfile

import numpy as np
from vc_reader import tifxyz

from surface_orientation import METADATA_KEY as ORIENTATION_METADATA_KEY, GridLayout, export_metadata


def new_surface(xyz, step_size, voxel_size_um, source, /, *, layout_metadata=None, **extra):
    """A surface with a new identity from an XYZ grid (-1 where missing) of cells ``step_size`` voxels apart.

    ``layout_metadata`` describes the export layout of the supplied grid.
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    valid = tifxyz.Surface(xyz).valid
    valid_quad = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    area_vx2 = int(valid_quad.sum()) * step_size ** 2
    metadata = {
        'area_vx2': area_vx2,
        'area_cm2': area_vx2 * voxel_size_um ** 2 / 1.e8 if voxel_size_um else None,
        'source': source,
        **extra,
    }
    if layout_metadata is not None:
        metadata[ORIENTATION_METADATA_KEY] = layout_metadata
    return tifxyz.Surface(xyz, scale=(1 / step_size, 1 / step_size), extra=metadata)


def save_combined_tifxyz(
    winding_zyxs,
    path,
    uuid,
    step_size,
    voxel_size_um,
    source,
    *,
    z_direction_is_top_to_bottom,
    first_winding=10,
    cleanup_erosion_cells=None,
    base_shape_zyx=None,
):
    """Atomically write consecutive winding grids as one QuadSurface.

    ``winding_zyxs`` is a mapping from integer winding id to equally tall
    ``[z, theta, 3]`` ZYX float arrays in spiral order. They are laid side by
    side innermost first and the whole grid is then put in the export layout
    (surface_orientation) for ``z_direction_is_top_to_bottom``, so the
    outermost winding comes first. Winding ranges use
    half-open column bounds, in the written grid, for display filtering, but
    they are not QuadSurface components: adjacent windings remain joined by
    quads across their shared seam. ``winding_column_ranges`` pairs with
    ``component_winding_ids`` and runs left to right.
    """
    if not winding_zyxs:
        raise ValueError("No winding grids were supplied")
    if base_shape_zyx is not None:
        if (not isinstance(base_shape_zyx, (list, tuple))
                or len(base_shape_zyx) != 3
                or any(type(value) is not int or value <= 0
                       for value in base_shape_zyx)):
            raise ValueError(
                "base_shape_zyx must be a ZYX list of three positive integers")
        base_shape_zyx = [int(value) for value in base_shape_zyx]
    by_id = {int(key): np.asarray(value, dtype=np.float32) for key, value in winding_zyxs.items()}
    last_winding = max(by_id)
    if last_winding < int(first_winding):
        raise ValueError(
            f"No preview winding is at or above {int(first_winding)} (last={last_winding})"
        )
    ids = list(range(int(first_winding), last_winding + 1))
    missing = [winding for winding in ids if winding not in by_id]
    if missing:
        raise ValueError(f"Missing preview winding grids: {missing}")

    rows = None
    blocks = []
    components = []
    cursor = 0
    for index, winding in enumerate(ids):
        block = by_id[winding]
        if block.ndim != 3 or block.shape[-1] != 3 or block.shape[1] < 2:
            raise ValueError(f"Winding {winding} must have shape [rows, cols>=2, 3]")
        if rows is None:
            rows = block.shape[0]
        if block.shape[0] != rows:
            raise ValueError("All winding grids must have the same row count")
        if not np.isfinite(block[block != -1]).all():
            raise ValueError(f"Winding {winding} contains non-finite coordinates")
        start = cursor
        blocks.append(block)
        cursor += block.shape[1]
        components.append([start, cursor])
    combined = np.concatenate(blocks, axis=1)
    cleanup_metadata = None
    if cleanup_erosion_cells is not None:
        import scipy.ndimage

        erosion_cells = int(cleanup_erosion_cells)
        if erosion_cells < 0:
            raise ValueError("cleanup_erosion_cells must be non-negative")
        valid = np.isfinite(combined).all(axis=-1) & ~np.all(combined == -1.0, axis=-1)
        if erosion_cells:
            valid = scipy.ndimage.binary_erosion(
                valid, iterations=erosion_cells, border_value=0)
        labels, component_count = scipy.ndimage.label(
            valid, structure=scipy.ndimage.generate_binary_structure(2, 1))
        if component_count == 0:
            raise ValueError(
                "Preview cleanup removed every valid TIFXYZ vertex "
                f"after {erosion_cells}-cell erosion")
        component_sizes = np.bincount(labels.ravel())
        component_sizes[0] = 0
        valid = labels == int(np.argmax(component_sizes))
        combined = combined.copy()
        combined[~valid] = -1.0
        cleanup_metadata = {
            "erosion_cells": erosion_cells,
            "component_connectivity": 4,
            "components_after_erosion": int(component_count),
        }
    width = combined.shape[1]
    combined = GridLayout.export(z_direction_is_top_to_bottom).apply(combined)
    # Reversing the columns puts the windings outermost first.
    components = [[width - end, width - begin] for begin, end in reversed(components)]
    component_ids = ids[::-1]
    destination = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temp_root = tempfile.mkdtemp(prefix=f".{os.path.basename(destination)}.", dir=parent)
    try:
        extra = {
            "voxel_size_um": float(voxel_size_um),
            "winding_column_ranges": components,
            "component_winding_ids": component_ids,
            "output_first_winding": ids[0],
            "output_last_winding": ids[-1],
        }
        if base_shape_zyx is not None:
            extra["base_shape_zyx"] = base_shape_zyx
        if cleanup_metadata is not None:
            extra["lasagna_input_cleanup"] = cleanup_metadata
        # ``uuid`` is the directory name (manifest surface_id); the surface's
        # identity is a fresh random UUID.
        tifxyz.write(
            os.path.join(temp_root, uuid),
            new_surface(combined[..., ::-1], step_size, voxel_size_um, source,
                        layout_metadata=export_metadata(z_direction_is_top_to_bottom), **extra),
            compression="none", tile=None, mask=False)

        published = {
            "schema_version": 2,
            "kind": "spiral_combined_preview",
            "surface_path": os.path.join(destination, uuid),
            "surface_id": uuid,
            "first_winding": ids[0],
            "last_winding": ids[-1],
            "winding_column_ranges": components,
            "winding_ids": component_ids,
            "voxel_size_um": float(voxel_size_um),
            ORIENTATION_METADATA_KEY: export_metadata(z_direction_is_top_to_bottom),
            "manifest_path": os.path.join(destination, "manifest.json"),
        }
        if base_shape_zyx is not None:
            published["base_shape_zyx"] = base_shape_zyx
        with open(os.path.join(temp_root, "manifest.json"), "w", encoding="utf-8") as stream:
            json.dump(published, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())

        # Publish the generation directory only after all files and metadata are complete.
        if os.path.exists(destination):
            raise FileExistsError(destination)
        os.replace(temp_root, destination)
        temp_root = ""
        try:
            parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError:
            pass
        return published
    finally:
        if temp_root:
            shutil.rmtree(temp_root, ignore_errors=True)
