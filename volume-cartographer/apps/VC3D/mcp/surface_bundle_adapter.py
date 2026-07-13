#!/usr/bin/env python3
"""Create and render bounded VC surface bundles from trusted local artifacts.

This fixed adapter never opens network resources. Remote/local volume access is
performed separately by volume_stager.py and supplied here as a bounded NPY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

MAX_SURFACE_PIXELS = 16_777_216
MAX_UV_SIDE = 8192
MAX_VOLUME_VOXELS = 16_777_216
MAX_ALIGNMENT_PIXELS = 1_048_576
MAX_SURFACE_VOLUME_VALUES = 67_108_864


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def read_tifxyz(surface: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    import tifffile

    if not surface.is_dir():
        raise ValueError("surface artifact must be a TIFXYZ directory")
    required = [surface / name for name in ("x.tif", "y.tif", "z.tif", "meta.json")]
    if not all(path.is_file() for path in required):
        raise ValueError("TIFXYZ surface requires x.tif, y.tif, z.tif, and meta.json")
    bands = [np.asarray(tifffile.imread(path)) for path in required[:3]]
    if any(band.ndim != 2 or band.shape != bands[0].shape for band in bands):
        raise ValueError("TIFXYZ coordinate bands must be matching rank-2 arrays")
    height, width = bands[0].shape
    if height < 1 or width < 1 or height > MAX_UV_SIDE or width > MAX_UV_SIDE or height * width > MAX_SURFACE_PIXELS:
        raise ValueError("TIFXYZ surface exceeds bounded UV limits")
    xyz = np.stack(bands, axis=-1).astype(np.float32, copy=False)
    valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0)
    mask_path = surface / "mask.tif"
    if mask_path.is_file():
        mask = np.asarray(tifffile.imread(mask_path))
        if mask.shape != valid.shape:
            raise ValueError("TIFXYZ mask dimensions do not match coordinates")
        valid &= mask != 0
    return xyz, valid, read_json(surface / "meta.json")


def crop_surface(xyz: np.ndarray, valid: np.ndarray, request: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    region = request.get("uv_region")
    if region is None:
        region = {"u": 0, "v": 0, "width": xyz.shape[1], "height": xyz.shape[0]}
    u, v, width, height = (int(region[key]) for key in ("u", "v", "width", "height"))
    if min(u, v) < 0 or min(width, height) < 1 or max(width, height) > MAX_UV_SIDE or width * height > MAX_SURFACE_PIXELS:
        raise ValueError("invalid or excessive UV region")
    if v + height > xyz.shape[0] or u + width > xyz.shape[1]:
        raise ValueError("UV region is outside the TIFXYZ surface")
    return xyz[v : v + height, u : u + width].copy(), valid[v : v + height, u : u + width].copy(), dict(region)


def volume_region(xyz: np.ndarray, valid: np.ndarray, padding: int) -> dict:
    points = xyz[valid]
    if points.size == 0:
        raise ValueError("selected UV region contains no valid surface coordinates")
    minimum = np.floor(points.min(axis=0)).astype(np.int64) - padding
    # The extra high-side voxel is the interpolation halo used by Phase 2
    # trilinear normal-profile sampling.
    maximum = np.ceil(points.max(axis=0)).astype(np.int64) + padding + 2
    minimum = np.maximum(minimum, 0)
    size = maximum - minimum
    if np.any(size < 1) or np.any(size > 256) or int(np.prod(size)) > MAX_VOLUME_VOXELS:
        raise ValueError(
            "surface volume bounds exceed 256 per XYZ axis or 16M voxels; submit a smaller uv_region"
        )
    return {
        "x": int(minimum[0]),
        "y": int(minimum[1]),
        "z": int(minimum[2]),
        "width": int(size[0]),
        "height": int(size[1]),
        "depth": int(size[2]),
    }


def normalize_preview(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros(values.shape, dtype=np.uint8)
    selected = values[valid & np.isfinite(values)]
    if selected.size == 0:
        return output
    low, high = np.percentile(selected, [1.0, 99.0])
    if high <= low:
        high = low + 1.0
    output[valid] = np.clip((values[valid] - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return output


def write_png(path: Path, values: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(values).save(path)


def import_surface(request: dict, surface: Path, output: Path) -> dict:
    import zarr

    xyz, valid, source_meta = read_tifxyz(surface)
    xyz, valid, uv_region = crop_surface(xyz, valid, request)
    padding = int(request.get("normal_padding_voxels", 2))
    if padding < 0 or padding > 64:
        raise ValueError("normal_padding_voxels must be from 0 to 64")
    required_region = volume_region(xyz, valid, padding)

    bundle = output / "surface.zarr"
    root = zarr.open_group(str(bundle), mode="w", zarr_format=2)
    geometry = root.require_group("geometry")
    chunks_2d = (min(256, xyz.shape[0]), min(256, xyz.shape[1]))
    geometry.create_array("xyz", data=xyz, chunks=(*chunks_2d, 3))
    geometry.create_array("valid", data=valid.astype(np.uint8), chunks=chunks_2d)
    root.attrs.update(
        {
            "kind": "vc_surface_bundle",
            "version": 1,
            "surface_representation": "regular_uv_grid",
            "geometry_axes": ["v", "u", "xyz"],
            "coordinate_order": ["x", "y", "z"],
            "coordinate_space": request["coordinate_space"],
            "uv_region": uv_region,
            "source_surface_artifact": request["surface"],
            "source_tifxyz_sha256": {name: sha256(surface / name) for name in ("x.tif", "y.tif", "z.tif", "meta.json")},
            "source_meta": source_meta,
            "triangulation": {"kind": "regular_grid", "diagonal": "u_plus_v_plus", "skip_invalid_cells": True},
            "normal_padding_voxels": padding,
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    valid_preview = valid.astype(np.uint8) * 255
    depth_preview = normalize_preview(xyz[..., 2], valid)
    write_png(output / "surface-validity.png", valid_preview)
    write_png(output / "surface-depth.png", depth_preview)
    manifest = {
        "kind": "vc_registered_surface_v1",
        "score_semantics": "registered_measurement_not_geometry_truth",
        "surface_bundle": str(bundle),
        "surface_shape_vu": list(valid.shape),
        "valid_surface_pixels": int(valid.sum()),
        "uv_region": uv_region,
        "coordinate_space": request["coordinate_space"],
        "geometry_axes": ["v", "u", "xyz"],
        "required_volume_region_xyz": required_region,
        "normal_padding_voxels": padding,
        "surface_validity_preview": "surface-validity.png",
        "surface_depth_preview": "surface-depth.png",
    }
    (output / "import-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def render_surface(request: dict, output: Path) -> dict:
    import tifffile
    import zarr

    bundle = Path(request["surface_bundle"])
    root = zarr.open_group(str(bundle), mode="a")
    xyz = np.asarray(root["geometry/xyz"])
    valid = np.asarray(root["geometry/valid"]) != 0
    volume = np.load(request["staged_volume"], allow_pickle=False)
    if volume.ndim != 3 or volume.size > MAX_VOLUME_VOXELS:
        raise ValueError("staged volume must be a bounded rank-3 ZYX array")
    region = request["staged_region_xyz"]
    indices = np.rint(xyz).astype(np.int64)
    indices[..., 0] -= int(region["x"])
    indices[..., 1] -= int(region["y"])
    indices[..., 2] -= int(region["z"])
    inside = (
        valid
        & (indices[..., 0] >= 0)
        & (indices[..., 1] >= 0)
        & (indices[..., 2] >= 0)
        & (indices[..., 0] < volume.shape[2])
        & (indices[..., 1] < volume.shape[1])
        & (indices[..., 2] < volume.shape[0])
    )
    rendered = np.zeros(valid.shape, dtype=volume.dtype)
    rendered[inside] = volume[indices[..., 2][inside], indices[..., 1][inside], indices[..., 0][inside]]
    renders = root.require_group("renders")
    chunks = (min(256, rendered.shape[0]), min(256, rendered.shape[1]))
    if "raw" in renders:
        del renders["raw"]
    renders.create_array("raw", data=rendered, chunks=chunks)
    root.attrs["registered_volume"] = {
        "source": request["volume_source"],
        "array_path": request["array_path"],
        "scale": request["scale"],
        "voxel_spacing": request["voxel_spacing"],
        "voxel_spacing_unit": request.get("voxel_spacing_unit", "um"),
        "voxel_spacing_explicit": bool(request.get("voxel_spacing_explicit", False)),
        "origin_xyz": request["origin_xyz"],
        "staged_region_xyz": region,
        "sampling": "nearest",
    }
    np.save(output / "registered-intensity.npy", rendered, allow_pickle=False)
    tifffile.imwrite(output / "registered-intensity.tif", rendered)
    preview = normalize_preview(rendered.astype(np.float32), inside)
    write_png(output / "registered-intensity.png", preview)
    coverage = inside.astype(np.uint8) * 255
    write_png(output / "registered-coverage.png", coverage)
    manifest = read_json(output / "import-manifest.json")
    manifest.update(
        {
            "volume": root.attrs["registered_volume"],
            "sampling": "nearest",
            "registered_pixels": int(inside.sum()),
            "coverage_fraction": float(inside.sum() / max(1, valid.sum())),
            "registered_intensity": {
                "zarr_array": "renders/raw",
                "npy": "registered-intensity.npy",
                "tiff": "registered-intensity.tif",
                "preview": "registered-intensity.png",
                "coverage_preview": "registered-coverage.png",
            },
            "artifacts": [
                "surface.zarr",
                "registered-intensity.npy",
                "registered-intensity.tif",
                "registered-intensity.png",
                "registered-coverage.png",
                "surface-validity.png",
                "surface-depth.png",
            ],
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def surface_normals(xyz: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic V×U×XYZ unit normals and their support mask."""
    if xyz.shape[0] < 2 or xyz.shape[1] < 2:
        raise ValueError("surface diagnostics require at least a 2x2 UV region")
    du = np.zeros_like(xyz, dtype=np.float32)
    dv = np.zeros_like(xyz, dtype=np.float32)
    du[:, 1:-1] = (xyz[:, 2:] - xyz[:, :-2]) * 0.5
    du[:, 0] = xyz[:, 1] - xyz[:, 0]
    du[:, -1] = xyz[:, -1] - xyz[:, -2]
    dv[1:-1] = (xyz[2:] - xyz[:-2]) * 0.5
    dv[0] = xyz[1] - xyz[0]
    dv[-1] = xyz[-1] - xyz[-2]
    support = valid.copy()
    support[:, 1:] &= valid[:, :-1]
    support[:, :-1] &= valid[:, 1:]
    support[1:] &= valid[:-1]
    support[:-1] &= valid[1:]
    normals = np.cross(du, dv)
    lengths = np.linalg.norm(normals, axis=-1)
    support &= np.isfinite(lengths) & (lengths > 1e-8)
    normals[~support] = 0
    normals[support] /= lengths[support, None]
    return normals.astype(np.float32, copy=False), support


def save_diagnostic(output: Path, name: str, values: np.ndarray, valid: np.ndarray) -> dict:
    import tifffile

    np.save(output / f"{name}.npy", values, allow_pickle=False)
    tifffile.imwrite(output / f"{name}.tif", values)
    preview = normalize_preview(values.astype(np.float32), valid)
    write_png(output / f"{name}.png", preview)
    return {"npy": f"{name}.npy", "tiff": f"{name}.tif", "preview": f"{name}.png"}


def geometry_diagnostics(request: dict, artifact: Path, output: Path) -> dict:
    from scipy import ndimage
    import zarr

    source_bundle = artifact / "surface.zarr"
    root = zarr.open_group(str(source_bundle), mode="r")
    xyz = np.asarray(root["geometry/xyz"], dtype=np.float32)
    valid = np.asarray(root["geometry/valid"]) != 0
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or valid.shape != xyz.shape[:2]:
        raise ValueError("registered surface artifact has an invalid geometry layout")
    normals, normal_support = surface_normals(xyz, valid)

    edge_u = np.zeros(valid.shape, np.float32)
    edge_v = np.zeros(valid.shape, np.float32)
    edge_u[:, 1:] = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)
    edge_v[1:] = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    support_u = valid.copy(); support_u[:, 0] = False; support_u[:, 1:] &= valid[:, :-1]
    support_v = valid.copy(); support_v[0] = False; support_v[1:] &= valid[:-1]
    edges_u = edge_u[support_u]; edges_u = edges_u[np.isfinite(edges_u) & (edges_u > 1e-8)]
    edges_v = edge_v[support_v]; edges_v = edges_v[np.isfinite(edges_v) & (edges_v > 1e-8)]
    median_edge_u = float(np.median(edges_u)) if edges_u.size else 1.0
    median_edge_v = float(np.median(edges_v)) if edges_v.size else 1.0
    stretch = np.zeros(valid.shape, np.float32)
    stretch_support = support_u & support_v
    stretch[stretch_support] = np.maximum(
        np.abs(np.log(np.maximum(edge_u[stretch_support], 1e-8) / median_edge_u)),
        np.abs(np.log(np.maximum(edge_v[stretch_support], 1e-8) / median_edge_v)),
    )

    normal_change = np.zeros(valid.shape, np.float32)
    horizontal = normal_support[:, 1:] & normal_support[:, :-1]
    vertical = normal_support[1:] & normal_support[:-1]
    if horizontal.any():
        dots = np.clip(np.sum(normals[:, 1:] * normals[:, :-1], axis=-1), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots)).astype(np.float32)
        normal_change[:, 1:][horizontal] = angles[horizontal]
        normal_change[:, :-1] = np.maximum(normal_change[:, :-1], np.where(horizontal, angles, 0))
    if vertical.any():
        dots = np.clip(np.sum(normals[1:] * normals[:-1], axis=-1), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots)).astype(np.float32)
        normal_change[1:][vertical] = np.maximum(normal_change[1:][vertical], angles[vertical])
        normal_change[:-1] = np.maximum(normal_change[:-1], np.where(vertical, angles, 0))

    p00, p10, p01, p11 = xyz[:-1, :-1], xyz[:-1, 1:], xyz[1:, :-1], xyz[1:, 1:]
    cell_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    n0 = np.cross(p10 - p00, p01 - p00)
    n1 = np.cross(p11 - p10, p01 - p10)
    area0 = np.linalg.norm(n0, axis=-1) * 0.5
    area1 = np.linalg.norm(n1, axis=-1) * 0.5
    cell_area = area0 + area1
    positive_areas = cell_area[cell_valid & np.isfinite(cell_area) & (cell_area > 1e-10)]
    median_area = float(np.median(positive_areas)) if positive_areas.size else 1.0
    degenerate_cells = cell_valid & (cell_area <= max(1e-10, median_area * 1e-6))
    triangle_fold = cell_valid & (np.sum(n0 * n1, axis=-1) < 0)
    fold_map = np.zeros(valid.shape, np.uint8)
    fold_map[:-1, :-1][triangle_fold | degenerate_cells] = 1

    boundary = np.zeros(valid.shape, np.uint8)
    boundary[1:-1, 1:-1] = (
        valid[1:-1, 1:-1]
        & (~valid[:-2, 1:-1] | ~valid[2:, 1:-1] | ~valid[1:-1, :-2] | ~valid[1:-1, 2:])
    )
    boundary[[0, -1], :] = valid[[0, -1], :]
    boundary[:, [0, -1]] = valid[:, [0, -1]]

    connectivity = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    component_labels, component_count = ndimage.label(valid, structure=connectivity)
    invalid_labels, invalid_count = ndimage.label(~valid, structure=connectivity)
    exterior_labels = set(np.unique(np.concatenate((invalid_labels[0], invalid_labels[-1], invalid_labels[:, 0], invalid_labels[:, -1]))))
    enclosed_hole_labels = [label for label in range(1, invalid_count + 1) if label not in exterior_labels]
    enclosed_hole_map = np.isin(invalid_labels, enclosed_hole_labels).astype(np.uint8)

    output.mkdir(parents=True, exist_ok=True)
    diagnostics = zarr.open_group(str(output / "geometry-diagnostics.zarr"), mode="w", zarr_format=2)
    chunks = (min(256, valid.shape[0]), min(256, valid.shape[1]))
    diagnostics.create_array("normals", data=normals, chunks=(*chunks, 3))
    diagnostics.create_array("stretch_log_ratio", data=stretch, chunks=chunks)
    diagnostics.create_array("normal_change_degrees", data=normal_change, chunks=chunks)
    diagnostics.create_array("fold_or_degenerate", data=fold_map, chunks=chunks)
    diagnostics.create_array("boundary", data=boundary, chunks=chunks)
    diagnostics.create_array("components", data=component_labels.astype(np.int32), chunks=chunks)
    diagnostics.create_array("enclosed_holes", data=enclosed_hole_map, chunks=chunks)
    files = {
        "stretch": save_diagnostic(output, "stretch-log-ratio", stretch, stretch_support),
        "normal_change": save_diagnostic(output, "normal-change-degrees", normal_change, normal_support),
        "fold_or_degenerate": save_diagnostic(output, "fold-or-degenerate", fold_map, valid),
        "boundary": save_diagnostic(output, "surface-boundary", boundary, valid),
        "enclosed_holes": save_diagnostic(output, "surface-enclosed-holes", enclosed_hole_map, ~valid),
    }
    p95_stretch = float(np.percentile(stretch[stretch_support], 95)) if stretch_support.any() else 0.0
    p95_normal = float(np.percentile(normal_change[normal_support], 95)) if normal_support.any() else 0.0
    manifest = {
        "kind": "vc_surface_geometry_diagnostics_v1",
        "score_semantics": "geometric_diagnostics_not_surface_correctness",
        "source_artifact": request["surface"],
        "source_surface_bundle": str(source_bundle),
        "surface_shape_vu": list(valid.shape),
        "valid_pixels": int(valid.sum()),
        "median_edge_length_u_voxels": median_edge_u,
        "median_edge_length_v_voxels": median_edge_v,
        "median_cell_area_voxels2": median_area,
        "p95_stretch_log_ratio": p95_stretch,
        "p95_normal_change_degrees": p95_normal,
        "fold_or_degenerate_cells": int((triangle_fold | degenerate_cells).sum()),
        "boundary_pixels": int(boundary.sum()),
        "connected_components": int(component_count),
        "enclosed_holes": len(enclosed_hole_labels),
        "global_self_intersection_tested": False,
        "limitations": [
            "regular-grid local fold and degeneracy tests do not prove correct papyrus-layer selection",
            "global nonlocal triangle self-intersections are not evaluated in v1",
        ],
        "diagnostics_zarr": "geometry-diagnostics.zarr",
        "maps": files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def trilinear_sample(volume: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64); z0 = np.floor(z).astype(np.int64)
    x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1
    inside = (x0 >= 0) & (y0 >= 0) & (z0 >= 0) & (x1 < volume.shape[2]) & (y1 < volume.shape[1]) & (z1 < volume.shape[0])
    result = np.zeros(x.shape, np.float32)
    if not inside.any():
        return result, inside
    xi0, yi0, zi0 = x0[inside], y0[inside], z0[inside]
    xi1, yi1, zi1 = x1[inside], y1[inside], z1[inside]
    xd = (x[inside] - xi0).astype(np.float32); yd = (y[inside] - yi0).astype(np.float32); zd = (z[inside] - zi0).astype(np.float32)
    v = volume.astype(np.float32, copy=False)
    c00 = v[zi0, yi0, xi0] * (1 - xd) + v[zi0, yi0, xi1] * xd
    c01 = v[zi0, yi1, xi0] * (1 - xd) + v[zi0, yi1, xi1] * xd
    c10 = v[zi1, yi0, xi0] * (1 - xd) + v[zi1, yi0, xi1] * xd
    c11 = v[zi1, yi1, xi0] * (1 - xd) + v[zi1, yi1, xi1] * xd
    c0 = c00 * (1 - yd) + c01 * yd; c1 = c10 * (1 - yd) + c11 * yd
    result[inside] = c0 * (1 - zd) + c1 * zd
    return result, inside


def villa_layer_contract(profile: str) -> dict:
    """Return layer selection matching the tracked Villa training/inference code."""
    contracts = {
        # Legacy TimeSformer reads layers [17,43) from canonical 00..64.
        "villa-timesformer-26": {"channels": 26, "source_layer_range": [17, 43]},
        # ResNet152/3D-decoder reads layers [1,63) from canonical 00..64.
        "villa-resnet152-62": {"channels": 62, "source_layer_range": [1, 63]},
    }
    if profile not in contracts:
        raise ValueError(f"unsupported Villa surface-volume profile: {profile}")
    contract = dict(contracts[profile])
    start, end = contract["source_layer_range"]
    contract["offsets_voxels"] = list(range(start - 32, end - 32))
    return contract


def normal_stack(request: dict, artifact: Path, output: Path) -> dict:
    """Sample a Villa-compatible H×W×C uint8 stack along surface normals."""
    import tifffile
    import zarr

    profile = request.get("villa_profile")
    contract = villa_layer_contract(profile)
    reverse_layers = bool(request.get("reverse_layers", False))
    layer_step = float(request.get("layer_step_voxels", 1.0))
    if not math.isfinite(layer_step) or layer_step <= 0 or layer_step > 4:
        raise ValueError("layer_step_voxels must be finite, positive, and at most 4")

    root = zarr.open_group(str(artifact / "surface.zarr"), mode="r")
    xyz = np.asarray(root["geometry/xyz"], dtype=np.float32)
    valid = np.asarray(root["geometry/valid"]) != 0
    normals, normal_support = surface_normals(xyz, valid)
    channels = int(contract["channels"])
    if valid.shape[0] <= channels or valid.shape[1] <= channels:
        raise ValueError(
            "surface chart height and width must both exceed the Villa channel count; "
            "the optimized loader otherwise cannot disambiguate HWC from CHW"
        )
    if valid.size * channels > MAX_SURFACE_VOLUME_VALUES:
        raise ValueError("surface volume exceeds the bounded 64M-value limit; use a smaller uv_region")

    offsets = [float(value) * layer_step for value in contract["offsets_voxels"]]
    required_padding = int(math.ceil(max(abs(value) for value in offsets))) + 1
    available_padding = int(root.attrs.get("normal_padding_voxels", 0))
    if required_padding > available_padding:
        raise ValueError(
            f"Villa profile requires normal padding {required_padding}, but registered surface has {available_padding}; "
            "rerun surface_render_registered_roi with larger normal_padding_voxels"
        )

    stage_manifest = read_json(artifact / "staging" / "stage-manifest.json")
    region = stage_manifest["submitted_region_xyz"]
    volume = np.load(artifact / "staging" / "staged-volume.npy", allow_pickle=False)
    if volume.ndim != 3 or volume.dtype != np.uint8:
        raise ValueError("Villa normal stacks require a staged rank-3 uint8 CT volume")
    local = xyz.copy()
    local[..., 0] -= int(region["x"])
    local[..., 1] -= int(region["y"])
    local[..., 2] -= int(region["z"])

    stack = np.zeros((*valid.shape, channels), dtype=np.uint8)
    supported = normal_support.copy()
    for index, offset in enumerate(offsets):
        coordinates = local + normals * offset
        sampled, inside = trilinear_sample(
            volume, coordinates[..., 0], coordinates[..., 1], coordinates[..., 2]
        )
        inside &= normal_support
        supported &= inside
        stack[..., index][inside] = np.clip(np.rint(sampled[inside]), 0, 255).astype(np.uint8)
    stack[~supported] = 0

    if reverse_layers:
        stack = stack[..., ::-1].copy()
        offsets = list(reversed(offsets))

    output.mkdir(parents=True, exist_ok=True)
    stack_path = output / "surface-volume.zarr"
    array = zarr.open(
        str(stack_path), mode="w", shape=stack.shape,
        chunks=(min(256, stack.shape[0]), min(256, stack.shape[1]), 1),
        dtype=np.uint8, compressor=None, zarr_format=2,
    )
    array[:] = stack
    array.attrs.update({
        "kind": "villa_surface_volume",
        "version": 1,
        "axes": ["v", "u", "normal_depth"],
        "profile": profile,
        "channels": channels,
        "source_layer_range": contract["source_layer_range"],
        "offsets_voxels": offsets,
        "layer_step_voxels": layer_step,
        "reverse_layers": reverse_layers,
        "normal_definition": "normalize(cross(dP_du,dP_dv))",
        "interpolation": "trilinear_then_round_to_uint8",
        "invalid_fill_value": 0,
    })

    layers = output / "layers"
    layers.mkdir()
    layer_hashes = []
    for index in range(channels):
        path = layers / f"{index:02d}.tif"
        tifffile.imwrite(path, stack[..., index])
        layer_hashes.append({"index": index, "offset_voxels": offsets[index], "sha256": sha256(path)})
    tifffile.imwrite(output / "surface-volume.tif", np.moveaxis(stack, -1, 0))
    write_png(output / "surface-volume-validity.png", supported.astype(np.uint8) * 255)
    middle = min(range(channels), key=lambda index: abs(offsets[index]))
    write_png(output / "surface-volume-middle.png", stack[..., middle])

    manifest = {
        "kind": "vc_villa_surface_volume_v1",
        "score_semantics": "raw_ct_samples_not_ink_probability",
        "source_artifact": request["surface"],
        "profile": profile,
        "shape_hwc": list(stack.shape),
        "dtype": "uint8",
        "axes": ["v", "u", "normal_depth"],
        "channels": channels,
        "villa_loader_compatible": True,
        "source_layer_range": contract["source_layer_range"],
        "offsets_voxels": offsets,
        "layer_step_voxels": layer_step,
        "reverse_layers": reverse_layers,
        "normal_orientation": "cross(dP_du,dP_dv)",
        "interpolation": "trilinear_then_round_to_uint8",
        "supported_pixels": int(supported.sum()),
        "support_fraction": float(supported.sum() / max(1, valid.sum())),
        "surface_volume_zarr": "surface-volume.zarr",
        "surface_volume_tiff": "surface-volume.tif",
        "layers_directory": "layers",
        "validity_preview": "surface-volume-validity.png",
        "middle_layer_preview": "surface-volume-middle.png",
        "middle_layer_index": middle,
        "layer_hashes": layer_hashes,
        "villa_contract": {
            "timesformer_26": "train_timesformer_og.py default layer range [17,43)",
            "resnet152_62": "train_resnet3d_3d_decoder.py layer range [1,63)",
            "optimized_inference_layout": "H,W,C uint8 Zarr with chunks Htile,Wtile,1",
        },
        "limitations": [
            "normal orientation must be checked per surface; reverse_layers is explicit",
            "this artifact contains raw CT samples and is not an ink prediction",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def ct_alignment(request: dict, artifact: Path, output: Path) -> dict:
    import zarr

    root = zarr.open_group(str(artifact / "surface.zarr"), mode="r")
    xyz = np.asarray(root["geometry/xyz"], dtype=np.float32)
    valid = np.asarray(root["geometry/valid"]) != 0
    if valid.size > MAX_ALIGNMENT_PIXELS:
        raise ValueError("CT alignment is limited to 1,048,576 UV pixels; create a smaller registered uv_region")
    normals, normal_support = surface_normals(xyz, valid)
    maximum_offset = int(request.get("maximum_offset_voxels", 2))
    if maximum_offset < 1 or maximum_offset > 16:
        raise ValueError("maximum_offset_voxels must be from 1 to 16")
    available_padding = int(root.attrs.get("normal_padding_voxels", 0))
    if maximum_offset > available_padding:
        raise ValueError(f"requested offset {maximum_offset} exceeds staged normal padding {available_padding}")
    stage_manifest = read_json(artifact / "staging" / "stage-manifest.json")
    region = stage_manifest["submitted_region_xyz"]
    volume = np.load(artifact / "staging" / "staged-volume.npy", allow_pickle=False)
    local = xyz.copy()
    local[..., 0] -= int(region["x"]); local[..., 1] -= int(region["y"]); local[..., 2] -= int(region["z"])

    minimum = np.full(valid.shape, np.inf, np.float32)
    maximum = np.full(valid.shape, -np.inf, np.float32)
    best_gradient = np.zeros(valid.shape, np.float32)
    best_offset = np.zeros(valid.shape, np.float32)
    supported = normal_support.copy()
    previous = None
    mean_profile = []
    offsets = list(range(-maximum_offset, maximum_offset + 1))
    for offset in offsets:
        coordinates = local + normals * float(offset)
        sample, inside = trilinear_sample(volume, coordinates[..., 0], coordinates[..., 1], coordinates[..., 2])
        inside &= normal_support
        supported &= inside
        minimum[inside] = np.minimum(minimum[inside], sample[inside])
        maximum[inside] = np.maximum(maximum[inside], sample[inside])
        mean_profile.append(float(sample[inside].mean()) if inside.any() else 0.0)
        if previous is not None:
            gradient = np.abs(sample - previous[0])
            pair_support = inside & previous[1]
            update = pair_support & (gradient > best_gradient)
            best_gradient[update] = gradient[update]
            best_offset[update] = offset - 0.5
        previous = (sample, inside)
    dynamic_range = np.maximum(maximum - minimum, 1e-6)
    confidence = np.zeros(valid.shape, np.float32)
    confidence[supported] = np.clip(best_gradient[supported] / dynamic_range[supported], 0, 1)
    best_gradient[~supported] = 0; best_offset[~supported] = 0
    consistency = np.zeros(valid.shape, np.float32)
    count = np.zeros(valid.shape, np.float32)
    for axis in (0, 1):
        difference = np.abs(np.diff(best_offset, axis=axis))
        pair = np.diff(supported.astype(np.int8), axis=axis) == 0
        if axis == 0:
            pair &= supported[1:] & supported[:-1]
            consistency[1:][pair] += difference[pair]; consistency[:-1][pair] += difference[pair]
            count[1:][pair] += 1; count[:-1][pair] += 1
        else:
            pair &= supported[:, 1:] & supported[:, :-1]
            consistency[:, 1:][pair] += difference[pair]; consistency[:, :-1][pair] += difference[pair]
            count[:, 1:][pair] += 1; count[:, :-1][pair] += 1
    consistency[count > 0] /= count[count > 0]

    output.mkdir(parents=True, exist_ok=True)
    diagnostics = zarr.open_group(str(output / "ct-alignment.zarr"), mode="w", zarr_format=2)
    chunks = (min(256, valid.shape[0]), min(256, valid.shape[1]))
    diagnostics.create_array("peak_gradient", data=best_gradient, chunks=chunks)
    diagnostics.create_array("peak_offset", data=best_offset, chunks=chunks)
    diagnostics.create_array("confidence", data=confidence, chunks=chunks)
    diagnostics.create_array("offset_consistency", data=consistency, chunks=chunks)
    diagnostics.create_array("supported", data=supported.astype(np.uint8), chunks=chunks)
    files = {
        "peak_gradient": save_diagnostic(output, "ct-peak-gradient", best_gradient, supported),
        "peak_offset": save_diagnostic(output, "ct-peak-offset", best_offset, supported),
        "confidence": save_diagnostic(output, "ct-alignment-confidence", confidence, supported),
        "offset_consistency": save_diagnostic(output, "ct-offset-inconsistency", consistency, supported),
    }
    manifest = {
        "kind": "vc_surface_ct_alignment_v1",
        "score_semantics": "normal_profile_evidence_not_surface_correctness",
        "source_artifact": request["surface"],
        "surface_shape_vu": list(valid.shape),
        "maximum_offset_voxels": maximum_offset,
        "offsets_voxels": offsets,
        "interpolation": "trilinear",
        "supported_pixels": int(supported.sum()),
        "support_fraction": float(supported.sum() / max(1, valid.sum())),
        "median_peak_gradient": float(np.median(best_gradient[supported])) if supported.any() else 0.0,
        "median_peak_offset_voxels": float(np.median(best_offset[supported])) if supported.any() else 0.0,
        "median_confidence": float(np.median(confidence[supported])) if supported.any() else 0.0,
        "mean_profile_by_offset": mean_profile,
        "ct_alignment_zarr": "ct-alignment.zarr",
        "maps": files,
        "limitations": [
            "strong CT normal gradients are evidence of a sheet boundary, not proof of the correct papyrus layer",
            "v1 does not estimate fiber orientation or physical sheet thickness",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    importer = subparsers.add_parser("import")
    importer.add_argument("--request", required=True, type=Path)
    importer.add_argument("--surface", required=True, type=Path)
    importer.add_argument("--output", required=True, type=Path)
    renderer = subparsers.add_parser("render")
    renderer.add_argument("--request", required=True, type=Path)
    renderer.add_argument("--output", required=True, type=Path)
    for name in ("geometry", "alignment", "normal-stack"):
        diagnostic = subparsers.add_parser(name)
        diagnostic.add_argument("--request", required=True, type=Path)
        diagnostic.add_argument("--artifact", required=True, type=Path)
        diagnostic.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    request = read_json(arguments.request)
    arguments.output.mkdir(parents=True, exist_ok=True)
    if arguments.command == "import":
        result = import_surface(request, arguments.surface, arguments.output)
    elif arguments.command == "render":
        result = render_surface(request, arguments.output)
    elif arguments.command == "geometry":
        result = geometry_diagnostics(request, arguments.artifact, arguments.output)
    elif arguments.command == "normal-stack":
        result = normal_stack(request, arguments.artifact, arguments.output)
    else:
        result = ct_alignment(request, arguments.artifact, arguments.output)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
