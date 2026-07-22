"""Build immutable Spiral fiber overlays through ``vc.annotation_volume``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np


def load_fiber_display_geometry(path, *, identity=None, coordinate_scale=0.25):
    """Load full display geometry without the fitter's constraint decimation."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)

    def points(name):
        values = np.asarray(document.get(name) or [], dtype=np.float64)
        if values.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError(f"Fiber {path} {name} must have shape [N, 3]")
        if not np.isfinite(values).all():
            raise ValueError(f"Fiber {path} {name} contains non-finite coordinates")
        return np.ascontiguousarray(values * float(coordinate_scale))

    line = points("line_points")
    controls = points("control_points")
    if len(line) == 0:
        line = controls
    if len(line) == 0:
        return None
    return {
        "identity": str(identity or path.name),
        "source_path": str(path),
        "line_points_xyz": line,
        "control_points_xyz": controls,
    }


def write_fiber_annotation_volume(destination, volume_spec, fibers: Mapping[str, dict]):
    """Write one immutable sparse uint16 volume and return its meta.json path."""
    if not fibers or not volume_spec:
        return None

    from vc import annotation_volume as annotation

    levels = [
        annotation.PyramidLevel(
            tuple(int(value) for value in level["shape_zyx"]),
            tuple(float(value) for value in level.get("scale_zyx", (2, 2, 2))),
            tuple(float(value) for value in level.get("translation_zyx", (0, 0, 0))),
        )
        for level in volume_spec.get("pyramid_levels", ())
    ]
    identities = sorted(fibers)
    if len(identities) > 65535:
        raise ValueError("A uint16 annotation volume supports at most 65,535 fibers")

    label_entries = []
    batches = []
    for label, identity in enumerate(identities, start=1):
        fiber = fibers[identity]
        label_entries.append({
            "label": label,
            "identity": identity,
            "source_path": fiber.get("source_path", ""),
        })
        line = fiber["line_points_xyz"]
        if len(line):
            batches.append(annotation.AnnotationPointBatch(
                label, line,
                annotation.CoordinateOrder.XYZ,
                annotation.GeometryMode.ORDERED_POLYLINE,
                1.0))
        controls = fiber["control_points_xyz"]
        if len(controls):
            batches.append(annotation.AnnotationPointBatch(
                label, controls,
                annotation.CoordinateOrder.XYZ,
                annotation.GeometryMode.POINTS,
                3.0))

    attributes = {
        "annotation_kind": "fibers",
        "fiber_labels": label_entries,
        "spiral_fiber_coordinate_scale": float(
            volume_spec.get("fiber_coordinate_scale", 0.25)),
    }
    attributes.update(dict(volume_spec.get("coordinate_identity") or {}))
    if "voxel_size_um" in volume_spec:
        attributes["voxel_size_um"] = float(volume_spec["voxel_size_um"])

    spec = annotation.SparseAnnotationVolumeSpec(
        tuple(int(value) for value in volume_spec["shape_zyx"]),
        pyramid_levels=levels,
        root_attributes=attributes)
    destination = Path(destination)
    annotation.write_sparse_annotation_volume(spec, batches, destination)
    return str(destination / "meta.json")
