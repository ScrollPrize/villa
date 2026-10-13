"""VC3D fiber JSON: annotated fibers loaded from a directory of fiber files."""
from __future__ import annotations

import glob
import hashlib
import os
import json

import numpy as np
from scipy.spatial import cKDTree
from vc3d_fiber_format import parse_vc3d_fiber_format

from vesuvius.neural_tracing.tracers.shared.geometry import arclength
from vesuvius.neural_tracing.tracers.shared.annotations.repair import foldbacks, repair_kinks
from vesuvius.neural_tracing.tracers.shared.annotations.traced import FiberSpan, TracedFiber


def load_fibers(fiber_dir: str, grid_scale: float = 8.0, spacing: float = 1.0) -> list[TracedFiber]:
    """Load only geometry between human control points, retaining span provenance.

    Every line point between the outer controls is equally valid supervision,
    regardless of tags or interpolation provenance. An outer control without an
    explicit termination tag is a censored boundary, not a physical endpoint.
    Short annotation kinks are repaired and fold-backs recorded (annotation_repair).
    """
    if grid_scale <= 0 or spacing <= 0:
        raise ValueError("grid_scale and spacing must be positive")
    out = []
    for path in sorted(glob.glob(os.path.join(fiber_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        fib = parse_vc3d_fiber_format(raw, path=path)
        if len(fib.control_points_xyz) < 2:
            continue  # a seed and its extrapolations provide no controlled span
        line = np.asarray(fib.line_points_xyz, dtype=np.float64)
        controls = np.asarray(fib.control_points_xyz, dtype=np.float64)
        if len(line) < 2:
            raise ValueError(f"{path}: controlled fiber has no dense line")
        dist, anchors = cKDTree(line).query(controls)
        if np.any(dist > 1e-4):
            raise ValueError(f"{path}: control points must be anchored in line_points")
        if np.all(np.diff(anchors) < 0):
            line = line[::-1]
            anchors = len(line) - 1 - anchors
        if np.any(np.diff(anchors) <= 0):
            raise ValueError(f"{path}: control points must have distinct ordered line anchors")
        line = line / grid_scale
        pieces, bounds = [], [0]
        for i, (a, b) in enumerate(zip(anchors[:-1], anchors[1:])):
            # Preserve every line vertex, not just human control anchors.
            original = line[a:b + 1]
            pieces_dense = []
            for left, right in zip(original[:-1], original[1:]):
                distance = np.linalg.norm(right-left)
                if distance > 1e-9:
                    count = max(1, int(np.ceil(distance/spacing)))
                    pieces_dense.append(left+(right-left)*np.arange(count)[:, None]/count)
            if not pieces_dense:
                raise ValueError(f"{path}: zero-length controlled span")
            piece = np.concatenate([*pieces_dense, original[-1:]])
            pieces.append(piece if i == 0 else piece[1:])
            bounds.append(bounds[-1]+len(piece)-1)
        pts = np.concatenate(pieces)
        if len(pts) < 8:
            continue
        folds = foldbacks(pts, arclength(pts))
        pts, repairs = repair_kinks(pts, arclength(pts))
        s = arclength(pts)
        spans = [FiberSpan(float(s[a]), float(s[b]), fib.control_point_segments[i])
                 for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
        tag = fiber_tag(fib.metadata)
        endpoints = (False, False)
        if fib.version == 3:
            endpoints = tuple("kollesis_termination" in raw["control_points"][i].get("tags", [])
                              for i in (0, -1))
        full_s = arclength(line)
        tails = float(full_s[anchors[0]] + full_s[-1] - full_s[anchors[-1]])
        source_hash = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()
        out.append(TracedFiber(os.path.basename(path), pts, s, tag,
                              spans=tuple(spans),
                              endpoint_stop=endpoints, source_hash=source_hash,
                              excluded_tail_length=tails, kink_repairs=len(repairs), foldbacks=folds))
    return out


def fiber_tag(record) -> str:
    """The H/V family of a VC3D fiber (its JSON, or its parsed metadata): the automatic classification's tag."""
    return (record.get("hv_classification") or {}).get("automatic_tag", "")
