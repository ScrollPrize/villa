"""The annotated part of a fiber line (Python twin of vc_object_store ``annotatedLineRange``).

A manual fiber (two or more control points) is annotated only between its first and last control point. Line points
past them are optimizer extrapolation (tails) and are not fiber: every consumer ignores them. A fiber with fewer than
two controls (automatic traces) has no tails; its whole line counts.

The range runs over the line vertices nearest the finite controls, from the lowest to the highest such index (ties take
the outermost vertex), inclusive; control order does not matter. Controls and line must be in the same frame.
"""
import numpy as np


def annotated_line_range(control_points, line_points):
    """``(begin, end)``, end exclusive, into ``line_points``."""
    line = np.asarray(line_points, dtype=np.float64).reshape(-1, 3)
    controls = np.asarray(control_points, dtype=np.float64).reshape(-1, 3)
    controls = controls[np.all(np.isfinite(controls), axis=1)]
    if len(controls) < 2 or not len(line):
        return 0, len(line)
    distance = np.sum((line[:, None, :]-controls[None, :, :])**2, axis=2)  # [points, controls]
    nearest = distance == distance.min(axis=0)[None, :]
    rows = np.nonzero(nearest)[0]
    return int(rows.min()), int(rows.max())+1


def annotated_line(control_points, line_points):
    """``line_points`` without its tails (a view when ``line_points`` is an array)."""
    begin, end = annotated_line_range(control_points, line_points)
    return np.asarray(line_points)[begin:end]


def packed_annotated_ranges(control_offsets, control_values, line_offsets, line_values):
    """Per-fiber ``(begin, end)`` columns for packed fibers (offsets of length N+1, values [M,3])."""
    control_offsets = np.asarray(control_offsets, dtype=np.int64)
    line_offsets = np.asarray(line_offsets, dtype=np.int64)
    if len(control_offsets) != len(line_offsets):
        raise ValueError('Control and line columns differ in fiber count')
    counts = np.diff(line_offsets)
    begin = np.zeros(len(counts), np.int64)
    end = counts.copy()
    controls = np.asarray(control_values, dtype=np.float64).reshape(-1, 3)
    line = np.asarray(line_values, dtype=np.float64).reshape(-1, 3)
    for k in np.flatnonzero((np.diff(control_offsets) >= 2) & (counts > 0)):
        begin[k], end[k] = annotated_line_range(controls[control_offsets[k]:control_offsets[k+1]],
                                                line[line_offsets[k]:line_offsets[k+1]])
    return begin, end


def document_control_points(document):
    """Control point XYZ of a VC3D fiber JSON document (v1 ``[x, y, z]`` or v3/v4 ``{"position": ...}``), [N,3]."""
    controls = [c['position'] if isinstance(c, dict) else c for c in document.get('control_points') or []]
    return np.asarray(controls, dtype=np.float64).reshape(-1, 3)
