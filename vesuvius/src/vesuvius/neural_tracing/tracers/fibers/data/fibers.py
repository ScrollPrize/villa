"""Annotated fibers: loading VC3D fiber JSON, the geometry manifest, and whole-fiber train/validation splits."""
from __future__ import annotations

import glob
import hashlib
import math
import os
from dataclasses import dataclass
import json

import numpy as np
from scipy.spatial import cKDTree
from vc3d_fiber_format import parse_vc3d_fiber_format, FiberTraceSegmentMetadata

from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec, arclength, block_start
from vesuvius.neural_tracing.tracers.fibers.data.annotation_repair import foldbacks, repair_kinks


DATA_VERSION = 2


@dataclass(frozen=True)
class FiberSpan:
    start: float
    end: float
    provenance: FiberTraceSegmentMetadata


@dataclass
class TracedFiber:
    name: str
    points: np.ndarray  # (N, 3) trace-grid xyz; all annotated vertices, gaps <= 1 voxel
    s: np.ndarray
    tag: str
    spans: tuple[FiberSpan, ...] = ()
    endpoint_stop: tuple[bool, bool] = (False, False)
    source_hash: str = ""
    excluded_tail_length: float = 0.0
    kink_repairs: int = 0  # annotation kinks replaced by smooth bridges at load (annotation_repair)
    foldbacks: tuple[float, ...] = ()  # arcs where the annotation doubles back; kept out of training

    @property
    def length(self) -> float:
        return float(self.s[-1])


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
        tag = (fib.metadata.get("hv_classification") or {}).get("automatic_tag", "")
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


def fiber_manifest(fibers):
    """Ordered identities for caches containing fiber indices and arc positions."""
    if hasattr(fibers, 'manifest_entries'):
        return fibers.manifest_entries()
    return [dict(name=f.name, source_hash=f.source_hash,
                 geometry_hash=hashlib.sha256(np.asarray(f.points, dtype="<f8").tobytes()).hexdigest(),
                 endpoint_stop=list(f.endpoint_stop)) for f in fibers]


@dataclass(frozen=True)
class ZBand:
    """Held-out band in trace-grid z (inclusive-exclusive)."""

    lo: float
    hi: float

    def contains(self, z):
        return (z >= self.lo) & (z < self.hi)


def split_fibers(fibers: list[TracedFiber], val_band: ZBand):
    """Split by the controlled geometry centroid; filter training states separately.

    Fibers whose annotation doubles back are kept out of training (validation is unchanged).
    """
    train, val = [], []
    for f in fibers:
        z = np.average((f.points[:-1, 2]+f.points[1:, 2])/2, weights=np.diff(f.s))
        if val_band.contains(z):
            val.append(f)
        elif not f.foldbacks:
            train.append(f)
    return train, val


def training_state_allowed(item, crop: CropSpec, band: ZBand | None):
    """Exclude the perturbed state, entire sampled block, and used history/labels.

    Keep the original 48-voxel position guard and also check the actual rotated
    crop's read footprint (including interpolation support). Applied before I/O
    to fresh and cached states. The circumsphere covers any CT-selected roll.
    """
    if band is None:
        return True
    pos, frame = item["pos"], item["frame"]
    if band.lo - 48 <= pos[2] < band.hi + 48:
        return False
    start = block_start(pos, frame, crop)[0]
    if start < band.hi and start + crop.block_size > band.lo:
        return False
    zs = [np.array([pos[2]])]
    for key, mask in (("hist_local", "hmask"), ("gt_history", "gt_history_mask"),
                      ("fut_local", "fmask")):
        if key in item:
            points = item[key][item[mask] > 0]
            if len(points):
                zs.append((points @ frame.T + pos)[:, 2])
    if "plane_ab" in item:
        # The forward-plane labels can look farther along GT than arclength targets.
        loc = np.c_[item["plane_ab"], item["planes"]]
        loc = loc[item["plane_mask"] > 0]
        if len(loc):
            zs.append((loc @ frame.T + pos)[:, 2])
    if "dense_ab" in item:
        loc = np.c_[item["dense_ab"], item["dense_planes"]][item["dense_mask"] > 0]
        if len(loc):
            zs.append((loc @ frame.T + pos)[:, 2])
    z = np.concatenate(zs)
    return not (z.min() < band.hi and z.max() >= band.lo)


def validate_split_policy(validation):
    if validation.get('strategy') != 'fiber_hash':
        raise ValueError('Expected validation.strategy="fiber_hash"')
    if ('fraction' in validation) == ('count' in validation):
        raise ValueError('Specify exactly one validation fraction or count')
    if 'count' in validation:
        count = validation['count']
        if type(count) is not int or count < 1:
            raise ValueError('Validation count must be a positive integer')
    else:
        fraction = float(validation['fraction'])
        if not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError('Validation fraction must be between zero and one')
    int(validation['seed'])


def heldout_ids(ids, validation):
    validate_split_policy(validation)
    ids = list(ids)
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError('Fiber split needs at least two unique IDs')
    seed = int(validation['seed'])
    order = sorted(ids, key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).digest())
    count = (validation['count'] if 'count' in validation else
             min(len(ids)-1,max(1,round(len(ids)*float(validation['fraction'])))))
    if count >= len(ids):
        raise ValueError('Validation count must leave at least one training fiber')
    return set(order[:count])
