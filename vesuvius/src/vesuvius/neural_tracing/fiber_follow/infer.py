"""Trace fibers from seed points and write VC3D fiber JSON.

Seeds are base-voxel ``x,y,z`` (the fiber JSON coordinate space). Each seed
is snapped to the local presence maximum, then traced in both directions
and written as one ``vc3d_fiber`` v3 file.

  python -m vesuvius.neural_tracing.fiber_follow.infer --checkpoint last.pt \
      --seed 18529.9,13044.9,51234.1 --out traced/

  # automatic seeds: the N strongest presence maxima in a base-voxel box
  python -m vesuvius.neural_tracing.fiber_follow.infer --checkpoint last.pt \
      --auto-seeds 50 --box x0,y0,z0,x1,y1,z1 --out traced/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.geometry import arclength, resample_polyline
from vesuvius.neural_tracing.fiber_follow.trace import DEFAULT_CONFIDENCE, ModelTracer, TraceParams, field_axis
from vesuvius.neural_tracing.fiber_follow.train import load_checkpoint
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec


def snap_to_presence(vol: FiberVolume, p_xyz: np.ndarray, radius: int = 2) -> np.ndarray:
    base = np.round(p_xyz[::-1]).astype(np.int64) - radius
    blk = vol.presence.read(base, (2 * radius + 1,) * 3).astype(np.float32)
    if blk.max() <= 0:
        return p_xyz
    z, y, x = np.unravel_index(np.argmax(blk), blk.shape)
    return (base + np.array([z, y, x]))[::-1].astype(np.float64)


def auto_seeds(vol: FiberVolume, box_grid_zyx: tuple[np.ndarray, np.ndarray], n: int, min_presence: float = 0.9,
               min_sep: int = 6) -> np.ndarray:
    lo, hi = box_grid_zyx
    blk = vol.presence.read(lo, hi - lo).astype(np.float32) / 255.0
    mx = ndimage.maximum_filter(blk, size=2 * min_sep + 1)
    peaks = np.argwhere((blk == mx) & (blk >= min_presence))
    order = np.argsort(-blk[tuple(peaks.T)], kind="stable")
    peaks = peaks[order][:n]
    return (peaks + lo)[:, ::-1].astype(np.float64)  # xyz grid


def make_fiber_json(points_base_xyz: np.ndarray, cp_every: float, meta: dict, control_indices=None) -> dict:
    """VC3D v3 fiber: dense line_points plus cspline control points.

    ``control_indices`` places the controls at given line indices instead of
    every ``cp_every`` base voxels."""
    s = arclength(points_base_xyz)
    if control_indices is None:
        n_cp = max(2, int(round(s[-1] / cp_every)) + 1)
        idx = np.unique(np.searchsorted(s, np.linspace(0, s[-1], n_cp)).clip(0, len(s) - 1))
    else:
        idx = np.unique(np.asarray(control_indices, np.int64).clip(0, len(s) - 1))
    seg = {
        "config": {
            "beam_lookahead_steps": 2, "beam_prune_distance_voxels": 1.0, "beam_width": 8,
            "cone_angle_degrees": 25.0, "cone_angle_step_degrees": 5.0, "cone_grid_size": 25,
            "cumulative_smoothness_steps": 4, "cumulative_smoothness_tangent_weight": 2.0,
            "endpoint_accept_threshold_base_voxels": 20.0, "initial_free_angle_degrees": 0.0,
            "max_step_factor": 3.0, "meeting_accept_max_error_ratio": 0.1,
            "smoothness_free_angle_degrees": 0.0, "smoothness_normal_weight": 0.1,
            "smoothness_tangent_weight": 10.0, "smoothness_weight": 2.0, "step_voxels": 4.0,
        },
        "failure_code": "", "failure_detail": "",
        "fiber_manifest": meta.get("fiber_manifest", ""),
        "interp_goal": "cspline", "interp_mode": "cspline",
        "lasagna_failure_code": "", "lasagna_failure_detail": "",
        "meeting_error_base_voxels": None, "meeting_error_ratio": None, "meeting_source": "",
        "metadata_version": 3, "metric": None, "msg": "fiber_follow",
        "normal_manifest": "", "optimizer": "native_fiber_trace3d",
        "tracer_version": 2, "trace_to_base_scale": 1.0,
    }
    cps = []
    for k, i in enumerate(idx):
        cp = {"position": [float(v) for v in points_base_xyz[i]]}
        if k + 1 < len(idx):
            cp["segment_to_next"] = json.loads(json.dumps(seg))
        cps.append(cp)
    return {
        "type": "vc3d_fiber",
        "version": 3,
        "optimization_mode": "native_fiber_trace3d",
        "generation": 1,
        "branches": [],
        "tags": [],
        "control_points": cps,
        "line_points": [[float(v) for v in p] for p in points_base_xyz],
        **{k: v for k, v in meta.items() if k != "fiber_manifest"},
    }


def trace_bidirectional(tracer: ModelTracer, vol: FiberVolume, seeds_grid_xyz: np.ndarray, headings=None):
    """Returns per-seed full polylines (grid xyz), ordered backward->forward."""
    seeds = np.stack([snap_to_presence(vol, s) for s in seeds_grid_xyz])
    axes = np.stack([field_axis(vol, s)[0] for s in seeds]) if headings is None else np.asarray(headings)
    fw, rf = tracer.trace(seeds, axes)
    bw, rb = tracer.trace(seeds, -axes)
    out = []
    for f, b, a, c in zip(fw, bw, rf, rb):
        out.append((np.concatenate([b[::-1], f[1:]], 0), (c, a)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fiber-zarrs", default=None, help="override the checkpoint's fiber zarr dir")
    ap.add_argument("--ct", default=None, help="override the checkpoint's CT zarr")
    ap.add_argument("--seed", action="append", default=[], help="base-voxel x,y,z (repeatable)")
    ap.add_argument("--heading", action="append", default=[], help="Optional seed direction x,y,z; otherwise estimate from the initialization field (presence PCA for CT)")
    ap.add_argument("--seeds-file", default=None, help="JSON list of base-voxel [x,y,z]")
    ap.add_argument("--auto-seeds", type=int, default=0)
    ap.add_argument("--box", default=None, help="base-voxel x0,y0,z0,x1,y1,z1 for --auto-seeds")
    ap.add_argument("--min-length", type=float, default=400.0, help="drop traces shorter than this (base voxels)")
    ap.add_argument("--dedupe", type=float, default=16.0,
                    help="skip seeds within this base-voxel distance of an already-written trace")
    ap.add_argument("--cp-every", type=float, default=800.0, help="control-point spacing, base voxels")
    ap.add_argument("--max-len", type=float, default=6000.0, help="per-direction limit, trace-grid voxels")
    ap.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    ap.add_argument("--n-commit", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    model, crop, n_hist, spec, _ = load_checkpoint(args.checkpoint, args.device)
    if args.fiber_zarrs:
        spec.fiber_zarr_dir = args.fiber_zarrs
    if args.ct:
        spec.ct_zarr = args.ct
    vol = FiberVolume(spec, cache_bytes=8 << 30)
    tracer = ModelTracer(model, vol, crop, n_hist, TraceParams(max_len=args.max_len, confidence=args.confidence, n_commit=args.n_commit), device=args.device)
    g = spec.grid_scale

    seeds = [np.array([float(v) for v in s.split(",")]) for s in args.seed]
    if args.seeds_file:
        seeds += [np.asarray(s, float) for s in json.load(open(args.seeds_file))]
    seeds = [s / g for s in seeds]
    if args.auto_seeds:
        if not args.box:
            raise SystemExit("--auto-seeds needs --box")
        b = np.array([float(v) for v in args.box.split(",")]) / g
        lo = np.floor(b[:3][::-1]).astype(np.int64)
        hi = np.ceil(b[3:][::-1]).astype(np.int64)
        seeds += list(auto_seeds(vol, (lo, hi), args.auto_seeds))
    if not seeds:
        raise SystemExit("no seeds given")
    headings = None
    if args.heading:
        headings = np.array([[float(v) for v in h.split(',')] for h in args.heading])
        if headings.shape != (len(seeds), 3) or not np.isfinite(headings).all() or np.any(np.linalg.norm(headings, axis=1) == 0):
            raise SystemExit('--heading must supply one finite nonzero xyz vector per seed')

    os.makedirs(args.out, exist_ok=True)
    manifest = [n for n in os.listdir(spec.fiber_zarr_dir) if n.endswith(".lasagna.json")]
    meta = {"username": "fiber_follow",
            "fiber_manifest": os.path.join(spec.fiber_zarr_dir, manifest[0]) if manifest else ""}
    written, tree_pts = [], np.zeros((0, 3))
    t0 = time.time()
    n_skipped = 0
    for b in range(0, len(seeds), args.batch):
        chunk = np.stack(seeds[b : b + args.batch])
        chunk_headings = None if headings is None else headings[b : b + args.batch]
        if len(tree_pts):
            d, _ = cKDTree(tree_pts).query(chunk)
            keep = d * g > args.dedupe
            n_skipped += int((~keep).sum())
            chunk = chunk[keep]
            if chunk_headings is not None:
                chunk_headings = chunk_headings[keep]
            if not len(chunk):
                continue
        for poly, reasons in trace_bidirectional(tracer, vol, chunk, chunk_headings):
            base = resample_polyline(poly * g, g)  # 1 grid voxel spacing, like VC3D traces
            L = arclength(base)[-1]
            if L < args.min_length:
                continue
            if len(tree_pts):
                d, _ = cKDTree(tree_pts).query(poly)
                if np.mean(d * g < args.dedupe) > 0.5:  # mostly re-traces an existing fiber
                    n_skipped += 1
                    continue
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
            name = f"fiber_follow_{stamp}_{len(written):06d}.json"
            obj = make_fiber_json(base, args.cp_every, dict(meta, filename=name, started_at=stamp,
                                                             sequence=len(written),
                                                             fiber_follow={"stop_reasons": list(reasons),
                                                                           "checkpoint": os.path.abspath(args.checkpoint)}))
            with open(os.path.join(args.out, name), "w") as fh:
                json.dump(obj, fh)
            written.append(name)
            tree_pts = np.concatenate([tree_pts, poly], 0)
    dt = time.time() - t0
    print(json.dumps(dict(seeds=len(seeds), written=len(written), skipped=n_skipped, seconds=round(dt, 2),
                          out=os.path.abspath(args.out))))


if __name__ == "__main__":
    main()
