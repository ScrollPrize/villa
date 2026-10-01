"""Trace fibers from seed points and write VC3D fiber JSON.

Seeds are unchanged base-voxel ``x,y,z`` (the fiber JSON coordinate space).
CT plus the supplied H/V family initializes each bidirectional trace.

  python -m vesuvius.neural_tracing.fiber_follow.flow_matching.infer --checkpoint last.pt \
      --seed 18529.9,13044.9,51234.1 --family H --out traced/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, resample_polyline
from vesuvius.neural_tracing.fiber_follow.shared.trace import DEFAULT_CONFIDENCE, DEFAULT_N_COMMIT, ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.shared.heading import (
    ct_seed_heading, SEED_HEADING_POLICY, TRACE_HEADING_POLICY, FRAME_POLICY,
)


def make_fiber_json(points_base_xyz: np.ndarray, cp_every: float, meta: dict) -> dict:
    """VC3D v3 fiber: dense line_points plus cspline controls every ``cp_every`` base voxels."""
    s = arclength(points_base_xyz)
    n_cp = max(2, int(round(s[-1] / cp_every)) + 1)
    idx = np.unique(np.searchsorted(s, np.linspace(0, s[-1], n_cp)).clip(0, len(s) - 1))
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


def trace_bidirectional(tracer: ModelTracer, vol: FiberVolume, seeds_grid_xyz: np.ndarray, families):
    """Returns per-seed full polylines (grid xyz), ordered backward->forward."""
    seeds = np.asarray(seeds_grid_xyz, dtype=np.float64)
    if seeds.ndim != 2 or seeds.shape[1] != 3 or not np.isfinite(seeds).all():
        raise ValueError('Seeds must be finite xyz positions')
    families = [families]*len(seeds) if isinstance(families, str) else list(families)
    if len(families) != len(seeds):
        raise ValueError('Supply one H/V family per seed')
    if not len(seeds):
        return []
    axes = np.stack([ct_seed_heading(vol, s, family) for s, family in zip(seeds, families)])
    fw, rf = tracer.trace(seeds, axes)
    bw, rb = tracer.trace(seeds, -axes)
    out = []
    for f, b, a, c in zip(fw, bw, rf, rb):
        out.append((np.concatenate([b[::-1], f[1:]], 0), (c, a)))
    return out


def main(argv=None, *, checkpoint_loader, tracer_class=ModelTracer):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fiber-zarrs", default=None, help="override the checkpoint's fiber zarr dir")
    ap.add_argument("--ct", default=None, help="override the checkpoint's CT zarr")
    ap.add_argument("--seed", action="append", default=[], help="base-voxel x,y,z (repeatable)")
    ap.add_argument("--family", action="append", choices=('H', 'V'), required=True,
                    help="One H/V family for all seeds, or repeat once per seed")
    ap.add_argument("--seeds-file", default=None, help="JSON list of base-voxel [x,y,z]")
    ap.add_argument("--min-length", type=float, default=400.0, help="drop traces shorter than this (base voxels)")
    ap.add_argument("--dedupe", type=float, default=16.0,
                    help="skip seeds within this base-voxel distance of an already-written trace")
    ap.add_argument("--cp-every", type=float, default=800.0, help="control-point spacing, base voxels")
    ap.add_argument("--max-len", type=float, default=6000.0, help="per-direction limit, trace-grid voxels")
    ap.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    ap.add_argument("--n-commit", type=int, help="max points committed per decision (default: checkpoint setting)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sampling-seed", type=int, default=0, help="Reproducible per-trace sampling noise")
    args = ap.parse_args(argv)

    model, crop, n_hist, spec, ck = checkpoint_loader(args.checkpoint, args.device)
    if args.n_commit is None:
        args.n_commit = ck.get('n_commit', DEFAULT_N_COMMIT)
    if args.fiber_zarrs:
        spec.fiber_zarr_dir = args.fiber_zarrs
    if args.ct:
        spec.ct_zarr = args.ct
    if spec.mode == 'ct':
        spec.load_presence = False
    from .ct_normalization import prepare_normalization
    prepare_normalization(args.out, [spec], known=ck['ct_normalization'])
    vol = FiberVolume(spec, cache_bytes=8 << 30)
    tracer = tracer_class(model, vol, crop, n_hist, TraceParams(max_len=args.max_len, confidence=args.confidence,
        n_commit=args.n_commit, seed=args.sampling_seed), device=args.device)
    g = spec.grid_scale

    seeds = [np.array([float(v) for v in s.split(",")]) for s in args.seed]
    if args.seeds_file:
        seeds += [np.asarray(s, float) for s in json.load(open(args.seeds_file))]
    seeds = [s / g for s in seeds]
    if not seeds:
        raise SystemExit("no seeds given")
    families = np.asarray(args.family*len(seeds) if len(args.family) == 1 else args.family)
    if len(families) != len(seeds):
        raise SystemExit('--family must be supplied once, or once per seed')

    os.makedirs(args.out, exist_ok=True)
    meta = {"username": "fiber_follow", "fiber_manifest": ""}
    written, tree_pts = [], np.zeros((0, 3))
    t0 = time.time()
    n_skipped = 0
    for b in range(0, len(seeds), args.batch):
        chunk = np.stack(seeds[b : b + args.batch])
        chunk_families = families[b : b + args.batch]
        if len(tree_pts):
            d, _ = cKDTree(tree_pts).query(chunk)
            keep = d * g > args.dedupe
            n_skipped += int((~keep).sum())
            chunk = chunk[keep]
            chunk_families = chunk_families[keep]
            if not len(chunk):
                continue
        for seed_index, (poly, reasons) in enumerate(trace_bidirectional(tracer, vol, chunk, chunk_families)):
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
                                                                           "seed_family": str(chunk_families[seed_index]),
                                                                           "seed_heading_policy": SEED_HEADING_POLICY,
                                                                           "heading_policy": TRACE_HEADING_POLICY,
                                                                           "frame_policy": FRAME_POLICY,
                                                                           "sampling_seed": args.sampling_seed,
                                                                           "sampler_mode": getattr(model.cfg, 'sampler_mode', 'zero'),
                                                                           "checkpoint": os.path.abspath(args.checkpoint)}))
            with open(os.path.join(args.out, name), "w") as fh:
                json.dump(obj, fh)
            written.append(name)
            tree_pts = np.concatenate([tree_pts, poly], 0)
    dt = time.time() - t0
    print(json.dumps(dict(seeds=len(seeds), written=len(written), skipped=n_skipped, seconds=round(dt, 2),
                          out=os.path.abspath(args.out))))
