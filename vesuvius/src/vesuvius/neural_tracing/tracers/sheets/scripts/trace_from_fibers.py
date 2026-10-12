"""Trace sheets from VC3D fiber annotations (``vc3d_fiber`` JSON: ``control_points``, ``line_points``,
``hv_classification``) instead of GT seed rows.

For each chosen fiber of tag ``--tag`` (manual tag if set, else the automatic one): a control point is picked at
random, and the fiber's line points from there on, cut to ``(--points - 1) * point spacing`` voxels of arc length,
are the seed row (``--points`` front points). Fiber coordinates are multiplied by ``--coord-scale`` into the model's
level-0 voxels (0.25: fibers traced on the 2.4 um Scroll 1 scan, model on s1_ds2). The advance direction at each
point lies in the sheet, perpendicular to the fiber: sheet normal (2/8 CT structure tensor on level 0, as
the fiber tracer estimates normals) x fiber tangent, oriented ``--direction`` (up = +z, down, or both). Seeds whose
points lie outside the volume or on empty CT are skipped. Each trace runs ``--steps`` steps without early stop (like
the trace benchmark), and is written as ``OUT/fiber_FILESTEM_cpK_up|down`` tifxyz (kept points only; ``trace.json``
provenance, ``confidence.npy``, area in meta.json); with ``--direction both`` the two are one tifxyz
``OUT/fiber_FILESTEM_cpK`` (``tracing.inference.write_two_way_trace``: the down rows reversed, the seed row, the up
rows). With ``--follow`` each trace also writes the fiber it extends
as a VC3D fiber JSON to ``OUT/fibers/`` (``extended_fiber``).

  python scripts/trace_from_fibers.py --fibers ~/Desktop/spiral_dataset/fibers --checkpoint CKPT --out TRACES \\
      --count 50 --points 9 --steps 60 --path-map /remote/data/s1_ds2.zarr=/local/s1_ds2.zarr
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vesuvius.neural_tracing.tracers.shared.geometry import arclength
from vc3d_fiber_format import parse_vc3d_fiber_format

from vesuvius.neural_tracing.tracers.shared.annotations.vc3d import fiber_tag
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import ChunkedArray
from vesuvius.neural_tracing.tracers.shared.ct_structure import CT_CONTEXT_RADIUS, MIN_EIGENGAP, SeedHeadingError, ct_sheet_normal
from vesuvius.neural_tracing.tracers.sheets import config as C
from vesuvius.neural_tracing.tracers.sheets.checkpoint import load_checkpoint
from vesuvius.neural_tracing.tracers.sheets.data.fibers import episode_fibers, fiber_pool
from vesuvius.neural_tracing.tracers.sheets.tracing.inference import (
    connected_keep, resample, trace, write_trace, write_two_way_trace)


def seed_from_fiber(fiber: dict, scale: float, length: float, rng) -> tuple[np.ndarray, int] | None:
    """Line points (N, 3) from a random control point on, cut to ``length`` voxels of arc length (scaled
    coordinates), and the index of that control point; None if no control point has that much fiber after it."""
    line = np.asarray(fiber['line_points'], np.float64)*scale
    arc = arclength(line)
    cps = np.asarray(parse_vc3d_fiber_format(fiber).control_points_xyz, np.float64)*scale  # version 1 or 3
    starts = [int(np.argmin(np.linalg.norm(line-cp, axis=1))) for cp in cps]
    usable = [k for k, i in enumerate(starts) if arc[-1]-arc[i] >= length]
    if not usable:
        return None
    k = int(rng.choice(usable))
    i = starts[k]
    end = int(np.searchsorted(arc, arc[i]+length))
    pts = line[i:end+1].copy()
    over = arc[end]-(arc[i]+length)  # pull the last point back onto exactly ``length``
    if end > i and over > 0:
        seg = line[end]-line[end-1]
        pts[-1] = line[end]-seg*over/max(np.linalg.norm(seg), 1e-9)
    return pts, k


def sheet_normal(volume, point: np.ndarray) -> np.ndarray | None:
    """Sheet normal (xyz) at a level-0 point: the largest-eigenvalue axis of the 2/8 CT structure tensor
    (``shared.ct_structure.ct_sheet_normal``, as the fiber tracer estimates it) over ``CT_CONTEXT_RADIUS`` voxels;
    None without a clear sheet (relative eigengap below ``MIN_EIGENGAP``) or near the volume edge."""
    start = np.floor(point[::-1]).astype(np.int64)-CT_CONTEXT_RADIUS
    size = np.full(3, 2*CT_CONTEXT_RADIUS+1)
    if np.any(start < 0) or np.any(start+size > np.asarray(volume.shape)):
        return None
    try:
        return ct_sheet_normal(volume.read(start, size), point[::-1]-start, min_gap=MIN_EIGENGAP)
    except SeedHeadingError:
        return None


def advance_directions(points: np.ndarray, sign: float, volume) -> np.ndarray | None:
    """Per point, the in-sheet direction perpendicular to the fiber, pointing to ``sign`` z: the structure-tensor
    sheet normal (``volume``: the level-0 CT) x the fiber tangent. None if a point has no clear sheet."""
    t = np.gradient(points, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    normals = [sheet_normal(volume, p) for p in points]
    if any(n is None for n in normals):
        return None
    d = np.cross(np.stack(normals), t)
    d *= np.where(d[:, 2]*sign < 0, -1., 1.)[:, None]  # point up (or down)
    return d/np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)


def follow_starts(fiber: dict, scale: float, from_end: float) -> list:
    """For each end of the fiber long enough: ((line points (N, 3) scaled, arc length of the start ``from_end``
    voxels in from that end), sign of the outward direction along the line)."""
    line = np.asarray(fiber['line_points'], np.float64)*scale
    arc = arclength(line)
    if arc[-1] < 2*from_end:
        return []
    return [((line, from_end), -1.), ((line, arc[-1]-from_end), 1.)]


def follow_front(line: np.ndarray, at: float, sign: float, spacing: float, volume, span: float = 10.):
    """A 3-point front across the fiber at arc ``at`` (below, on, above it: one point spacing along the sheet,
    perpendicular to the fiber, from the 2/8 structure-tensor normal) and the advance direction at each point (the
    fiber's direction, ``sign`` along the line); (None, None) without a clear sheet."""
    arc = arclength(line)
    point = lambda a: np.array([np.interp(a, arc, line[:, k]) for k in range(3)])
    p = point(at)
    t = sign*(point(min(at+span, arc[-1]))-point(max(at-span, 0.)))
    t /= max(np.linalg.norm(t), 1e-9)
    n = sheet_normal(volume, p)
    if n is None:
        return None, None
    up = np.cross(n, t)
    up *= (1. if up[2] >= 0 else -1.)/max(np.linalg.norm(up), 1e-9)
    seed = np.stack([p-spacing*up, p, p+spacing*up])
    return seed, np.tile(t, (3, 1))


def extended_fiber(fiber: dict, line: np.ndarray, at: float, sign: float, rows: np.ndarray, confs: np.ndarray,
                   threshold: float, scale: float, name: str) -> dict | None:
    """A VC3D fiber (version-1 ``vc3d_fiber`` JSON, the source fiber's coordinates) extending ``fiber`` past its end:
    its line up to the trace's start (arc ``at`` of ``line``, the source line in model voxels), then the traced
    middle point (the one on the fiber) row by row while it stays kept (``connected_keep``). Control points: the
    line's ends. None if no row past the seed was kept."""
    keep = connected_keep(confs, threshold)
    mid = rows.shape[1]//2
    count = 1
    while count < len(rows) and keep[count, mid]:
        count += 1
    if count == 1:
        return None
    arc = arclength(line)
    traced = rows[1:count, mid]
    if sign > 0:  # extending past the line's last point
        points = np.concatenate([line[arc < at], rows[:1, mid], traced])
    else:  # past its first point: the extension first, in line order
        points = np.concatenate([traced[::-1], rows[:1, mid], line[arc > at]])
    points = (points/scale).round(3).tolist()
    hv = dict(fiber.get('hv_classification') or {}, automatic_tag='H')
    return dict(type='vc3d_fiber', optimization_mode='lasagna', username='sheet_tracer', generation=1,
                started_at=time.strftime('%Y%m%dT%H%M%S'), filename=f'{name}.json', hv_classification=hv,
                tags=['sheet_tracer_extension'], branches=[], control_points=[points[0], points[-1]],
                line_points=points, extended_from=fiber.get('filename', ''), extension_points=len(traced))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fibers', required=True, type=Path)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--tag', default='H')
    ap.add_argument('--count', type=int, default=50)
    ap.add_argument('--points', type=int, default=9)
    ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--direction', choices=('up', 'down', 'both'), default='up', help='advance along +z, -z or both')
    ap.add_argument('--coord-scale', type=float, default=.25)
    ap.add_argument('--threshold', type=float, default=.4)
    ap.add_argument('--suffix', default='', help='appended to each trace name')
    ap.add_argument('--follow', action='store_true', help='fiber-following mode (model.fiber_follow_mode): a 3-point '
                    'front across the fiber (middle on it, the others one point spacing above and below on the sheet) '
                    'tracing outward from each end, starting --from-end voxels in from it; --points is ignored')
    ap.add_argument('--from-end', type=float, default=120.)
    ap.add_argument('--fiber-pool', help='show the model the fibers around each trace (model.fiber_tokens): a pool '
                    'directory from scripts/build_fiber_pool.py, the traced fiber as the seed instance')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shard', default='0/1', metavar='I/N', help='this process takes every N-th fiber of the order, '
                    'from the I-th (run N processes with I = 0..N-1 and --count split between them)')
    ap.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    model, cfg, _ = load_checkpoint(args.checkpoint, args.device)
    C.path_mapper(args.path_map)(cfg)
    cfg.trace.confidence_threshold = args.threshold
    d = cfg.data
    volume = ChunkedArray(f'{d.volume}/{d.volume_level}', 1 << 28)
    level0 = ChunkedArray(f'{d.volume}/0', 1 << 28)  # sheet normals are estimated on level-0 CT
    shape = np.asarray(volume.shape)[::-1]*(1 << d.volume_level)  # xyz, level 0
    length = (args.points-1)*d.point_spacing
    rng = np.random.default_rng(args.seed)
    files = sorted(args.fibers.glob('*.json'))
    fibers = [(f, fiber) for f in files if fiber_tag(fiber := json.loads(f.read_text())) == args.tag]
    order = rng.permutation(len(fibers))
    shard, shards = (int(v) for v in args.shard.split('/'))
    order = order[shard::shards]
    signs = {'up': [1.], 'down': [-1.], 'both': [1., -1.]}[args.direction]
    pool = fiber_pool(args.fiber_pool) if args.fiber_pool else None
    reach = args.steps*d.step+2*d.point_spacing+100.

    def around(path, pts):
        """The fibers within reach of a trace from ``pts`` (the traced fiber as instance 0), or None."""
        if pool is None:
            return None
        return episode_fibers(pool, pts.min(0)-reach, pts.max(0)+reach, f'json:{path.stem}', None,
                              d.fiber_max_instances, d.fiber_max_points)

    done = 0
    print(f'{len(fibers)} {args.tag} fibers; tracing {args.count}' +
          (' in fiber-following mode' if args.follow else f' at {args.points} points') + f', {args.steps} steps',
          flush=True)
    if args.follow:
        for j in order:
            if done == args.count:
                break
            path, fiber = fibers[j]
            traced = False
            for end, (start, sign) in enumerate(follow_starts(fiber, args.coord_scale, args.from_end)):
                line, arc = start
                seed, direction = follow_front(line, arc, sign, d.point_spacing, level0)
                if seed is None or np.any(seed < 0) or np.any(seed >= shape):
                    continue
                idx = np.floor((seed[:, ::-1]+.5)/(1 << d.volume_level)-.5).astype(int)
                if np.mean([volume.read(i, (1, 1, 1))[0, 0, 0] > 0 for i in idx]) < .9:
                    continue
                rows, confs = trace(model, cfg, seed, direction, args.steps, args.device, follow=True,
                                    fibers=around(path, seed))
                name = f'fiber_{path.stem}_end{end}_follow{args.suffix}'
                extended = extended_fiber(fiber, line, arc, sign, rows, confs, args.threshold, args.coord_scale,
                                          f'fiber_{path.stem}_end{end}_extended')
                if extended is not None:  # the extended fiber, for VC3D's fiber tools
                    (args.out/'fibers').mkdir(parents=True, exist_ok=True)
                    (args.out/'fibers'/extended['filename']).write_text(json.dumps(extended))
                write_trace(args.out/name, rows, confs, cfg, args.threshold,
                            dict(fiber=str(path), end=end, from_end=args.from_end, follow=True, steps=args.steps,
                                 steps_traced=len(rows)-1, checkpoint=args.checkpoint, coord_scale=args.coord_scale,
                                 threshold=args.threshold, fiber_pool=args.fiber_pool), overwrite=True)
                meta = json.loads((args.out/name/'meta.json').read_text())
                grown = extended['extension_points'] if extended else 0
                print(f"{name}: {len(rows)-1} steps, {meta.get('area_cm2', 0):.3f} cm^2, fiber extended by {grown} "
                      f"points", flush=True)
                traced = True
            done += traced
        return
    for j in order:
        if done == args.count:
            break
        path, fiber = fibers[j]
        got = seed_from_fiber(fiber, args.coord_scale, length, rng)
        if got is None:
            continue
        points, cp = got
        if np.any(points < 0) or np.any(points >= shape):
            continue
        idx = np.floor((points[:, ::-1]+.5)/(1 << d.volume_level)-.5).astype(int)
        if np.mean([volume.read(i, (1, 1, 1))[0, 0, 0] > 0 for i in idx]) < .9:  # off the scan or on masked CT
            continue
        traced = {}
        for sign in signs:
            direction = advance_directions(points, sign, level0)
            if direction is None:
                continue
            seed = resample(points, d.point_spacing)
            dirs = np.stack([np.interp(np.linspace(0, len(points)-1, len(seed)), np.arange(len(points)), direction[:, k])
                             for k in range(3)], 1)
            traced[sign] = trace(model, cfg, seed, dirs, args.steps, args.device, fibers=around(path, seed))
        if not traced:
            continue
        # the file name, not the fiber's ``sequence``: sequence numbers repeat across annotators' files
        base = f'fiber_{path.stem}_cp{cp:02d}'
        provenance = dict(fiber=str(path), control_point=cp, points=len(seed), steps=args.steps, checkpoint=args.checkpoint,
                          coord_scale=args.coord_scale, threshold=args.threshold)
        if len(traced) == 2:  # one surface: the down trace reversed, the seed row, the up trace
            name = base+args.suffix
            write_two_way_trace(args.out/name, traced[1.], traced[-1.], cfg, args.threshold,
                                dict(provenance, direction='both', steps_traced_up=len(traced[1.][0])-1,
                                     steps_traced_down=len(traced[-1.][0])-1), overwrite=True)
            steps = f"{len(traced[1.][0])-1} up + {len(traced[-1.][0])-1} down steps"
        else:
            (sign, (rows, confs)), = traced.items()
            name = f"{base}_{'up' if sign > 0 else 'down'}{args.suffix}"
            write_trace(args.out/name, rows, confs, cfg, args.threshold,
                        dict(provenance, direction='+z' if sign > 0 else '-z', steps_traced=len(rows)-1), overwrite=True)
            steps = f'{len(rows)-1} steps'
        meta = json.loads((args.out/name/'meta.json').read_text())
        print(f"{name}: {len(seed)} points, {steps}, {meta.get('area_cm2', 0):.3f} cm^2", flush=True)
        done += 1


if __name__ == '__main__':
    main()
