"""Trace every fiber of a box of a volume from presence seeds, writing each fiber once (small overlaps kept for joining).

Coordinates are trace-grid xyz (``grid_scale`` base voxels; Paris4: 8 base voxels = 19.2 um, the CT read at level 2,
9.6 um). The presence zarr level must be the trace grid (Paris4 lasagna presence level 3); the next level (2x
coarser, e.g. level 4) weighs tiles for progress and skips empty ones.

Seeds: cross-section dots. The box is processed in tiles (--tile grid voxels, z-major order). Inside a tile, planes
lie at global multiples of --plane-spacing: z planes for vertical (V) fibers, x planes for horizontal (H) fibers
running mostly along x, y planes for H fibers running mostly along y. In each plane, the presence (smoothed, sigma 1)
local maxima within a (2 * --dot-radius + 1)^2 window above --threshold are dots. A dot keeps its fiber direction
from the presence Hessian (sigma 1.5): the weakest axis of a ridge whose two other curvatures are negative and at least
twice as strong. Its family is V when |dir_z| >= 0.75 and H when <= 0.45, and it is used only on the plane its fiber
crosses steeply (|dir . normal| >= 0.7); other dots are dropped. Dots are taken plane by plane, strongest first.
Then a fill-in pass seeds from the tile's 3D ridge maxima (same ridge and family tests, --fill-spacing apart,
strongest first; skipped within --fill-skip of a parallel written fiber) for fibers that cross no plane steeply
inside the tile.

Tracing streams: about --rows traces run at once (each dot traced both ways from the CT seed heading of its family;
a dot whose CT heading is more than 35 degrees from its presence direction is skipped), and new dots start as traces
end, in tile order, one tracer call per z row of tiles. A dot is skipped when it lies within --dedupe of a parallel
(|cos| > 0.8) written fiber, and waits while it is within --batch-spread of a running seed. Every step, for all
running traces at once, a trace stops
- at the box (beyond a 4 voxel margin),
- after --outside-decisions decisions with its head outside the scan (CT value 0: Paris4's CT is 0 beyond the scanned
  and masked region; a scan without zeros never stops this way),
- after --stop-decisions decisions with its head within --dedupe of a parallel written fiber ('owned') or of the path
  another running trace laid down in an earlier step ('live'; a seed's two directions ignore each other).

Writing: when both directions of a dot have ended, they form one polyline, cut to the box and to the scan (points at
CT 0 at either end, and interior runs of more than --outside-gap of them, are removed; outside_ct_cut). A run of more than
--trim-run points within --dedupe of a parallel written fiber is removed, keeping --overlap points at each side;
shorter contacts (such as the stretch that stopped a trace) stay. Pieces of at least --min-length points, at most half
on written fibers, are written, each with its stop reasons and the fibers its ends touch ('links': within --dedupe + 1,
parallel), so overlapping pieces of one fiber can be joined later.

Outputs (OUT): tiles/TILE.npz (the pieces of the tile's dots in grid xyz, ids, seeds, families, stop reasons, links;
written last, it marks the tile done), fibers/TILE/*.json (VC3D fibers in base coordinates; --no-json skips them),
tiles.jsonl (one line per finished tile), progress.json, run.json; status lines on stdout; at the end fibers.vcobj,
a VC object store of every fiber for VC3D (volume_store.py; native L0 xyz bound to the volume of the open-data CT, or
of --fibers-template; fiber UUIDs derived from run.json's fiber_namespace). Rerunning with the same OUT resumes: finished tiles are kept (their fibers own their space), the
rest are redone; settings that change the fibers must be the same.

  python -m vesuvius.neural_tracing.tracers.fibers.tracing.volume --presence PRESENCE_LEVEL \\
      --out OUT --ct-z 7000,17000 [--box x0,y0,z0,x1,y1,z1] [--checkpoint CKPT | ORG/NAME]

CT, chunk cache and frame model: the checkpoint's, on this machine (tracing/runtime.py; --ct, --cache-dir,
--frame-checkpoint).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import zarr
from scipy import ndimage
from .volume_policy import (KDForest, Owned, LivePaths, PARALLEL, unit, tangents, runs,
                            outside_scan, outside_ct_cut, novel_parts, seed_deferred)

from vesuvius.neural_tracing.tracers.fibers.frames.ct import SEED_HEADING_POLICY, ct_seed_heading
from vesuvius.neural_tracing.tracers.fibers.frames.learned import trace_family_kwargs
from vesuvius.neural_tracing.tracers.fibers.tracing.inference import make_fiber_json
from vesuvius.neural_tracing.tracers.fibers.tracing.runtime import (
    add_checkpoint_argument, add_volume_arguments, checkpoint_source, configure_volume, resolve_checkpoint, use_gpu_crops)
from vesuvius.neural_tracing.tracers.shared.annotations.open_data import file_sha256
from vesuvius.neural_tracing.tracers.fibers.tracing.trace import FiberTracer, TraceParams
from vesuvius.neural_tracing.tracers.fibers.tracing.volume_store import export_store
from vesuvius.neural_tracing.tracers.models.checkpoint import load_checkpoint
from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import prepare_normalization
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolume
from vesuvius.neural_tracing.tracers.shared.ct_structure import SeedHeadingError
from vesuvius.neural_tracing.tracers.shared.geometry import arclength, resample_polyline
from vesuvius.neural_tracing.tracers.shared.policy import checkpoint_policy

PARALLEL = 0.8  # |cos| above which two fiber directions count as the same fiber direction
BOX_MARGIN = 4.
HESSIAN_SIGMA, HESSIAN_RADIUS = 1.5, 6  # scipy's truncate 4: radius int(4 * 1.5 + 0.5)


# ---------------------------------------------------------------------------------------------------------- seeds

def _hessian_kernels():
    k = np.arange(-HESSIAN_RADIUS, HESSIAN_RADIUS+1, dtype=np.float64)
    s2 = HESSIAN_SIGMA**2
    g = np.exp(-k**2/(2*s2))
    g /= g.sum()
    d = {0: g, 1: -k/s2*g, 2: (k**2-s2)/s2**2*g}
    kernels = []
    for a in range(3):
        for b in range(a, 3):
            order = [0, 0, 0]
            order[a] += 1
            order[b] += 1
            kz, ky, kx = (d[o][::-1] for o in order)  # convolution, as scipy.ndimage.gaussian_filter
            kernels.append(np.einsum('i,j,k->ijk', kz, ky, kx))
    return np.stack(kernels)  # (6, 13, 13, 13): zz, zy, zx, yy, yx, xx


_KERNELS = _hessian_kernels()


def ridge_directions(presence, points_zyx):
    """Fiber direction xyz and ridge validity at integer zyx points of a raw presence block (Hessian, sigma 1.5)."""
    r = HESSIAN_RADIUS
    offsets = np.arange(-r, r+1)
    zi = points_zyx[:, 0, None]+offsets
    yi = points_zyx[:, 1, None]+offsets
    xi = points_zyx[:, 2, None]+offsets
    patches = presence[zi[:, :, None, None], yi[:, None, :, None], xi[:, None, None, :]].astype(np.float64)
    h6 = np.einsum('nijk,cijk->nc', patches, _KERNELS)
    hess = np.empty((len(points_zyx), 3, 3))
    for c, (a, b) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
        hess[:, a, b] = hess[:, b, a] = h6[:, c]
    w, v = np.linalg.eigh(hess)
    weakest = np.argmin(np.abs(w), 1)
    direction = unit(v[np.arange(len(w)), :, weakest][:, ::-1])  # zyx eigenvector -> xyz
    others = np.where(np.arange(3)[None] == weakest[:, None], np.nan, w)
    strong = np.nanmax(others, 1)  # the weaker of the two cross curvatures (both must be negative)
    ridge = (strong < 0) & (np.abs(strong) > 2*np.abs(w[np.arange(len(w)), weakest]))
    return direction, ridge


def tile_dots(presence_array, lo, hi, args):
    """Seeds of one tile [lo, hi) (grid xyz): positions, directions, families, presence values, in tracing order."""
    margin = HESSIAN_RADIUS+2
    want_lo = lo[::-1].astype(int)-margin
    want_hi = np.ceil(hi[::-1]).astype(int)+margin
    start = np.maximum(want_lo, 0)
    stop = np.minimum(want_hi, presence_array.shape)
    raw = np.asarray(presence_array[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]])
    if raw.max(initial=0) <= args.threshold:
        return None
    raw = np.pad(raw, [(int(start[k]-want_lo[k]), int(want_hi[k]-stop[k])) for k in range(3)])
    raw = raw.astype(np.float32)
    smooth = ndimage.gaussian_filter(raw, 1.0)
    size = 2*int(args.dot_radius)+1
    plans = (('z', 2, 0, 'V'), ('x', 0, 2, 'H'), ('y', 1, 1, 'H'))  # name, xyz axis, zyx axis, family
    out = []
    for name, axis_xyz, axis_zyx, family in plans:
        first = int(np.ceil(lo[axis_xyz]/args.plane_spacing)*args.plane_spacing)
        for c in range(first, int(hi[axis_xyz]), int(args.plane_spacing)):
            k = c-int(lo[axis_xyz])+margin
            plane = np.take(smooth, k, axis=axis_zyx)
            peak = (plane == ndimage.maximum_filter(plane, size)) & (plane > args.threshold)
            ij = np.argwhere(peak)
            if not len(ij):
                continue
            zyx = np.insert(ij, axis_zyx, k, axis=1)
            inside = np.all((zyx >= margin) & (zyx < np.asarray(smooth.shape)-margin), 1)
            zyx = zyx[inside]
            if not len(zyx):
                continue
            direction, ridge = ridge_directions(raw, zyx)
            dz = np.abs(direction[:, 2])
            fam_ok = (dz >= .75) if family == 'V' else (dz <= .45)
            steep = np.abs(direction[:, axis_xyz]) >= .7
            keep = ridge & fam_ok & steep
            if not keep.any():
                continue
            zyx, direction = zyx[keep], direction[keep]
            value = smooth[tuple(zyx.T)]
            order = np.argsort(-value, kind='stable')
            xyz = (zyx[order]-margin)[:, ::-1]+lo
            out.append(dict(xyz=xyz.astype(np.float64), direction=direction[order], value=value[order],
                            family=np.full(len(order), family), plane=np.full(len(order), f'{name}{c}')))
    # Fill-in: 3D ridge maxima (at least --fill-spacing apart, strongest first) catch fibers that cross no plane
    # steeply inside the tile; most of them lie on fibers written by then and are skipped as owned.
    peak = (smooth == ndimage.maximum_filter(smooth, 3)) & (smooth > args.threshold)
    zyx = np.argwhere(peak)
    zyx = zyx[np.all((zyx >= margin) & (zyx < np.asarray(smooth.shape)-margin), 1)]
    if len(zyx):
        direction, ridge = ridge_directions(raw, zyx)
        dz = np.abs(direction[:, 2])
        family = np.where(dz >= .75, 'V', np.where(dz <= .45, 'H', ''))
        keep = ridge & (family != '')
        zyx, direction, family = zyx[keep], direction[keep], family[keep]
        value = smooth[tuple(zyx.T)]
        order = np.argsort(-value, kind='stable')
        chosen, kept_pts = [], []
        for i in order:
            if kept_pts and np.min(np.linalg.norm(np.asarray(kept_pts)-zyx[i], axis=1)) < args.fill_spacing:
                continue
            chosen.append(i)
            kept_pts.append(zyx[i])
        chosen = np.asarray(chosen, int)
        if len(chosen):
            out.append(dict(xyz=((zyx[chosen]-margin)[:, ::-1]+lo).astype(np.float64), direction=direction[chosen],
                            value=value[chosen], family=family[chosen], plane=np.full(len(chosen), 'fill')))
    if not out:
        return None
    return {k: np.concatenate([o[k] for o in out]) for k in out[0]}


# ------------------------------------------------------------------------------------------------------- ownership

# --------------------------------------------------------------------------------------------------------- tracing

def build_tracer(args):
    path = resolve_checkpoint(args.checkpoint)
    model, crop, n_hist, spec, ck = load_checkpoint(path, args.device)
    if args.precision == 'fp32':
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    policy = checkpoint_policy(ck, model.cfg, confidence=args.confidence, n_commit=args.n_commit, gate=args.gate)
    configure_volume(args, spec, model.cfg, path)
    prepare_normalization(args.out, [spec], known=ck['ct_normalization'])
    vol = CTVolume(spec, cache_bytes=8 << 30)
    params = TraceParams.from_policy(policy, max_len=args.max_len, precision=args.precision,
                                     forward_chunk=args.forward_chunk, seed=args.sampling_seed)
    tracer = FiberTracer(model, vol, crop, n_hist, params, device=args.device)
    if getattr(args, 'cuda_graph', False):
        if args.forward_chunk <= 0 or getattr(model.cfg, 'model_type', '') != 'regression':
            raise ValueError('CUDA graph inference requires regression and a positive forward chunk')
        from vesuvius.neural_tracing.tracers.shared.inference_graph import InferenceGraph
        tracer.forward_model = InferenceGraph(model, args.forward_chunk)
    tracer.checkpoint_path = path
    if args.axis_workers:
        # Continuation crop axes in worker processes (spawned: the parent holds CUDA); same values as in-process.
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        tracer.axis_pool = ProcessPoolExecutor(args.axis_workers, mp_context=multiprocessing.get_context('spawn'),
                                               initializer=_worker_init, initargs=(spec.to_dict(),))
        tracer.observations.geometry_pool = tracer.axis_pool
    if args.crops == 'gpu':
        use_gpu_crops(tracer, vol, args.device, args.gpu_chunks)
    return tracer, vol, spec, policy


_WORKER_VOL = None


def _worker_init(spec):
    """Worker process: its own CT reader (the volume's spec, normalization included)."""
    global _WORKER_VOL
    from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolumeSpec
    _WORKER_VOL = CTVolume(CTVolumeSpec.from_dict(spec), cache_bytes=1 << 30)


def _seed_heading(task):
    """CT seed heading of one dot, signed along its presence direction; None when unusable."""
    xyz, family, direction = task
    try:
        h = ct_seed_heading(_WORKER_VOL, xyz, family)
    except SeedHeadingError:
        return None
    if abs(h @ direction) < np.cos(np.radians(35)):
        return None
    return h if h @ direction > 0 else -h


def seed_headings(vol, dots, pool):
    """CT seed headings (signed along the presence direction) in the worker processes; rows without a usable one
    are dropped. (numpy/scipy work in tracer threads crashed CPython 3.14, so it runs in processes.)"""
    tasks = list(zip(dots['xyz'], dots['family'], dots['direction']))
    # Each seed has substantial independent CT work. Dispatch individually so
    # workers can take the next seed immediately; map retains input order.
    heads = list(pool.map(_seed_heading, tasks, chunksize=1))
    keep = np.array([h is not None for h in heads], bool)
    out = {k: v[keep] for k, v in dots.items()}
    out['heading'] = np.asarray([h for h in heads if h is not None]).reshape(-1, 3)
    return out, keep


_TILE_PRESENCE = None


def _tile_worker_init(presence_path):
    global _TILE_PRESENCE
    _TILE_PRESENCE = zarr.open(presence_path, mode='r')


def _tile_dots(task):
    lo, hi, args = task
    return tile_dots(_TILE_PRESENCE, lo, hi, args)


class TilePrefetch:
    """tile_dots of the coming tiles in worker processes, up to ``depth`` tiles ahead of the one read (the same
    function on the same presence: the same dots as in the main process)."""

    def __init__(self, presence_path, args, tiles, workers, depth):
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        self.pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn'),
                                        initializer=_tile_worker_init, initargs=(presence_path,))
        self.args, self.depth = args, depth
        self.order = [t for t in tiles if t['weight'] > 0]  # the tiles with presence, in reading order
        self.index = {t['name']: k for k, t in enumerate(self.order)}
        self.futures = {}
        self.submitted = 0

    def get(self, tile):
        k = self.index[tile['name']]
        while self.submitted < min(len(self.order), k+1+self.depth):
            t = self.order[self.submitted]
            self.futures[t['name']] = self.pool.submit(_tile_dots, (t['lo'], t['hi'], self.args))
            self.submitted += 1
        return self.futures.pop(tile['name']).result()

    def close(self):
        self.pool.shutdown(cancel_futures=True)


class Streamer:
    """Feeds tile dots to one streaming tracer call, stops traces each step, writes each dot when both directions end."""

    def __init__(self, tracer, vol, owned, lo, hi, args, out, spec, provenance, next_id, tiles):
        self.tracer, self.vol, self.owned, self.lo, self.hi, self.args = tracer, vol, owned, lo, hi, args
        self.out, self.spec, self.provenance, self.next_id = out, spec, provenance, next_id
        self.tiles = tiles  # list of dict(name, lo, hi, weight), in order; dots read when reached
        self.cursor = 0  # next tile to read
        self.open = []  # tiles with dots left or traces running, in order
        self.rows = {}  # tracer row -> (pair, half)
        self.pairs = {}  # pair -> dict(tile, seed, family, plane, halves, reasons)
        self.prev = {}  # row -> previous head
        self.streak = defaultdict(int)
        self.outside_streak = defaultdict(int)
        self.live = LivePaths(args.dedupe)
        self.step = 0
        self.row_count = 0
        self.decisions = 0
        self.finished_tiles = []
        self.read_weight = 0  # presence weight of the tiles read (their dots started or queued)
        self.timing = defaultdict(float)  # seconds in feed (incl. tile reads), per-step checks, writes
        self.prefetch = None  # TilePrefetch, else tiles are read in the main process

    # -- tiles
    def _read_next_tile(self):
        tile = self.tiles[self.cursor]
        self.cursor += 1
        t0 = time.time()
        if tile['weight'] <= 0:
            dots = None
        elif self.prefetch is not None:
            dots = self.prefetch.get(tile)
        else:
            dots = tile_dots(self.presence, tile['lo'], tile['hi'], self.args)
        tile.update(dots=dots, pending=list(range(len(dots['xyz']))) if dots is not None else [], running=0,
                    pieces=[], stats=defaultdict(int), t0=t0)
        self.read_weight += tile['weight']
        tile['stats']['dots'] = 0 if dots is None else len(dots['xyz'])
        self.open.append(tile)

    def _close_tiles(self):
        for tile in [t for t in self.open if not t['pending'] and t['running'] == 0]:  # each when its dots are written
            self.open.remove(tile)
            save_tile(self.out, tile['name'], tile['pieces'], self.spec, self.provenance, self.args.json)
            kept = int(sum(arclength(p['path'])[-1] for p in tile['pieces']))
            line = dict(tile=tile['name'], seconds=round(time.time()-tile['t0'], 1), fibers=len(tile['pieces']),
                        kept_vox=kept, weight=tile['weight'], **tile['stats'])
            with open(self.out/'tiles.jsonl', 'a') as fh:
                fh.write(json.dumps(line)+'\n')
            (self.out/'progress.json').write_text(json.dumps(dict(next_id=self.next_id)))
            self.finished_tiles.append(line)
            tile['dots'] = None

    # -- tracer hooks
    def feed(self, active):
        t0 = time.time()
        try:
            return self._feed(active)
        finally:
            self.timing['feed'] += time.time()-t0

    def _feed(self, active):
        args = self.args
        free = (args.rows-active)//2
        if free < min(args.min_feed, args.rows//4):
            return None
        chosen = []  # (tile, k)
        running = np.array([p['seed'] for p in self.pairs.values()]).reshape(-1, 3)
        scanned = set()
        while len(chosen) < free:
            self._close_tiles()
            tile = next((t for t in self.open if t['pending'] and id(t) not in scanned), None)
            if tile is None:  # nothing startable in the open tiles: read ahead
                if self.cursor >= len(self.tiles) or len(self.open) >= args.max_open:
                    break
                self._read_next_tile()
                continue
            scanned.add(id(tile))
            dots, later = tile['dots'], []
            pend = np.asarray(tile['pending'])
            picked = [c[1] for c in chosen if c[0] is tile]
            at = 0
            while at < len(pend) and len(chosen) < free:  # only as many pending dots as needed, in order
                part = pend[at:at+max(64, 4*(free-len(chosen)))]
                at += len(part)
                radius = np.where(dots['plane'][part] == 'fill', args.fill_skip, args.dedupe)
                owned = np.zeros(len(part), bool)
                for r in np.unique(radius):
                    m = radius == r
                    owned[m] = self.owned.owner(dots['xyz'][part][m], dots['direction'][part][m], r) >= 0
                tile['stats']['owned_dots'] += int(owned.sum())
                for k, own in zip(part.tolist(), owned):
                    if own:
                        continue
                    if len(chosen) >= free:
                        later.append(k)
                        continue
                    near = np.concatenate([running, dots['xyz'][picked]]) if picked else running
                    if seed_deferred(dots['xyz'][k], near, args.batch_spread):
                        later.append(k)  # waits for the running seed nearby
                        continue
                    chosen.append((tile, k))
                    picked.append(k)
                    tile['running'] += 1  # reserved: the tile stays open until this dot is written
            tile['pending'] = later+pend[at:].tolist()
        if not chosen:
            self._close_tiles()
            return None
        batch = dict(xyz=np.array([t['dots']['xyz'][k] for t, k in chosen]),
                     direction=np.array([t['dots']['direction'][k] for t, k in chosen]),
                     family=np.array([t['dots']['family'][k] for t, k in chosen]),
                     plane=np.array([t['dots']['plane'][k] for t, k in chosen]))
        batch, keep = seed_headings(self.vol, batch, self.tracer.axis_pool)
        tile_by_index = [t for (t, _), ok in zip(chosen, keep) if ok]
        for (t, _), ok in zip(chosen, keep):
            t['stats']['traced' if ok else 'no_heading'] += 1
            t['running'] -= int(not ok)
        self._close_tiles()
        m = len(batch['xyz'])
        if not m:
            return None
        base = self.row_count
        for j in range(m):
            pair = base+j  # pair id = its forward row
            self.pairs[pair] = dict(tile=tile_by_index[j], seed=batch['xyz'][j], family=str(batch['family'][j]),
                                    plane=str(batch['plane'][j]), halves={}, reasons={})
            self.rows[base+j] = (pair, 0)
            self.rows[base+m+j] = (pair, 1)
            self.prev[base+j] = self.prev[base+m+j] = batch['xyz'][j]
        self.row_count += 2*m
        return (np.concatenate([batch['xyz'], batch['xyz']]), np.concatenate([batch['heading'], -batch['heading']]),
                list(batch['family'])*2)

    def before_step(self, idx, pos, frames):
        t0 = time.time()
        try:
            return self._before_step(idx, pos, frames)
        finally:
            self.timing['checks'] += time.time()-t0

    def _before_step(self, idx, pos, frames):
        self.step += 1
        self.decisions += len(idx)
        args = self.args
        dirs = frames[:, :, 2]
        stops = {}
        outside = np.any(pos < self.lo-BOX_MARGIN, 1) | np.any(pos > self.hi+BOX_MARGIN, 1)
        unscanned = outside_scan(self.vol, pos)
        pair = np.array([self.rows[i][0] for i in idx])
        near = (self.owned.owner(pos, dirs, args.dedupe) >= 0) | self.live.hits(pos, dirs, pair, self.step)
        prev = np.array([self.prev[i] for i in idx])
        self.live.add(prev, pos, pair, self.step)
        for i, p in zip(idx, pos):
            self.prev[i] = p
        for i, out_, nr, un in zip(idx, outside, near, unscanned):
            if out_:
                stops[int(i)] = 'bounds'
                continue
            self.outside_streak[i] = self.outside_streak[i]+1 if un else 0
            if self.outside_streak[i] >= args.outside_decisions:
                stops[int(i)] = 'outside_ct'
                continue
            self.streak[i] = self.streak[i]+1 if nr else 0
            if self.streak[i] >= args.stop_decisions:
                stops[int(i)] = 'owned_or_live'
        return stops

    def on_finished(self, i, path, reason):
        t0 = time.time()
        try:
            self._on_finished(i, path, reason)
        finally:
            self.timing['write'] += time.time()-t0

    def _on_finished(self, i, path, reason):
        pair, half = self.rows.pop(i)
        self.prev.pop(i, None)
        self.streak.pop(i, None)
        self.outside_streak.pop(i, None)
        info = self.pairs[pair]
        info['halves'][half] = path
        info['reasons'][half] = reason
        if len(info['halves']) == 2:
            del self.pairs[pair]
            self.live.drop([pair])
            self._write(info)
            info['tile']['running'] -= 1
            for r in (info['reasons'][1], info['reasons'][0]):
                info['tile']['stats'][f'stop_{r}'] += 1

    # -- writing
    def _write(self, info):
        args = self.args
        fw, bw = info['halves'][0], info['halves'][1]
        poly = np.concatenate([bw[::-1], fw[1:]])  # backward reversed, then forward
        info['tile']['stats']['traced_vox'] += int(arclength(poly)[-1]) if len(poly) > 1 else 0
        inside = np.all((poly >= self.lo) & (poly <= self.hi), 1) & \
            ~outside_ct_cut(outside_scan(self.vol, poly), args.outside_gap)
        new = []
        for a, b in runs(inside):
            part = poly[a:b]
            if len(part) < 2:
                continue
            for s, e in novel_parts(part,self.owned,args.dedupe,args.trim_run,args.overlap,args.min_length):
                new.append(part[s:e])
        for piece in new:
            fid = self.next_id
            self.next_id += 1
            t = tangents(piece)
            links = self.owned.owner(piece[[0, -1]], t[[0, -1]], args.dedupe+1)
            self.owned.add(piece, fid)
            info['tile']['pieces'].append(dict(id=fid, path=piece, seed=info['seed'], family=info['family'],
                                               plane=info['plane'], reasons=(info['reasons'][1], info['reasons'][0]),
                                               links=[int(v) for v in links]))


# ---------------------------------------------------------------------------------------------------------- output

def save_tile(out, name, pieces, spec, provenance, write_json):
    fibers = out/'fibers'/name
    if write_json and pieces:
        fibers.mkdir(parents=True, exist_ok=True)
        g = spec.grid_scale
        for p in pieces:
            fname = f'{provenance["prefix"]}{p["id"]:08d}.json'
            meta = dict(username='fiber_follow', filename=fname, fiber_manifest='',
                        fiber_follow=dict(id=p['id'], tile=name, seed=np.asarray(p['seed']).tolist(),
                                          seed_family=p['family'], seed_plane=p['plane'], stop_reasons=list(p['reasons']),
                                          links=p['links'], **provenance['trace']))
            with open(fibers/fname, 'w') as fh:
                json.dump(make_fiber_json(resample_polyline(p['path']*g, g), 800., meta), fh)
    pts = [p['path'] for p in pieces]
    offsets = np.cumsum([0]+[len(p) for p in pts])
    tmp = out/'tiles'/f'{name}.partial.npz'
    np.savez(tmp, points=np.concatenate(pts).astype(np.float32) if pts else np.zeros((0, 3), np.float32),
             offsets=offsets, id=np.array([p['id'] for p in pieces], int),
             seed=np.array([p['seed'] for p in pieces]).reshape(-1, 3), family=np.array([p['family'] for p in pieces]),
             plane=np.array([p['plane'] for p in pieces]), reasons=np.array([p['reasons'] for p in pieces]).reshape(-1, 2),
             links=np.array([p['links'] for p in pieces], int).reshape(-1, 2))
    os.replace(tmp, out/'tiles'/f'{name}.npz')


def load_done(out, owned):
    """Finished tiles' pieces back into ``owned``; returns (done tile names, next fiber id)."""
    done, next_id = set(), 0
    started = time.monotonic()
    for f in sorted((out/'tiles').glob('*.npz')):
        if f.name.endswith('.partial.npz'):
            continue
        # NpzFile does not cache members: d['points'] inside the fiber loop
        # rereads the entire tile array for every fiber in that tile.
        with np.load(f) as d:
            o, ids, points = d['offsets'], d['id'], d['points']
            for fid, a, b in zip(ids, o[:-1], o[1:]):
                owned.add(points[a:b].astype(np.float64), int(fid))
                next_id = max(next_id, int(fid)+1)
        done.add(f.stem)
        if len(done) % 100 == 0:
            print(f'restore: tiles={len(done)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    owned.flush()
    progress = out/'progress.json'
    if progress.exists():  # ids issued before a stop may be linked from saved tiles: never reuse them
        next_id = max(next_id, int(json.loads(progress.read_text())['next_id']))
    return done, next_id


def tile_weights(presence_path, threshold, tiles, shape):
    """Presence voxels above ``threshold`` per tile, from the 2x coarser pyramid level (None if absent)."""
    coarse = Path(presence_path).parent/str(int(Path(presence_path).name)+1)
    if not coarse.exists():
        return None
    a = zarr.open(str(coarse), mode='r')
    weights = []
    for t in tiles:
        z0, y0, x0 = (t['lo'][::-1]//2).astype(int)
        z1, y1, x1 = np.ceil(t['hi'][::-1]/2).astype(int)
        weights.append(int((np.asarray(a[z0:z1, y0:y1, x0:x1]) > threshold).sum()))
    return weights


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    add_checkpoint_argument(ap)
    ap.add_argument('--presence', required=True, help='presence zarr array on the trace grid (uint8 level directory)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--box', help='grid xyz x0,y0,z0,x1,y1,z1 (default: the whole volume, z from --ct-z)')
    ap.add_argument('--ct-z', help="z range in CT voxels of the model's CT level, e.g. 7000,17000")
    ap.add_argument('--prefix', default='fiber_', help='fiber file name prefix (distinct per box when merging runs)')
    ap.add_argument('--tile', type=float, default=256.)
    ap.add_argument('--plane-spacing', type=float, default=64.)
    ap.add_argument('--threshold', type=float, default=128., help='presence value of a seed dot')
    ap.add_argument('--dot-radius', type=float, default=2.)
    ap.add_argument('--fill-spacing', type=float, default=4., help='spacing of the fill-in 3D ridge seeds')
    ap.add_argument('--fill-skip', type=float, default=4.,
                    help='fill-in seeds this close to a parallel written fiber are skipped (presence maxima sit off the '
                         'traced centre line)')
    ap.add_argument('--rows', type=int, default=512, help='traces running at once (two per dot)')
    ap.add_argument('--min-feed', type=int, default=32, help='dots started at least at once')
    ap.add_argument('--max-open', type=int, default=64, help='tiles read ahead at most, to keep --rows traces running')
    ap.add_argument('--batch-spread', type=float, default=16., help='min distance of a new dot from running seeds')
    ap.add_argument('--dedupe', type=float, default=2., help='same-fiber radius, grid voxels')
    ap.add_argument('--stop-decisions', type=int, default=3)
    ap.add_argument('--outside-decisions', type=int, default=2,
                    help='a trace stops after this many decisions with its head at CT 0 (outside the scan)')
    ap.add_argument('--outside-gap', type=int, default=16,
                    help='interior runs of more than this many points at CT 0 are cut from written fibers (end runs always)')
    ap.add_argument('--overlap', type=int, default=16, help='points kept on each side of a removed overlap')
    ap.add_argument('--trim-run', type=int, default=24, help='overlaps longer than this many points are removed')
    ap.add_argument('--min-length', type=int, default=48, help='points (grid voxels) per written fiber')
    ap.add_argument('--max-len', type=float, default=8000., help='per direction, grid voxels')
    ap.add_argument('--confidence', type=float, default=0.5)
    ap.add_argument('--n-commit', type=int)
    ap.add_argument('--gate', choices=('full', 'prefix'))
    ap.add_argument('--precision', choices=('bf16', 'fp32'), default='bf16')
    ap.add_argument('--crops', choices=('gpu', 'cpu'), default='gpu',
                    help='GPU: follower and frame-model crops from a GPU chunk cache (shared/crop_sampling/gpu.py) and '
                         'the frame model on the GPU; CPU: the CPU sampler and CPU frame model')
    ap.add_argument('--gpu-chunks', type=int, default=2048, help='GPU chunk cache slots (2 MiB each for 128^3 uint8)')
    ap.add_argument('--forward-chunk', type=int, default=128)
    ap.add_argument('--cuda-graph', action='store_true', help='Replay full regression forward chunks with a CUDA graph')
    ap.add_argument('--axis-workers', type=int, default=8,
                    help='worker processes for continuation crop axes and seed headings (at least 1)')
    ap.add_argument('--tile-workers', type=int, default=3,
                    help='worker processes reading the coming tiles\' seed dots ahead (0: read in the main process)')
    ap.add_argument('--tile-ahead', type=int, default=8, help='tiles with presence read ahead by the tile workers')
    ap.add_argument('--sampling-seed', type=int, default=0)
    add_volume_arguments(ap)
    ap.add_argument('--no-json', dest='json', action='store_false')
    ap.add_argument('--fibers-template',
                    help='fiber store of the same scan: the volume of OUT/fibers.vcobj, written at the end (default: '
                         "the volume of the CT, when a Vesuvius open-data store; volume_store.py writes it any time "
                         'from the finished tiles)')
    ap.add_argument('--no-fibers', dest='fibers', action='store_false')
    ap.add_argument('--status-every', type=float, default=60., help='seconds between status lines')
    ap.add_argument('--base-um', type=float, default=2.4, help='base voxel size for reported lengths (Paris4 2.4)')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    if args.axis_workers < 1:
        raise SystemExit('--axis-workers must be at least 1 (seed headings run in worker processes)')
    # SIGTERM exits through the finally below, which shuts down the axis worker processes (else they are orphaned)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    out = Path(args.out)
    (out/'tiles').mkdir(parents=True, exist_ok=True)
    tracer, vol, spec, policy = build_tracer(args)
    presence = zarr.open(args.presence, mode='r')
    if tuple(presence.shape) != tuple(vol.shape):
        raise SystemExit(f'presence shape {presence.shape} is not the trace grid {vol.shape} (zyx)')
    if args.box:
        b = np.array([float(v) for v in args.box.split(',')])
        lo, hi = b[:3], b[3:]
    else:
        lo, hi = np.zeros(3), np.array(vol.shape[::-1], float)
        if args.ct_z:
            z0, z1 = (float(v)*spec.ct_grid_scale/spec.grid_scale for v in args.ct_z.split(','))
            lo[2], hi[2] = z0, z1
    record = dict(vars(args), box=dict(lo=lo.tolist(), hi=hi.tolist()), operating_policy=policy.to_dict(),
                  seed_heading_policy=SEED_HEADING_POLICY, grid_scale=spec.grid_scale, ct_zarr=spec.ct_zarr,
                  checkpoint_sha256=file_sha256(tracer.checkpoint_path))
    run_json = out/'run.json'
    if run_json.exists():
        old = json.loads(run_json.read_text())
        record['fiber_namespace'] = old.get('fiber_namespace')  # drawn when the run started
        # settings that only change speed or reporting may differ on resume
        speed_only = ('device', 'status_every', 'rows', 'min_feed', 'open_tiles', 'max_open', 'forward_chunk', 'gpu_chunks', 'cuda_graph',
                      'axis_workers', 'tile_workers', 'tile_ahead', 'fibers', 'fibers_template',  # speed or the end-of-run export
                      'cache_dir', 'frame_checkpoint')  # locations (the frame model is pinned by its sha256)
        # derived from the (compared) checkpoint and --ct; absent from runs started before they were recorded
        derived = ('ct_zarr', 'checkpoint_sha256')
        changed = [k for k in record if k not in speed_only and old.get(k) != record[k]
                   and not (k in derived and k not in old)]
        if changed:
            raise SystemExit(f'{out} holds a run with different settings: {changed}')
        if args.fibers and not record['fiber_namespace']:
            raise SystemExit(f'{out}/run.json has no fiber_namespace for the fiber store export (or pass --no-fibers)')
    else:
        record['fiber_namespace'] = str(uuid.uuid4())  # the UUID namespace of the run's exported fibers
        run_json.write_text(json.dumps(record, indent=1))
    owned = Owned(args.tile)
    done, next_id = load_done(out, owned)
    for name in (out/'fibers').glob('*') if (out/'fibers').exists() else ():
        if name.name not in done:
            shutil.rmtree(name)  # fibers of an unfinished tile are rewritten
    tiles = [dict(name=f'z{int(z):05d}_y{int(y):05d}_x{int(x):05d}', lo=np.array([x, y, z]),
                  hi=np.minimum(np.array([x, y, z])+args.tile, hi))
             for z in np.arange(lo[2], hi[2], args.tile) for y in np.arange(lo[1], hi[1], args.tile)
             for x in np.arange(lo[0], hi[0], args.tile)]
    t_w = time.time()
    weights = tile_weights(args.presence, args.threshold, tiles, vol.shape)
    for t, w in zip(tiles, weights if weights is not None else [1]*len(tiles)):
        t['weight'] = w
    total_w = sum(t['weight'] for t in tiles)
    done_w = sum(t['weight'] for t in tiles if t['name'] in done)
    todo = [t for t in tiles if t['name'] not in done]
    print(f'box {lo.tolist()} -> {hi.tolist()}: {len(tiles)} tiles ({sum(t["weight"] > 0 for t in tiles)} with '
          f'presence; weights in {time.time()-t_w:.0f}s), {len(done)} done ({done_w/max(total_w, 1):.1%} of presence)',
          flush=True)
    provenance = dict(prefix=args.prefix, trace=dict(
        checkpoint=checkpoint_source(args.checkpoint, tracer.checkpoint_path), operating_policy=policy.to_dict(),
        run=str(out.resolve())))
    mm = spec.grid_scale*args.base_um/1000
    t_start, last = time.time(), time.time()
    kept_total = traced_total = fibers_total = 0
    if (out/'tiles.jsonl').exists():  # totals of the tiles already saved (the last line of each, for redone tiles)
        lines = {}
        for line in open(out/'tiles.jsonl'):
            line = json.loads(line)
            lines[line['tile']] = line
        for name, line in lines.items():
            if name in done:
                kept_total += line['kept_vox']
                traced_total += line.get('traced_vox', 0)
                fibers_total += line['fibers']
    w_run = w_bands = 0
    prefetch = TilePrefetch(args.presence, args, todo, args.tile_workers, args.tile_ahead) if args.tile_workers else None
    try:
        for z in sorted({float(t['lo'][2]) for t in todo}):  # one streaming tracer call per z row of tiles
            band = [t for t in todo if float(t['lo'][2]) == z]
            st = Streamer(tracer, vol, owned, lo, hi, args, out, spec, provenance, next_id, band)
            st.presence = presence
            st.prefetch = prefetch

            t_band = time.time()
            reported_time, reported_decisions = t_band, 0
            reported_stage = dict(getattr(tracer, 'stage_seconds', {}))

            def status(force=False):
                nonlocal last, kept_total, traced_total, w_run, fibers_total
                nonlocal reported_time, reported_decisions
                nonlocal reported_stage
                for line in st.finished_tiles:
                    kept_total += line['kept_vox']
                    fibers_total += line['fibers']
                    traced_total += line.get('traced_vox', 0)
                    w_run += line['weight']
                    if line['weight'] == 0:
                        continue
                    print(f"  tile {line['tile']}: {line['fibers']} fibers, {line['kept_vox']*mm:.1f} mm kept, "
                          f"{line['seconds']}s", flush=True)
                st.finished_tiles = []
                if force or time.time()-last >= args.status_every:
                    last = time.time()
                    interval_rate = (st.decisions-reported_decisions)/max(last-reported_time, 1e-9)
                    reported_time, reported_decisions = last, st.decisions
                    el = time.time()-t_start
                    started = w_bands+st.read_weight  # presence weight of tiles whose dots were started, this session
                    eta = el/max(started, 1)*(total_w-done_w-started)
                    print(f'[{time.strftime("%H:%M:%S")}] started {(done_w+started)/max(total_w, 1):.2%} of presence, '
                          f'saved {(done_w+w_run)/max(total_w, 1):.2%} | saved tiles: {fibers_total} fibers, '
                          f'{kept_total*mm/1000:.2f} m kept, {traced_total*mm/1000:.2f} m traced | '
                          f'{len(st.rows)} traces running, '
                          f'{len(st.open)} tiles open | {st.decisions/max(time.time()-t_band, 1e-9):.0f} decisions/s | '
                          f'interval {interval_rate:.0f} decisions/s | '
                          f'feed {st.timing["feed"]/max(time.time()-t_band, 1e-9):.0%} checks '
                          f'{st.timing["checks"]/max(time.time()-t_band, 1e-9):.0%} write '
                          f'{st.timing["write"]/max(time.time()-t_band, 1e-9):.0%} of time | '
                          f'GPU chunk loads {getattr(getattr(tracer, "chunk_cache", None), "loads", 0)/max(el, 1e-9):.0f}/s | '
                          f'elapsed {el/3600:.2f} h, ETA {eta/3600:.1f} h', flush=True)
                    stages = dict(getattr(tracer, 'stage_seconds', {}))
                    delta = {k: v-reported_stage.get(k, 0.) for k, v in stages.items()}
                    total = max(sum(delta.values()), 1e-9)
                    print('stage wall: '+ ' '.join(f'{k}={v:.2f}s({v/total:.0%})' for k, v in delta.items()), flush=True)
                    reported_stage = stages

            orig_before = st.before_step
            def before_step(idx, pos, frames):
                status()
                return orig_before(idx, pos, frames)
            while st.cursor < len(st.tiles) or any(t['pending'] for t in st.open):
                tracer.trace(np.zeros((0, 3)), np.zeros((0, 3)), families=[], feed=st.feed, before_step=before_step,
                             on_finished=st.on_finished)
            st._close_tiles()
            if st.open:
                raise RuntimeError(f'tiles left open after their traces ended: {[t["name"] for t in st.open]}')
            next_id = st.next_id
            status(force=True)
            w_bands += st.read_weight
            print(f'z row {int(z)} done in {(time.time()-t_band)/60:.1f} min', flush=True)
        if args.fibers:
            t_fibers = time.time()
            result = export_store(out, out/'fibers.vcobj', args.fibers_template)
            print(f'wrote {result["path"]}: {result["fibers"]} fibers, {result["points"]} points '
                  f'({time.time()-t_fibers:.0f}s)', flush=True)
    finally:
        tracer.close()
        if getattr(tracer, 'axis_pool', None) is not None:
            tracer.axis_pool.shutdown()
        if prefetch is not None:
            prefetch.close()


if __name__ == '__main__':
    main()
