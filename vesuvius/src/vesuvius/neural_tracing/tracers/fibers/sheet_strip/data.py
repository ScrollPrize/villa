"""Sheet-strip training data: decisions on AFV fibers where they lie on a verified patch, their tube inputs and strip
targets.

A decision is the fiber tracer's own training state (data/samples.make_sample: a simulated trace of the fiber up to
its head, the crop frame from the frozen frame model, augmentation) with its head on a span of the fiber matched to a
patch (sheet_strip/build_index.py). On top of the fiber follower's observation it carries:

  * the tube curve: crop-local points of the fiber the model is shown, spaced half a sample apart. With probability
    ``full_tube_share`` the whole fiber through the crop ('full': the existing fiber, ahead of the head too; its path
    continuation is then not supervised), else the simulated trace up to the head ('observed'). The trainer renders
    the tube channel on the GPU (sheet_strip/model.render_tube). The full fiber is cut to its matched spans
    (+-``span_margin``): beyond them it is unverified and may have left the sheet;
  * with an umbilicus, the crop's orientation against the scroll (``crop_orientation``);
  * strip targets: the sheet height field over the strip grid (sheet_strip/targets.py) from the patches of every run
    of the fiber within reach of the crop (at most ``max_patches``, those with most samples near the head).

Units: the index is in patch voxels (the CT level of the patches, 9.6 um for Paris4); decisions in trace voxels,
``trace_per_patch`` of them per patch voxel.
"""
from __future__ import annotations

from collections.abc import Sequence
import json
import math
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.tracers.fibers.data.follow_dataset import FollowDataset
from vesuvius.neural_tracing.tracers.fibers.data.observations import IdentityObservationBuilder
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.build_index import SPLITS
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.model import strip_grid
from vesuvius.neural_tracing.tracers.fibers.sheet_strip.targets import strip_targets
from vesuvius.neural_tracing.tracers.shared.annotations.afv import AFVFibers
from vesuvius.neural_tracing.tracers.shared.geometry import dense_line
from vesuvius.neural_tracing.tracers.sheets.data.patches import PatchInfo, PatchStore


class StripIndex:
    """The run index (memory-mapped) of one split: its fibers ('units', AFV catalog indices) and their runs."""

    def __init__(self, path, split):
        path = Path(path)
        load = lambda name: np.load(path/f'{name}.npy', mmap_mode='r')
        self.meta = json.loads((path/'meta.json').read_text())
        self.patches = [PatchInfo(**dict(row, bbox=tuple(row['bbox']))) for row in json.loads((path/'patches.json').read_text())]
        keep = np.flatnonzero(np.asarray(load('runs_split')) == SPLITS[split])
        self.fiber = np.asarray(load('runs_fiber'))[keep]
        self.patch = np.asarray(load('runs_patch'))[keep]
        self.start = np.asarray(load('runs_start'))[keep].astype(np.float64)
        self.end = np.asarray(load('runs_end'))[keep].astype(np.float64)
        samples = np.asarray(load('run_samples'))
        self.sample_lo, self.sample_hi = samples[keep], samples[keep+1]
        self.rc, self.xyz = load('sample_rc'), load('sample_xyz')
        # Runs are sorted by fiber: each unit's runs are one slice.
        self.units, first = np.unique(self.fiber, return_index=True)
        self.unit_runs = (np.stack((first, np.r_[first[1:], len(self.fiber)]), 1)
                          if len(first) else np.empty((0, 2), np.int64))
        self.unit_spans = [merged_spans(self.start[a:b], self.end[a:b]) for a, b in self.unit_runs]
        self.unit_length = np.array([float((s[:, 1]-s[:, 0]).sum()) for s in self.unit_spans])

    def __len__(self):
        return len(self.units)

    def runs(self, unit):
        a, b = self.unit_runs[unit]
        return range(int(a), int(b))


def merged_spans(start, end):
    """Union (K, 2) of [start, end] spans."""
    order = np.argsort(start)
    out = []
    for a, b in zip(start[order], end[order]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.asarray(out, np.float64).reshape(-1, 2)


class RunFibers(Sequence):
    """The index's units as fibers (AFVFibers geometry in trace voxels), with their matched lengths as sampling
    weights (``lengths``, read by FollowDataset)."""

    def __init__(self, afv, index: StripIndex, grid_scale, length_power=1.):
        self.afv = AFVFibers(afv, grid_scale, split='all', cache_size=64)
        self.index = index
        self.lengths = index.unit_length**length_power

    def __len__(self):
        return len(self.index)

    def __getitem__(self, unit):
        return self.afv[int(self.index.units[unit])]


class StripDataset(FollowDataset):
    """FollowDataset (fresh decisions only) whose heads lie on the matched spans: a fiber in proportion to its
    matched length (``RunFibers.lengths``), then a point uniformly on its matched spans, either direction."""

    def __init__(self, fibers: RunFibers, *args, trace_per_patch, **kwargs):
        super().__init__(fibers, *args, **kwargs)
        self.trace_per_patch = trace_per_patch

    def fresh_location(self, rng, windows):
        unit = int(rng.choice(len(self.fibers), p=self.weights))
        spans = self.fibers.index.unit_spans[unit]
        lengths = spans[:, 1]-spans[:, 0]
        k = int(rng.choice(len(spans), p=lengths/lengths.sum()))
        length = self.fibers[unit].length
        original_t = float(np.clip(rng.uniform(*spans[k])*self.trace_per_patch, 0., length))
        reverse = bool(rng.integers(2))
        return unit, length-original_t if reverse else original_t, reverse, 0


def polyline_points(world, pos, frame, crop, spacing, margin, limit):
    """Crop-local points (P, 3) of a world polyline, densified to ``spacing`` and kept inside the crop grown by
    ``margin`` (at most ``limit``, evenly thinned)."""
    world = np.asarray(world, np.float64)
    reach = math.sqrt(((crop.width-1)*crop.spacing/2+margin)**2*2+(max(crop.behind, crop.depth-crop.behind)*crop.spacing+margin)**2)
    near = np.flatnonzero(np.linalg.norm(world-pos, axis=1) <= reach+2.)
    if not len(near):
        return np.zeros((0, 3), np.float32)
    world = world[max(0, near[0]-1):near[-1]+2]
    if len(world) > 1 and np.linalg.norm(np.diff(world, axis=0), axis=1).sum() > 1e-9:
        world = dense_line(world, spacing)
    local = (world-pos) @ frame
    half = (crop.width-1)*crop.spacing/2+margin
    inside = ((np.abs(local[:, :2]) <= half).all(1) & (local[:, 2] >= -crop.behind*crop.spacing-margin)
              & (local[:, 2] <= (crop.depth-1-crop.behind)*crop.spacing+margin))
    local = local[inside]
    if len(local) > limit:
        local = local[np.linspace(0, len(local)-1, limit).round().astype(np.int64)]
    return local.astype(np.float32)


def load_umbilicus(path):
    """Umbilicus control points (K, 3) x, y, z (patch voxels) sorted by z, from an umbilicus.json."""
    points = np.array([[p['x'], p['y'], p['z']] for p in json.loads(Path(path).read_text())['control_points']], np.float64)
    return points[np.argsort(points[:, 2])]


def crop_orientation(pos, frame, umbilicus, trace_per_patch):
    """Per crop (positions (B, 3) trace voxels, frames (B, 3, 3) columns u, v, f in world xyz): its axes against the
    scroll (B, 5): the outward horizontal component of u and of v (away from the umbilicus at the crop's height) and
    the z components of u, v and f."""
    pos, frame = np.asarray(pos, np.float64)/trace_per_patch, np.asarray(frame, np.float64)
    centre = np.stack([np.interp(pos[:, 2], umbilicus[:, 2], umbilicus[:, k]) for k in (0, 1)], 1)
    radial = pos[:, :2]-centre
    radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-9)
    outward = np.einsum('bi,bij->bj', radial, frame[:, :2, :2])  # (u, v) . outward
    return np.concatenate((outward, frame[:, 2, :]), 1).astype(np.float32)


class StripBuilder(IdentityObservationBuilder):
    """The fiber follower's observation (augmentation included) plus the tube curve and strip targets (module
    docstring)."""

    def __init__(self, cfg, fibers: RunFibers, sampling, *, augment, trace_per_patch, full_tube_share=.5,
                 max_patches=4, reach=110., fold_tolerance=2., max_slope_deg=70., tube_points=2048, span_margin=8.,
                 umbilicus=None):
        super().__init__(cfg, fibers, sampling, augment=augment)
        self.index = fibers.index
        self.trace_per_patch, self.full_tube_share, self.max_patches = trace_per_patch, full_tube_share, max_patches
        self.reach, self.fold_tolerance, self.max_slope_deg = reach, fold_tolerance, max_slope_deg
        self.tube_points, self.span_margin = tube_points, span_margin
        f, v = strip_grid(cfg)
        self.f, self.v = f.numpy().astype(np.float64), v.numpy().astype(np.float64)
        self.patch_store = PatchStore(max_open=64)
        self.umbilicus = load_umbilicus(umbilicus) if umbilicus else None

    def prepare(self, item, fiber, rng):
        item = super().prepare(item, fiber, rng)
        item['tube_full'] = bool(rng.random() < self.full_tube_share)
        return item

    def windows(self, item):
        """Crop-local node windows (h, w, 3) and validity of the patches of the item's fiber within reach."""
        unit, t, reverse = item['fiber_ref']
        fiber = self.fibers[unit]
        arc = (fiber.length-t if reverse else t)/self.trace_per_patch  # patch voxels along the fiber
        head = np.asarray(item['pos'])/self.trace_per_patch
        crop = self.cfg.fine
        along = max(crop.behind, crop.depth-crop.behind)*crop.spacing/self.trace_per_patch+16.
        groups = {}
        index = self.index
        for r in index.runs(unit):
            if index.end[r] < arc-along or index.start[r] > arc+along:
                continue
            lo, hi = int(index.sample_lo[r]), int(index.sample_hi[r])
            xyz = np.asarray(index.xyz[lo:hi], np.float64)
            near = np.linalg.norm(xyz-head, axis=1) <= self.reach
            if near.any():
                groups.setdefault(int(index.patch[r]), []).append(np.asarray(index.rc[lo:hi])[near])
        chosen = sorted(groups, key=lambda p: -sum(len(rc) for rc in groups[p]))[:self.max_patches]
        out = []
        for p in chosen:
            info = index.patches[p]
            rc = np.concatenate(groups[p])
            margin = int(math.ceil((self.cfg.strip_half_width/self.trace_per_patch+16.)/info.cell))+1
            r0, c0 = np.maximum(rc.min(0)-margin, 0)
            r1, c1 = np.minimum(rc.max(0)+margin+1, (info.rows, info.cols))
            arrays = self.patch_store.arrays(info)
            xyz = np.stack([np.asarray(a[r0:r1, c0:c1], np.float32) for a in arrays], -1)
            valid = np.all(xyz != -1, -1) & np.all(np.isfinite(xyz), -1)
            local = (xyz.astype(np.float64)*self.trace_per_patch-item['pos']) @ np.asarray(item['frame'])
            out.append((local, valid))
        return out, len(chosen)

    def tube_curve(self, item):
        """The tube's crop-local points: the observed trace, or the fiber over its matched spans (+-``span_margin``
        patch voxels; elsewhere the fiber is unverified and may have left the sheet the targets describe)."""
        unit = item['fiber_ref'][0]
        crop = self.cfg.fine
        points = lambda world: polyline_points(world, np.asarray(item['pos']), np.asarray(item['frame']), crop,
                                               crop.spacing/2, 3*self.cfg.tube_sigma*crop.spacing, self.tube_points)
        if not item['tube_full']:
            return points(item['observed_path'])
        fiber = self.fibers[unit]
        arc = np.asarray(fiber.s)/self.trace_per_patch
        pieces = [points(fiber.points[(arc >= a-self.span_margin) & (arc <= b+self.span_margin)])
                  for a, b in self.index.unit_spans[unit]]
        local = np.concatenate([p for p in pieces if len(p)] or [np.zeros((0, 3), np.float32)])
        if len(local) > self.tube_points:
            local = local[np.linspace(0, len(local)-1, self.tube_points).round().astype(np.int64)]
        return local

    def __call__(self, items, vol):
        batch = super().__call__(items, vol)
        crop = self.cfg.fine
        height = (crop.width-1)*crop.spacing/2
        points = np.zeros((len(items), self.tube_points, 3), np.float32)
        valid = np.zeros((len(items), self.tube_points), bool)
        targets = np.zeros((len(items), len(self.f), len(self.v)), np.float32)
        masks = np.zeros((len(items), len(self.f), len(self.v)), bool)
        patches = np.zeros(len(items), np.int64)
        for row, item in enumerate(items):
            curve = self.tube_curve(item)
            points[row, :len(curve)], valid[row, :len(curve)] = curve, True
            windows, patches[row] = self.windows(item)
            targets[row], masks[row] = strip_targets(windows, self.f, self.v, fold_tolerance=self.fold_tolerance,
                                                     max_slope_deg=self.max_slope_deg, max_height=height)
        batch.update(tube_points=torch.from_numpy(points), tube_valid=torch.from_numpy(valid),
                     strip_target=torch.from_numpy(targets), strip_mask=torch.from_numpy(masks),
                     strip_patches=torch.from_numpy(patches),
                     tube_full=torch.tensor([bool(item['tube_full']) for item in items]))
        if self.umbilicus is not None:
            batch['crop_orientation'] = torch.from_numpy(crop_orientation(
                np.stack([i['pos'] for i in items]), np.stack([i['frame'] for i in items]), self.umbilicus,
                self.trace_per_patch))
        return batch
