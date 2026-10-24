"""Training/evaluation episodes: a canonical GT window plus the volume block the front moves in.

An episode is drawn by rejection sampling: patch (weight ``valid ** patch_weight_power``), orientation and
anchor; the seed rows must be valid, the supervised region mostly valid, the reachable region plus the crop radius
must fit within ``roi``, and every corner of the ``roi`` cube around it must fall in split cells of the requested
split. The block read is the reachable region plus the view margin. Split
cells are ``split_block``-voxel cubes assigned by a seeded hash, so overlapping patches share a split.

Live chains (``train.live_chains``, see ``train/live.py``): ``EpisodeStream`` with ``batch`` yields whole batches and first
serves continuation jobs from the trainer, each the next window and block of the chains of one finished rollout
(``EpisodeSampler.continuation``).
"""
from __future__ import annotations

import hashlib
import json
import math
import queue
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from vesuvius.neural_tracing.tracers.shared.geometry import arclength, interp_at
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import ChunkedArray
from vesuvius.neural_tracing.tracers.sheets.data.fibers import episode_fibers, fiber_pool
from vesuvius.neural_tracing.tracers.sheets.data.patches import GridSpec, PatchInfo, PatchStore


def split_of(cell, val_fraction: float, seed: int) -> str:
    digest = hashlib.blake2b(f'{seed}:{cell[0]},{cell[1]},{cell[2]}'.encode(), digest_size=8).digest()
    return 'val' if int.from_bytes(digest, 'little')/2**64 < val_fraction else 'train'


class EpisodeSampler:
    """Picklable; each process opens its own volume reader and patch maps."""

    def __init__(self, data, index: list[PatchInfo], split: str, steps: int, roi: int, pad: float,
                 region_split: bool = False, seed_split: bool = False):
        """``region_split``: split by the reachable GT region instead of the block, and skip the block fit
        (for callers that read their own volume, e.g. trace benchmarks). ``seed_split`` (with ``region_split``): only
        the seed rows must lie in split cells (for benchmarks that score just the part of a trace over such cells)."""
        self.data, self.split, self.steps, self.roi, self.pad = data, split, steps, roi, pad
        self.region_split, self.seed_split = region_split, seed_split
        self.spec = GridSpec.for_episode(data, steps)
        self.front_len = (data.front_points-1)*data.point_spacing
        # Drop patches too small for an episode: some orientation must fit the seed rows plus all rollout steps
        # along one grid axis and the whole front along the other (exactly when ``_anchor`` can succeed).
        need_adv = (data.history+steps)*data.step
        fits = lambda p: any((a-1)*p.cell >= need_adv and (f-1)*p.cell >= self.front_len
                             for a, f in ((p.rows, p.cols), (p.cols, p.rows)))
        self.index = [p for p in index if fits(p)]
        if not self.index:
            raise ValueError('No patch is large enough for the configured front and rollout')
        # Weight by size, times the fraction of the split cells the patch's bbox touches that are in this split.
        fractions = np.array([self._split_fraction(p.bbox) for p in self.index])
        keep = fractions > 0
        self.index = [p for p, k in zip(self.index, keep) if k]
        if not self.index:
            raise ValueError(f'No patch touches a {split} split cell')
        weights = np.array([p.valid for p in self.index], np.float64)**data.patch_weight_power*fractions[keep]
        if data.dir_weights:  # each directory's share, split among its patches by their own weights
            where = np.array([self._dir_of[Path(p.root)] for p in self.index])
            share = np.array(data.dir_weights, np.float64)
            totals = np.bincount(where, weights, len(share))
            weights = weights*np.divide(share, totals, out=np.zeros_like(share), where=totals > 0)[where]
        self.weights = weights/weights.sum()
        self.store = PatchStore()
        self.level = data.volume_level
        iu0, _ = self.spec.index(0., 0.)
        self.seed_rows = [int(round(iu0-j*data.step/self.spec.res)) for j in range(data.history, -1, -1)]
        self.point_cols = [int(round(self.spec.index(0., k*data.point_spacing)[1])) for k in range(data.front_points)]
        u_end = int(round(self.spec.index(steps*data.step, 0.)[0]))
        self.sup_rows = slice(self.seed_rows[-1]+1, u_end+1)
        self.sup_cols = slice(self.point_cols[0], self.point_cols[-1]+1)
        self.reach_rows = slice(self.seed_rows[0], u_end+1)
        self.hard = []  # failure spots to start training episodes at (data.hard_windows)
        if split == 'train' and data.hard_fraction:
            self.hard = json.loads(Path(data.hard_windows).read_text())['records']
        # Matched fiber stretches (data.fiber_runs) long enough for this sampler: as long as the front for
        # fiber-seeded episodes, as long as the whole GT window's advance for fiber-following ones. Training episodes
        # and benchmark seeds (region_split, trace_benchmark.py --fiber-seeds) use them; validation stays on GT rows.
        self.follow = bool(data.fiber_follow)
        self.fiber_runs = []
        if (split == 'train' or region_split) and (data.fiber_fraction or self.follow):
            need = self.spec.u_vals[-1]-self.spec.u_vals[0] if self.follow else self.front_len
            keys = {(info.root, info.name) for info in self.index}
            for line in Path(data.fiber_runs).read_text().splitlines():
                run = json.loads(line)
                if (run['root'], run['patch']) in keys and run['end']-run['start'] >= need:
                    self.fiber_runs.append(dict(fiber=run['fiber'], root=run['root'], patch=run['patch'],
                                                start=float(run['start']), xyz=np.asarray(run['xyz'], np.float64),
                                                rc=np.asarray(run['rc'], np.float64)))
            if self.follow and not self.fiber_runs:
                raise ValueError('No fiber run is long enough for a fiber-following episode')

    def _split_fraction(self, bbox) -> float:
        d = self.data
        lo = np.floor(np.asarray(bbox[:3])/d.split_block).astype(int)
        hi = np.floor(np.asarray(bbox[3:])/d.split_block).astype(int)
        cells = [(x, y, z) for x in range(lo[0], hi[0]+1) for y in range(lo[1], hi[1]+1) for z in range(lo[2], hi[2]+1)]
        return sum(split_of(c, d.val_fraction, d.split_seed) == self.split for c in cells)/len(cells)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('volumes', None)
        return state

    @property
    def _dirs(self) -> list:
        return self.data.patches_dir if isinstance(self.data.patches_dir, (list, tuple)) else [self.data.patches_dir]

    @cached_property
    def _dir_of(self) -> dict:
        """Patch directory (a patch's ``root``) -> its position in data.patches_dir."""
        return {Path(p): i for i, p in enumerate(self._dirs)}

    @cached_property
    def volumes(self) -> dict:
        """CT volume path -> reader, opened on first use."""
        return {}

    def volume(self, root: str) -> ChunkedArray:
        """Reader of the CT volume the patches in directory ``root`` are on (data.patch_volumes)."""
        path = self.data.volume
        if self.data.patch_volumes:
            path = self.data.patch_volumes[self._dir_of[Path(root)]] or path
        if path not in self.volumes:
            self.volumes[path] = ChunkedArray(f'{path}/{self.level}', self.data.worker_cache_mb << 20)
        return self.volumes[path]

    def _anchor(self, rng, info: PatchInfo, axis: int, sign_u: int, sign_v: int, check: bool = False):
        """Random anchor (patch grid row/col of ``u = 0, v = 0``), or None if the orientation cannot fit; with
        ``check`` only whether it fits (the fit does not depend on the signs)."""
        d = self.data
        adv_len, front_len = (info.rows, info.cols) if axis == 0 else (info.cols, info.rows)
        back, fwd = d.history*d.step/info.cell, self.steps*d.step/info.cell
        span = self.front_len/info.cell
        lo_a, hi_a = (back, adv_len-1-fwd) if sign_u > 0 else (fwd, adv_len-1-back)
        lo_f, hi_f = (0., front_len-1-span) if sign_v > 0 else (span, front_len-1.)
        if hi_a < lo_a or hi_f < lo_f:
            return None
        if check:
            return True
        return rng.uniform(lo_a, hi_a), rng.uniform(lo_f, hi_f)

    def window(self, rng: np.random.Generator) -> dict | None:
        """GT window, validity, level-0 block origin (xyz) and provenance of one episode, without the block.

        A drawn patch gets up to ``data.anchor_tries`` attempts (fitting orientations only) before the sampler moves
        to another patch, so small, irregular patches are sampled at their weight instead of mostly rejected.
        """
        d = self.data
        if self.hard and rng.random() < d.hard_fraction:
            for _ in range(d.max_tries):
                episode = self._hard_attempt(rng)
                if episode is not None:
                    return episode
        if self.follow:
            for _ in range(d.max_tries):
                episode = self._follow_attempt(rng)
                if episode is not None:
                    return episode
            return None
        if self.fiber_runs and rng.random() < d.fiber_fraction:
            for _ in range(d.max_tries):
                episode = self._fiber_attempt(rng)
                if episode is not None:
                    return episode
        for _ in range(d.max_tries):
            info = self.index[int(rng.choice(len(self.index), p=self.weights))]
            for _ in range(d.anchor_tries):
                episode = self._attempt(rng, info)
                if episode is not None:
                    return episode
        return None

    def _attempt(self, rng: np.random.Generator, info: PatchInfo) -> dict | None:
        axes = [a for a in (0, 1) if self._anchor(None, info, a, 1, 1, check=True)]
        if not axes:
            return None
        axis = int(axes[rng.integers(len(axes))])
        sign_u, sign_v = int(rng.choice((-1, 1))), int(rng.choice((-1, 1)))
        return self._episode(info, axis, sign_u, sign_v, self._anchor(rng, info, axis, sign_u, sign_v))

    @cached_property
    def _by_dir(self) -> dict:
        """(position of the patch directory in data.patches_dir, patch name) -> PatchInfo, for hard spots."""
        roots = self._dir_of
        return {(roots[Path(info.root)], info.name): info for info in self.index if Path(info.root) in roots}

    def _hard_attempt(self, rng: np.random.Generator) -> dict | None:
        """An episode at a random recorded hard spot, jittered, or None if it does not fit there."""
        d, spot = self.data, self.hard[int(rng.integers(len(self.hard)))]
        info = self._by_dir.get((spot['dir'], spot['patch']))
        if info is None:
            return None
        adv = spot['anchor'][0]+spot['sign_u']*(spot['shift']+rng.uniform(-2, 2)*d.step)/info.cell
        front = spot['anchor'][1]+rng.uniform(-.5, .5)*d.point_spacing/info.cell
        episode = self._episode(info, spot['axis'], spot['sign_u'], spot['sign_v'], (adv, front))
        if episode is not None:
            episode['meta']['hard'] = True
        return episode

    def _episode(self, info: PatchInfo, axis: int, sign_u: int, sign_v: int, anchor) -> dict | None:
        """The episode window at ``anchor``, or None if it fails the validity, size or split checks."""
        gt, valid = self.store.window(info, self.spec, axis, sign_u, sign_v, *anchor)
        return self._checked(info, gt, valid, dict(axis=axis, sign_u=sign_u, sign_v=sign_v,
                                                   anchor=[float(anchor[0]), float(anchor[1])]))

    def _fiber_attempt(self, rng: np.random.Generator) -> dict | None:
        """An episode seeded from a matched fiber stretch (``data.fiber_runs``), or None if it does not fit.

        A random ``front_len`` stretch of a random run gives the seed row (the fiber's points at the point spacing
        along it). The GT window is rotated so its front axis follows the stretch's course on the patch grid and its
        advance axis is perpendicular to it (either way); the rows behind the seed step back along the GT advance
        direction at each point, made perpendicular to the fiber. The episode gets them as ``seed``."""
        d = self.data
        run = self.fiber_runs[int(rng.integers(len(self.fiber_runs)))]
        info = self._by_key.get((run['root'], run['patch']))
        if info is None:
            return None
        arc = arclength(run['xyz'])
        if arc[-1] < self.front_len:
            return None
        at = rng.uniform(0., arc[-1]-self.front_len)+np.arange(d.front_points)*d.point_spacing
        pts = interp_at(run['xyz'], arc, at)
        rc = np.stack([np.interp(at, arc, run['rc'][:, k]) for k in range(2)], 1)
        v_rc = rc[-1]-rc[0]
        if np.linalg.norm(v_rc) < 1e-6:
            return None
        v_rc /= np.linalg.norm(v_rc)
        first = int(rng.choice((-1, 1)))
        for sign_u in (first, -first):  # advance to either side of the fiber; the patch may extend only one way
            u_rc = sign_u*np.array([-v_rc[1], v_rc[0]])
            gt, valid = self.store.window_rotated(info, self.spec, rc[0], u_rc, v_rc)
            episode = self._checked(info, gt, valid, dict(fiber=run['fiber'], sign_u=sign_u, origin_rc=rc[0].tolist(),
                                                          u_rc=u_rc.tolist(), v_rc=v_rc.tolist(), at=float(at[0])),
                                    d.fiber_min_valid)
            if episode is not None:
                break
        else:
            return None
        back = self.seed_rows[-1]-self.seed_rows[-2] if len(self.seed_rows) > 1 else int(round(d.step/self.spec.res))
        rows, cols = self.seed_rows[-1], np.asarray(self.point_cols)
        if not valid[rows-back, cols].all():
            return None
        advance = gt[rows, cols]-gt[rows-back, cols]
        tangent = np.gradient(pts, axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        advance -= (advance*tangent).sum(1, keepdims=True)*tangent
        advance /= np.maximum(np.linalg.norm(advance, axis=1, keepdims=True), 1e-9)
        lags = np.arange(d.history, -1, -1)*d.step  # oldest first, like tracing.seeding.seed_rows
        episode['seed'] = (pts[None]-lags[:, None, None]*advance[None]).astype(np.float32)
        return self._with_fibers(episode, run, None)

    def _follow_attempt(self, rng: np.random.Generator) -> dict | None:
        """A fiber-following episode (``data.fiber_follow``), or None if it does not fit: the front (3 points) stands
        across a matched fiber, its middle point on the fiber, and advances along it. The GT window is laid along
        the fiber's course on the patch grid: lattice row ``u`` is the fiber point ``u`` voxels of arc further on
        (either way along the fiber), and column ``v`` lies ``v - mid`` voxels from it across the fiber on the
        patch (``mid``: the middle point's ``v``), so the middle column is the fiber itself. The GT seed rows and the
        usual losses then keep the middle point on the fiber and the others on the sheet beside it."""
        d = self.data
        run = self.fiber_runs[int(rng.integers(len(self.fiber_runs)))]
        info = self._by_key.get((run['root'], run['patch']))
        if info is None:
            return None
        arc = arclength(run['xyz'])
        u_vals, v_vals = self.spec.u_vals, self.spec.v_vals
        span = u_vals[-1]-u_vals[0]
        if arc[-1] < span:
            return None
        forward = int(rng.choice((-1, 1)))
        a0 = rng.uniform(-u_vals[0], arc[-1]-u_vals[-1]) if forward > 0 else rng.uniform(u_vals[-1], arc[-1]+u_vals[0])
        at = a0+forward*u_vals
        rc = np.stack([np.interp(at, arc, run['rc'][:, k]) for k in range(2)], 1)  # fiber course (Gu, 2)
        smooth = max(1, int(round(d.step/self.spec.res)))  # tangent over about one step: rc steps by whole nodes
        tangent = np.gradient(np.stack([np.convolve(np.pad(rc[:, k], smooth, mode='edge'), np.ones(2*smooth+1)/(2*smooth+1),
                                                    'valid') for k in range(2)], 1), axis=0)
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        sign_v = int(rng.choice((-1, 1)))
        across = sign_v*np.stack([-tangent[:, 1], tangent[:, 0]], 1)
        mid = (d.front_points-1)*d.point_spacing/2
        offset = (v_vals[None, :, None]-mid)/info.cell*across[:, None, :]
        rows, cols = rc[:, None, 0]+offset[..., 0], rc[:, None, 1]+offset[..., 1]
        gt, valid = self.store._sample(info, rows, cols)
        episode = self._checked(info, gt, valid, dict(fiber=run['fiber'], follow=True, forward=forward, sign_v=sign_v,
                                                      at=float(a0)), d.fiber_min_valid)
        if episode is None:
            return None
        # the seed fiber as known up to a cut ahead of the start (run arc a0 is fiber arc run['start'] + a0)
        cut = 0. if rng.random() < d.fiber_cut_at_start else rng.uniform(0., self.spec.u_vals[-1])
        start = run['start']+a0
        return self._with_fibers(episode, run, lambda arc: forward*(arc-start) <= cut)

    def _with_fibers(self, episode: dict, run: dict, seed_keep) -> dict:
        """The episode with the fibers around it (``data.fiber_pool``): the run's fiber as instance 0 (points where
        ``seed_keep(arc)``, all if None), the others numbered from 1."""
        d = self.data
        if not d.fiber_pool:
            return episode
        margin = self.pad+d.point_spacing
        key = run['fiber'] if isinstance(run['fiber'], str) else f"afv:{run['fiber']}"
        episode['fibers'] = episode_fibers(fiber_pool(d.fiber_pool), episode['reach_lo']-margin,
                                           episode['reach_hi']+margin, key, seed_keep, d.fiber_max_instances,
                                           d.fiber_max_points)
        return episode

    def _checked(self, info: PatchInfo, gt, valid, meta: dict, min_valid: float | None = None) -> dict | None:
        """The episode for window ``gt``/``valid`` of ``info``, or None if it fails the validity (at least
        ``min_valid``, default data.min_valid_fraction, of the supervised region), size or split checks."""
        d = self.data
        if not valid[np.ix_(self.seed_rows, self.point_cols)].all():
            return None
        if valid[self.sup_rows, self.sup_cols].mean() < (d.min_valid_fraction if min_valid is None else min_valid):
            return None
        reach = gt[self.reach_rows, self.sup_cols][valid[self.reach_rows, self.sup_cols]]
        lo, hi = reach.min(0), reach.max(0)
        if not self.region_split and (hi-lo).max()+2*self.pad > self.roi:
            return None
        origin0 = np.floor((lo+hi)/2-self.roi/2)  # xyz
        if self.region_split and self.seed_split:
            rows = np.ix_(self.seed_rows, range(self.sup_cols.start, self.sup_cols.stop))
            corners = gt[rows][valid[rows]]
        else:
            corners = ([np.where(c, hi, lo) for c in np.ndindex(2, 2, 2)] if self.region_split else
                       [origin0+np.array(c)*self.roi for c in np.ndindex(2, 2, 2)])
        cells = {tuple(int(v) for v in np.floor(c/d.split_block)) for c in corners}
        if any(split_of(c, d.val_fraction, d.split_seed) != self.split for c in cells):
            return None
        return dict(gt=gt, valid=valid, origin0=origin0, reach_lo=lo, reach_hi=hi,
                    meta=dict(patch=info.name, root=info.root, **meta))

    def sample(self, rng: np.random.Generator) -> dict | None:
        w = self.window(rng)
        if w is None:
            return None
        seed = {k: w[k] for k in ('seed', 'fibers') if k in w}  # fiber-started episodes: seed rows, fibers around
        return dict(gt=w['gt'], valid=w['valid'], meta=w['meta'], **seed,
                    **self._read(w['meta']['root'], w['reach_lo']-self.pad, w['reach_hi']+self.pad))

    def _read(self, root: str, lo, hi) -> dict:
        """Block (1, D, H, W) and its level-voxel origin (zyx) covering level-0 ``xyz`` [lo, hi] of the volume of
        patch directory ``root``."""
        scale = 1 << self.level
        origin = np.floor((lo[::-1]+.5)/scale-.5).astype(np.int64)  # zyx, level voxels
        size = tuple(int(v) for v in np.ceil((hi-lo)[::-1]/scale).astype(np.int64)+2)
        return dict(block=self.volume(root).read(origin, size)[None], origin=origin)

    @cached_property
    def _by_key(self) -> dict:
        return {(info.root, info.name): info for info in self.index}

    def continuation(self, chain: dict, min_valid: float) -> dict | None:
        """The next episode of a live chain, or None if its patch has none: the window of the chain's episode
        (``chain['meta']``) moved ``chain['shift']`` voxels further along the advance (to where its front got),
        same front alignment, and the block covering that window's reach and the front's current extent
        (``front_lo``, ``front_hi``). Needs ``min_valid`` of the supervised region valid and the block in the split."""
        meta = chain['meta']
        info = self._by_key.get((meta['root'], meta['patch']))
        if info is None:
            return None
        anchor = (meta['anchor'][0]+meta['sign_u']*chain['shift']/info.cell, meta['anchor'][1])
        gt, valid = self.store.window(info, self.spec, meta['axis'], meta['sign_u'], meta['sign_v'], *anchor)
        if valid[self.sup_rows, self.sup_cols].mean() < min_valid:
            return None
        reach = gt[self.reach_rows, self.sup_cols][valid[self.reach_rows, self.sup_cols]]
        if not len(reach):
            return None
        lo = np.minimum(reach.min(0), chain['front_lo'])-self.pad
        hi = np.maximum(reach.max(0), chain['front_hi'])+self.pad
        if (hi-lo).max() > self.roi:
            return None
        cells = {tuple(int(v) for v in np.floor(np.where(c, hi, lo)/self.data.split_block)) for c in np.ndindex(2, 2, 2)}
        if any(split_of(c, self.data.val_fraction, self.data.split_seed) != self.split for c in cells):
            return None
        return dict(gt=gt, valid=valid, meta=dict(meta, anchor=[float(anchor[0]), float(anchor[1])]),
                    **self._read(meta['root'], lo, hi))


class EpisodeStream(torch.utils.data.IterableDataset):
    """Endless episodes; each loader worker draws from its own seeded generator.

    With ``batch``, yields collated batches instead (load with ``batch_size=None``), and before drawing a fresh
    batch serves pending live-chain jobs from ``jobs`` (a multiprocessing queue of ``dict(id, min_valid, chains)``,
    one per finished rollout): as many whole jobs as fit in a batch, the chains' next episodes, with ``chains`` the
    (job id, chain index) of each episode. Chains without a next episode are left out; the trainer ages them out.
    """

    def __init__(self, sampler: EpisodeSampler, seed: int, batch: int = 0, jobs=None):
        self.sampler, self.seed, self.batch, self.jobs = sampler, seed, batch, jobs

    def _fresh(self, rng):
        return fresh_episode(self.sampler, rng)

    def _jobs(self):
        if self.jobs is None:
            return None
        jobs, total = [], 0
        while total < self.batch:
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                break
            if jobs and total+len(job['chains']) > self.batch:
                try:
                    self.jobs.put_nowait(job)  # for the next batch
                except queue.Full:
                    pass
                break
            jobs.append(job)
            total += len(job['chains'])
        served = [((job['id'], i), e) for job in jobs for i, chain in enumerate(job['chains'])
                  if (e := self.sampler.continuation(chain, job['min_valid'])) is not None]
        if not served:
            return None
        return dict(collate([e for _, e in served]), chains=[key for key, _ in served])

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, 0 if info is None else info.id])
        while True:
            if not self.batch:
                yield self._fresh(rng)
                continue
            batch = self._jobs()
            yield batch if batch is not None else collate([self._fresh(rng) for _ in range(self.batch)])


def fresh_episode(sampler: EpisodeSampler, rng) -> dict:
    for _ in range(21):
        episode = sampler.sample(rng)
        if episode is not None:
            return episode
    raise RuntimeError('Episode sampling keeps failing; relax data constraints or enlarge data.roi')


class MixedFrontStream(torch.utils.data.IterableDataset):
    """Endless collated batches (load with ``batch_size=None``) of one kind each, drawn per batch from ``samplers``
    (kind -> sampler, e.g. point count N -> episodes with an N-point front) with probabilities ``shares`` (default
    uniform), ``sizes[kind]`` episodes each; the batch records the kind as ``front_points``
    (``data.front_points_range``)."""

    def __init__(self, samplers: dict, sizes: dict, seed: int, shares: dict | None = None):
        self.samplers, self.sizes, self.seed = samplers, sizes, seed
        self.shares = shares

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, 0 if info is None else info.id])
        kinds = list(self.samplers)
        p = None if self.shares is None else np.array([self.shares[k] for k in kinds])/sum(self.shares.values())
        while True:
            kind = kinds[int(rng.choice(len(kinds), p=p))]
            yield dict(collate([fresh_episode(self.samplers[kind], rng) for _ in range(self.sizes[kind])]),
                       front_points=kind)


class NeighborStream(torch.utils.data.IterableDataset):
    """Endless collated batches (load with ``batch_size=None``) for neighbour continuations
    (``data.neighbor_fraction``): ``plain`` episodes from ``plain_sampler``, then ``wide`` episodes from
    ``wide_sampler``, whose window spans two fronts sharing the middle point (``2 * front_points - 1`` points).

    A wide episode's ``gt``/``valid`` are the half the model traces (so every episode of the batch has the same GT
    window); the whole window is kept as ``wide_gt``/``wide_valid`` (for the shared seed rows and the neighbour
    half). ``neighbor_side`` (one per batch, alternating): +1 the neighbour is the second half (the episode's last
    point is the shared one), -1 the first half (its first point is). ``offset``: GT-grid columns between the two
    halves' first points."""

    def __init__(self, plain_sampler: EpisodeSampler, wide_sampler: EpisodeSampler, plain: int, wide: int, offset: int,
                 seed: int):
        self.plain_sampler, self.wide_sampler, self.plain, self.wide = plain_sampler, wide_sampler, plain, wide
        self.offset, self.seed = offset, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, 0 if info is None else info.id])
        width = self.plain_sampler.spec.Gv
        side = 1
        while True:
            side = -side
            episodes = [fresh_episode(self.plain_sampler, rng) for _ in range(self.plain)]
            wide = [fresh_episode(self.wide_sampler, rng) for _ in range(self.wide)]
            start = 0 if side > 0 else self.offset  # the traced half's first column
            halves = [dict(e, gt=e['gt'][:, start:start+width], valid=e['valid'][:, start:start+width]) for e in wide]
            batch = collate(episodes+halves)
            batch.update(wide_gt=torch.from_numpy(np.stack([e['gt'] for e in wide])),
                         wide_valid=torch.from_numpy(np.stack([e['valid'] for e in wide])),
                         neighbor_side=side)
            yield batch


NEIGHBOR_KEYS = ('wide_gt', 'wide_valid')


def keep_batch(batch):
    """``collate_fn`` for streams that yield whole batches."""
    return batch


def pack_blocks(blocks: list[np.ndarray]) -> dict:
    """Tight blocks (channels, D, H, W) of different sizes, unpadded, as one flat array (``data``) plus their
    shapes (B, 4): padding them to a common size here would ship (and pin and copy) the padding too. ``unpack_blocks``
    pads on the device."""
    return dict(data=torch.from_numpy(np.concatenate([b.ravel() for b in blocks])),
                shapes=torch.tensor([b.shape for b in blocks], dtype=torch.int64))


def unpack_blocks(packed: dict, device, dtype=torch.float32, pad: bool = True):
    """(B, channels, D, H, W) ``dtype`` on ``device`` from ``pack_blocks``: blocks zero-padded at the far end (each keeps
    its own origin, so sampling coordinates are unchanged; padding reads as outside the volume). ``pad=False``: a list
    of the B blocks at their own sizes instead (``data.unpadded_blocks``; ``shared.crop_sampling.gpu.sample_block`` reads either)."""
    shapes = packed['shapes'].tolist()
    data = packed['data'].to(device, non_blocking=True)
    if not pad:
        starts = np.cumsum([0]+[int(np.prod(s)) for s in shapes])
        return [data[a:b].view(*s).to(dtype) for a, b, s in zip(starts, starts[1:], shapes)]
    out = torch.zeros((len(shapes), *np.max(shapes, 0).tolist()), dtype=dtype, device=device)
    start = 0
    for i, (c, d, h, w) in enumerate(shapes):
        out[i, :c, :d, :h, :w] = data[start:start+c*d*h*w].view(c, d, h, w)
        start += c*d*h*w
    return out


def collate(episodes: list[dict]) -> dict:
    return dict(gt=torch.from_numpy(np.stack([e['gt'] for e in episodes])),
                valid=torch.from_numpy(np.stack([e['valid'] for e in episodes])),
                block=pack_blocks([e['block'] for e in episodes]),  # CT, (B, 1, D, H, W) after unpack_blocks
                origin=torch.from_numpy(np.stack([e['origin'] for e in episodes])),
                meta=[e['meta'] for e in episodes], **_collate_seeds(episodes), **collate_fibers(episodes))


def collate_fibers(episodes: list[dict]) -> dict:
    """Fibers of the episodes that have them (``episode_fibers``), padded to the batch's most points: ``fiber_points``
    and ``fiber_tangent`` (B, P, 3), ``fiber_family`` and ``fiber_instance`` (B, P; instance -1 marks padding), and
    ``fiber_codes`` (B, I, code_dim): a random unit code per episode instance, drawn here (seeded by the episode's
    fibers), the same for all points of a fiber and different between fibers."""
    fibers = [e.get('fibers') for e in episodes]
    if not any(f is not None for f in fibers):
        return {}
    P = max(1, max(len(f['points']) for f in fibers if f is not None))
    I = max(1, max(int(f['instance'].max())+1 if f is not None and len(f['instance']) else 1 for f in fibers))
    B = len(episodes)
    out = dict(fiber_points=np.zeros((B, P, 3), np.float32), fiber_tangent=np.zeros((B, P, 3), np.float32),
               fiber_family=np.zeros((B, P), np.int64), fiber_instance=np.full((B, P), -1, np.int64),
               fiber_codes=np.zeros((B, I, FIBER_CODE_DIM), np.float32))
    for b, f in enumerate(fibers):
        if f is None or not len(f['points']):
            continue
        n = len(f['points'])
        out['fiber_points'][b, :n], out['fiber_tangent'][b, :n] = f['points'], f['tangent']
        out['fiber_family'][b, :n], out['fiber_instance'][b, :n] = f['family'], f['instance']
        rng = np.random.default_rng(int(abs(f['points'][0]).sum()*1000) % (1 << 32))
        codes = rng.standard_normal((I, FIBER_CODE_DIM))
        out['fiber_codes'][b] = codes/np.linalg.norm(codes, axis=1, keepdims=True)
    return {k: torch.from_numpy(v) for k, v in out.items()}


FIBER_CODE_DIM = 8  # model.fiber_code_dim of the collated instance codes
FIBER_KEYS = ('fiber_points', 'fiber_tangent', 'fiber_family', 'fiber_instance', 'fiber_codes')


def _collate_seeds(episodes: list[dict]) -> dict:
    """``seed`` (B, history+1, N, 3) and ``seed_mask`` (B,) when any episode brings its own seed rows (fiber
    seeds); the others are zero-filled and use the GT seed rows."""
    seeded = [e for e in episodes if 'seed' in e]
    if not seeded:
        return {}
    zero = np.zeros_like(seeded[0]['seed'])
    return dict(seed=torch.from_numpy(np.stack([e.get('seed', zero) for e in episodes])),
                seed_mask=torch.tensor(['seed' in e for e in episodes]))


def fixed_episodes(sampler: EpisodeSampler, count: int, seed: int) -> list[dict]:
    """A deterministic episode list (same seed, config and data -> same episodes)."""
    rng = np.random.default_rng([seed, 1 << 20])
    out, misses = [], 0
    while len(out) < count:
        episode = sampler.sample(rng)
        if episode is None:
            misses += 1
            if misses > 20+count:
                raise RuntimeError(f'Could not draw {count} {sampler.split} episodes')
            continue
        out.append(episode)
    return out


def batches(episodes: list[dict], size: int):
    for i in range(0, len(episodes), size):
        yield collate(episodes[i:i+size])


def episode_pad(model, train) -> float:
    """Block margin beyond the reachable GT region: view radius (crops) plus room for drift before a reset."""
    from vesuvius.neural_tracing.tracers.sheets.sampling import view_radius
    return math.ceil(view_radius(model)+train.reset_distance+4)
