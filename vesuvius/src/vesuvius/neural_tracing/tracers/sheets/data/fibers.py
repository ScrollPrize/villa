"""Fiber instances around an episode for the model's fiber inputs (``data.fiber_pool``, built by
``scripts/build_fiber_pool.py``): a pool of fiber point samples, memory-mapped and indexed by ``CELL``-voxel cells."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

CELL = 128  # voxels per side of the pool's spatial cells


def cell_key(points: np.ndarray) -> np.ndarray:
    """int64 key of the ``CELL`` cube holding each level-0 point (N, 3) xyz (coordinates below 2**20 cells)."""
    c = np.floor(np.asarray(points)/CELL).astype(np.int64)+(1 << 19)
    return (c[..., 0] << 40) | (c[..., 1] << 20) | c[..., 2]


class FiberPool:
    """The pool's arrays (memory-mapped, so loader workers share them) and region queries."""

    def __init__(self, path: str | Path):
        path = Path(path)
        load = lambda name: np.load(path/f'{name}.npy', mmap_mode='r')
        self.points, self.tangent, self.arc = load('points'), load('tangent'), load('arc')
        self.instance, self.family = load('instance'), load('family')
        self.cell_keys, self.cell_starts = np.asarray(load('cell_keys')), np.asarray(load('cell_starts'))
        self.keys = json.loads((path/'keys.json').read_text())
        self.index = {key: i for i, key in enumerate(self.keys)}
        aliases = path/'aliases.json'  # fibers left out as duplicates -> the instance standing for them
        if aliases.is_file():
            self.index.update({a: self.index[k] for a, k in json.loads(aliases.read_text()).items() if k in self.index})

    def region(self, lo, hi) -> np.ndarray:
        """Indices of the pool points within the level-0 box [lo, hi] (xyz)."""
        c0 = np.floor(np.asarray(lo)/CELL).astype(np.int64)
        c1 = np.floor(np.asarray(hi)/CELL).astype(np.int64)
        grid = np.stack(np.meshgrid(*(np.arange(a, b+1) for a, b in zip(c0, c1)), indexing='ij'), -1).reshape(-1, 3)
        keys = cell_key(grid*CELL+CELL/2)
        at = np.searchsorted(self.cell_keys, keys)
        hit = (at < len(self.cell_keys)) & (self.cell_keys[np.minimum(at, len(self.cell_keys)-1)] == keys)
        idx = [np.arange(self.cell_starts[i], self.cell_starts[i+1]) for i in at[hit]]
        idx = np.concatenate(idx) if idx else np.zeros(0, np.int64)
        pts = self.points[idx]
        inside = np.all((pts >= np.asarray(lo)) & (pts <= np.asarray(hi)), 1)
        return idx[inside]


@lru_cache(maxsize=2)
def fiber_pool(path: str) -> FiberPool:
    """One pool per process (samplers hold only the path, so pickled samplers stay small)."""
    return FiberPool(path)


def episode_fibers(pool: FiberPool, lo, hi, seed_key: str | None, seed_keep=None, max_instances: int = 32,
                   max_points: int = 4096, spacing: float = 20.) -> dict:
    """The fibers in box [lo, hi] as arrays: ``points`` (P, 3), ``tangent`` (P, 3), ``family`` (P,) and ``instance``
    (P,), the episode's instance numbers: 0 the seed fiber (``seed_key``; points kept where ``seed_keep(arc)`` is
    true, e.g. those not past a cut ahead of the start), 1.. the others, at most ``max_instances`` of them (those
    nearest the seed fiber or the box centre); points every ``spacing`` voxels along each fiber (a multiple of the
    pool's sample spacing), at most ``max_points`` in all (evenly thinned)."""
    idx = pool.region(lo, hi)
    arc = np.asarray(pool.arc[idx])
    idx = idx[np.isclose(np.mod(arc+1e-3, spacing), 0., atol=2e-3)]  # every ``spacing`` voxels along each fiber
    inst = pool.instance[idx]
    seed = pool.index.get(seed_key, -1) if seed_key is not None else -1
    is_seed = inst == seed
    if seed_keep is not None:
        drop = is_seed & ~seed_keep(pool.arc[idx])
        idx, inst, is_seed = idx[~drop], inst[~drop], is_seed[~drop]
    pts = pool.points[idx]
    centre = pts[is_seed].mean(0) if is_seed.any() else (np.asarray(lo)+np.asarray(hi))/2
    others = np.unique(inst[~is_seed])
    if len(others) > max_instances:
        near = np.array([np.linalg.norm(pts[inst == o]-centre, axis=1).min() for o in others])
        others = others[np.argsort(near)[:max_instances]]
    number = {int(o): k+1 for k, o in enumerate(others)}
    keep = is_seed | np.isin(inst, others)
    idx, inst, is_seed = idx[keep], inst[keep], is_seed[keep]
    if len(idx) > max_points:
        pick = np.linspace(0, len(idx)-1, max_points).astype(np.int64)
        idx, inst, is_seed = idx[pick], inst[pick], is_seed[pick]
    episode_instance = np.array([0 if s else number[int(i)] for i, s in zip(inst, is_seed)], np.int64)
    return dict(points=np.asarray(pool.points[idx], np.float32), tangent=np.asarray(pool.tangent[idx], np.float32),
                family=np.asarray(pool.family[idx], np.int64), instance=episode_instance)
