"""Training plan streams: anchor decks, per-view augmentation plans, weak-site selection and KD lattices.

Every random draw comes from an explicit numpy Generator in a fixed order, so a run's plans (and therefore its
checkpoint resume) are reproducible. The draw order of each function is part of the recipe: changing it changes
every later plan.
"""
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree

from . import geometry as g

VIEWS = 16


class ClassDeck:
    """Anchors without replacement per class; each take() returns `per_class` no-ink then `per_class` ink ids.

    A class list is refilled with rng.permutation(pool) when empty; ids pop from its end. An id already chosen in
    this take() goes back to the front of the list and is skipped.
    """
    def __init__(self, ids, y, rng, per_class=4):
        self.rng, self.per_class = rng, per_class
        self.pool = [np.array(ids)[y[ids] == v] for v in (0, 1)]
        self.left = [[], []]

    def take(self):
        out = []
        for v in (0, 1):
            chosen = []
            while len(chosen) < self.per_class:
                if not self.left[v]:
                    self.left[v] = self.rng.permutation(self.pool[v]).tolist()
                a = self.left[v].pop()
                if a in chosen:
                    self.left[v].insert(0, a)
                    continue
                chosen.append(a)
            out += chosen
        assert len(set(out)) == 2 * self.per_class
        return out

    def refresh(self):
        """Fresh full permutations of both classes (no-ink first), as teacher stage 4 does at its start."""
        self.left = [self.rng.permutation(self.pool[v]).tolist() for v in (0, 1)]


class PoolDeck:
    """Unlabelled parents without replacement, reshuffled from rng when exhausted."""
    def __init__(self, pool, rng):
        self.pool, self.rng, self.left = list(pool), rng, []

    def take(self):
        if not self.left:
            self.left = self.rng.permutation(self.pool).tolist()
        return int(self.left.pop())


def _photo(contrast, gamma, blur=0., noise=0., noise_seed=0):
    return dict(contrast=contrast, gamma=gamma, blur=blur, noise=noise, noise_seed=noise_seed)


def draw_rotated(rng, deck, q, parents, eval_parents):
    """Teacher stage 1: 8 anchors x 2 views; cube symmetry, a rotation of up to 30 degrees per axis for 85% of the
    views, and contrast / gamma / blur / noise. The rotated crop may reach 32 voxels outside the parent (the
    320^3 context) but never a held-out parent."""
    anchors = np.repeat(deck.take(), 2).tolist()
    offsets = [g.draw_offset(rng) for _ in anchors]
    plans = []
    for anchor, offset in zip(anchors, offsets):
        perm = g.PERMS[int(rng.integers(6))]
        flips = tuple(bool(v) for v in rng.integers(0, 2, 3))
        continuous = bool(rng.random() < .85)
        for attempt in range(1, 4097):
            angles = rng.uniform(-30, 30, 3) if continuous else np.zeros(3)
            t = g.construct(offset, perm, flips, angles)
            point = g.points(q[anchor] - parents[anchor], t)
            lo, hi = g.support_box(t, parents[anchor])
            good = np.all(t['source_corners'] >= -32) & np.all(t['source_corners'] <= 287)
            good = good & np.all(point >= 3.5) & np.all(point <= 123.5)
            if good and not g.intersects(lo, hi, eval_parents):
                break
            if not continuous:
                raise ValueError('invalid unrotated crop')
        else:
            raise RuntimeError('no valid rotation in 4096 draws')
        # `a if rng.random() < p else b` draws the coin first, then the value only if the coin succeeds.
        photo = _photo(contrast=float(rng.uniform(.5, 1.8)) if rng.random() < .85 else 1.,
                       gamma=float(np.exp(rng.uniform(np.log(.5), np.log(1.8)))) if rng.random() < .85 else 1.,
                       blur=float(rng.uniform(.35, .8)) if rng.random() < .30 else 0.,
                       noise=float(rng.uniform(.005, .015)) if rng.random() < .20 else 0.,
                       noise_seed=int(rng.integers(0, 2 ** 62)))
        plans.append(dict(anchor=int(anchor), offset=offset.tolist(), permutation=list(perm), flips=list(flips),
                          angles=angles.tolist(), continuous=continuous, attempts=attempt, photo=photo))
    return dict(anchors=anchors, plans=plans)


def _cube_item(rng):
    perm = g.PERMS[int(rng.integers(6))]
    flips = tuple(bool(v) for v in rng.integers(0, 2, 3))
    contrast = float(rng.uniform(.65, 1.5))
    gamma = float(np.exp(rng.uniform(np.log(.7), np.log(1.5))))
    if rng.random() >= .8:
        contrast = 1.
    if rng.random() >= .8:
        gamma = 1.
    return perm, flips, contrast, gamma


def draw_cube(rng, deck, q, parents, eval_parents):
    """Teacher stages 2-4 and the students: 8 anchors x 2 views, cube symmetry and moderate contrast / gamma
    (each applied with probability .8). All 16 offsets are drawn before the per-view draws."""
    anchors = np.repeat(deck.take(), 2).tolist()
    offsets = [g.draw_offset(rng) for _ in anchors]
    plans = []
    for anchor, offset in zip(anchors, offsets):
        perm, flips, contrast, gamma = _cube_item(rng)
        t = g.construct(offset, perm, flips, [0, 0, 0])
        lo, hi = g.support_box(t, parents[anchor])
        assert g.valid(t, q[anchor] - parents[anchor]) and not g.intersects(lo, hi, eval_parents)
        plans.append(dict(anchor=int(anchor), offset=offset.tolist(), permutation=list(perm), flips=list(flips),
                          angles=[0, 0, 0], continuous=False, attempts=1, photo=_photo(contrast, gamma)))
    return dict(anchors=anchors, plans=plans)


def unlabelled_item(rng, index, origin):
    """One unlabelled view: here the offset is drawn first, then the view's other draws (as draw_cube's per view)."""
    offset = g.draw_offset(rng)
    perm, flips, contrast, gamma = _cube_item(rng)
    return dict(unlabelled=int(index), origin=[int(v) for v in origin], offset=offset.tolist(), permutation=list(perm),
                flips=list(flips), photo=_photo(contrast, gamma))


def draw_unlabelled(rng, deck, origins, views=VIEWS):
    """`origins[index]` is the parent256 origin of pool row `index`."""
    plans = []
    for _ in range(views):
        i = deck.take()
        plans.append(unlabelled_item(rng, i, origins[i]))
    return dict(plans=plans)


def plan_sha(plan):
    """sha256 of the plan's canonical JSON (the digest stored with the original cached teacher targets)."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True,
                                     default=lambda v: v.tolist() if hasattr(v, 'tolist') else repr(v)).encode()).hexdigest()


def weak_generator(item):
    """Per-view generator for subsampling weak (tube) sites, seeded by the plan item: independent of the plan stream."""
    key = json.dumps(item, sort_keys=True, separators=(',', ':')).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))


def select_weak(item, ids, weak_y, per_class=64):
    """Indices into `ids` of at most `per_class` weak sites per class, sorted."""
    generator = weak_generator(item)
    selected = []
    for label in (0, 1):
        indices = np.flatnonzero(weak_y[ids] == label)
        if len(indices) > per_class:
            indices = generator.choice(indices, per_class, replace=False)
        selected.extend(indices.tolist())
    return np.asarray(sorted(selected), np.int64)


def lattice(item, clicks, workers=16):
    """KD sites of one view: a 32^3 lattice (spacing 4 voxels) with a per-view phase in [0, 4)^3 seeded by the plan
    item, minus sites within 1 voxel of a training click. Returns (sites float32 [32768, 3], keep mask)."""
    seed = int.from_bytes(hashlib.sha256(json.dumps(item, sort_keys=True).encode()).digest()[:8], 'little')
    phase = np.random.default_rng(seed).uniform(0, 4, 3)
    cap = np.nextafter(np.float32(128), np.float32(-np.inf))
    axes = [np.minimum(np.arange(32, dtype=np.float32) * 4 + p, cap) for p in phase]
    q = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3).astype(np.float32)
    # keep = (nearest click farther than 1); a query bounded at 1.5 returns the same decision.
    keep = cKDTree(clicks).query(q, workers=workers, distance_upper_bound=1.5)[0] > 1 if len(clicks) else np.ones(len(q), bool)
    assert keep.any() and (q >= 0).all() and (q <= 127.99999).all()
    return q, keep
