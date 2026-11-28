"""Stitch the splines of a zone processed block by block into long fibers.

Every block's splines enter one catalogue, which suppresses duplicates and
joins splines that overlap across neighbouring blocks. Each chain can then be
grown from both ends by the reciprocal Expander: a gap join needs the learned
geometry score, a clear margin over the alternatives, the same choice from
the other side, no obstruction and CT at the join.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .candidates import bridge, collect
from .catalog import Catalog, point_count
from .expander import Expander
from .overlap import MIN_OVERLAP

RADIUS = 40
MIN_SCORE = .9
MIN_MARGIN = .15
MIN_LENGTH = 8.
FAMILIES = {"V": 0, "H": 1}
# A block's splines bend over their last few voxels where its border cuts a
# fiber, enough to fail the overlap and duplicate tests of the catalogue.
# Such an end is dropped where another block of the zone reads past the
# border and has its own copy of the fiber there: a spline of the same family
# within MATCH_RADIUS voxels of MATCH_SHARE of the dropped points. The end is
# kept when no other block has such a copy, and when dropping the cut ends of
# both copies would leave them less than KEEP_OVERLAP voxels of shared length,
# so the trim never removes geometry or a join that main keeps. CUT_END is half
# the CLI's 32-voxel block margin.
CUT_END = 16.
MATCH_RADIUS = 3.
MATCH_SHARE = .8
SHARED_RADIUS = 2.
KEEP_OVERLAP = MIN_OVERLAP + 4.


def _length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _kept_range(absolute, low, high, reach_low, reach_high, distance):
    """First and end index of the points kept when both cut ends of a spline read
    from ``low`` to ``high`` (absolute ZYX) are dropped. An end at the zone's reach is kept."""
    cut = np.zeros(len(absolute), dtype=bool)
    outer = np.zeros(len(absolute), dtype=bool)
    for axis in range(3):
        if low[axis] > reach_low[axis]:
            cut |= absolute[:, axis] - low[axis] < distance
        else:
            outer |= absolute[:, axis] - reach_low[axis] < 2
        if high[axis] < reach_high[axis]:
            cut |= high[axis] - 1 - absolute[:, axis] < distance
        else:
            outer |= reach_high[axis] - 1 - absolute[:, axis] < 2
    kept = np.flatnonzero(~cut)
    first = 0 if not cut[0] or outer[0] else (kept[0] if len(kept) else len(absolute))
    last = len(absolute) if not cut[-1] or outer[-1] else (kept[-1] + 1 if len(kept) else 0)
    return first, last


def _shared_length(points, tree):
    """Length of the polyline ``points`` whose segments lie within SHARED_RADIUS of ``tree``'s points."""
    if len(points) < 2:
        return 0.
    near = np.isfinite(tree.query(points, distance_upper_bound=SHARED_RADIUS)[0])
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(steps[near[:-1] & near[1:]].sum())


def trim_cut_ends(traces: Iterable[tuple[str, np.ndarray]], low_zyx: Sequence[int], high_zyx: Sequence[int],
                  reach_low_zyx: Sequence[int], reach_high_zyx: Sequence[int],
                  neighbours: Iterable[tuple[str, np.ndarray, Sequence[int], Sequence[int]]] = (),
                  distance: float = CUT_END) -> list[tuple[str, np.ndarray]]:
    """``traces`` of a block read from ``low_zyx`` to ``high_zyx``, in block-local ZYX points, with
    the ends a border of the block bends dropped where a neighbouring block has its own copy (see
    CUT_END). ``neighbours`` are the untrimmed splines of the zone's other blocks, each as family,
    absolute ZYX points and the low and high ZYX corners of the box its block reads. The zone's
    blocks read from ``reach_low_zyx`` to ``reach_high_zyx`` together. A trimmed spline left shorter
    than MIN_LENGTH is kept whole."""
    low, high, reach_low, reach_high = (np.asarray(v, dtype=float)
                                        for v in (low_zyx, high_zyx, reach_low_zyx, reach_high_zyx))
    others = {}
    for family, points, other_low, other_high in neighbours:
        points = np.asarray(points, dtype=np.float64)
        first, last = _kept_range(points, np.asarray(other_low, float), np.asarray(other_high, float),
                                  reach_low, reach_high, distance)
        others.setdefault(family, []).append((points, points[first:last]))
    trees = {}
    for family, rows in others.items():
        owner = np.concatenate([np.full(len(p), i) for i, (p, _) in enumerate(rows)])
        trees[family] = (cKDTree(np.concatenate([p for p, _ in rows])), owner)
    result = []
    for family, line in traces:
        points = np.asarray(line, dtype=np.float64)
        absolute = points + low
        first, last = _kept_range(absolute, low, high, reach_low, reach_high, distance)
        if family not in trees or last - first < 2 or _length(points[first:last]) < MIN_LENGTH:
            result.append((family, points))
            continue
        tree, owner = trees[family]
        kept_tree = cKDTree(absolute[first:last])
        keep = []
        for dropped in (absolute[:first], absolute[last:][::-1]):
            if len(dropped) == 0:
                keep.append(False)
                continue
            dist, index = tree.query(dropped, distance_upper_bound=MATCH_RADIUS)
            hit = np.isfinite(dist)
            copy = False
            for candidate in np.unique(owner[index[hit]]):
                if np.mean(hit & (owner[np.minimum(index, len(owner) - 1)] == candidate)) < MATCH_SHARE:
                    continue
                _, other_kept = others[family][candidate]
                if len(other_kept) >= 2 and min(_shared_length(other_kept, kept_tree),
                                                _shared_length(absolute[first:last], cKDTree(other_kept))) >= KEEP_OVERLAP:
                    copy = True
                    break
            keep.append(not copy)
        start = 0 if keep[0] else first
        stop = len(points) if keep[1] else last
        result.append((family, points[start:stop]))
    return result


class _Frozen:
    """Read access to a catalogue that no longer changes, decoding each chain and source once."""

    def __init__(self, catalog):
        self.catalog, self.db = catalog, catalog.db
        self.curves, self.sources = {}, {}

    def get_curve(self, chain_id, include_provenance=False):
        key = int(chain_id)
        if key not in self.curves:
            curve = self.catalog.get_curve(key, include_provenance=False)
            if curve is not None:
                # float32 holds the catalogue's coordinates exactly, at a quarter of the memory of lists.
                curve['points'] = np.asarray(curve['points'], dtype=np.float32)
            self.curves[key] = curve
        return self.curves[key]

    def source(self, sid):
        key = int(sid)
        if key not in self.sources:
            self.sources[key] = self.catalog.source(key)
        return self.sources[key]


class Stitcher:
    def __init__(self, root, check: Callable[[], None] = lambda: None):
        self.catalog = Catalog(root)
        self.check = check
        self.short = []

    def close(self):
        self.catalog.close()

    def add_block(self, name: str, origin_xyz: Sequence[int], size_xyz: Sequence[int],
                  traces: Iterable[tuple[str, np.ndarray]], core: tuple[np.ndarray, np.ndarray]):
        """``traces`` are ``(family, zyx points)`` local to the block.

        Splines too short for the catalogue are kept as they are when their
        middle lies in ``core``, the part of the zone this block owns.
        """
        self.check()
        origin = np.asarray(origin_xyz, dtype=float)
        curves = []
        for family, line in traces:
            local = np.asarray(line, dtype=np.float64)[:, ::-1]
            length = _length(local)
            curves.append((FAMILIES[family], local, point_count(length)))
            if point_count(length) < 5 and length >= MIN_LENGTH:
                middle = local[len(local)//2]+origin
                if np.all(middle >= core[0]) and np.all(middle < core[1]):
                    self.short.append(dict(family=FAMILIES[family], points=local+origin, gaps=[]))
        return self.catalog.ingest(name, origin_xyz, tuple(size_xyz)[::-1], curves)

    def chain_ids(self):
        return [int(r[0]) for r in self.catalog.db.execute('SELECT id FROM chains ORDER BY length DESC,id')]

    def chains(self):
        """The current catalogue chains: blocks joined where they overlap."""
        for cid in self.chain_ids():
            curve = self.catalog.get_curve(cid, include_provenance=False)
            points = np.asarray(curve['points'], float)
            yield dict(family=int(curve['family']), points=points, gaps=_gap_ranges(curve, points))

    def stitched(self):
        """Every fiber without extension: the catalogue chains and the short splines kept aside."""
        return list(self.chains()) + [dict(s, gaps=[]) for s in self.short]

    def expand(self, score, measure, progress: Callable[[int, int], None] | None = None):
        """Grow every chain, longest first; a chain joined into another is not grown again."""
        cat = _Frozen(self.catalog)
        identities = self.chain_ids()
        fetched, scored = {}, {}

        def fetch(cid, end):
            # The catalogue no longer changes, so a neighbourhood read serves every walk.
            key = (int(cid), int(end))
            if key not in fetched:
                fetched[key] = collect(cat, cid, end, RADIUS, self.check)
            return fetched[key]

        def score_once(proposals, row):
            # Proposals stay alive in ``fetched``, so their identities are stable keys.
            key = tuple(map(id, proposals))
            if key not in scored:
                scored[key] = score(proposals, row)
            return scored[key]

        completed, claimed = [], set()
        for position, identity in enumerate(identities):
            self.check()
            if progress:
                progress(position, len(identities))
            if identity in claimed:
                continue
            seed = cat.get_curve(identity, include_provenance=False)
            engine = Expander(seed, fetch, score_once, self.check, bridge, minimum=MIN_SCORE, margin=MIN_MARGIN,
                              measure=measure, max_seconds=math.inf, max_links=math.inf, max_points=math.inf)
            engine.run()
            result = engine.result()
            if set(result['curveIds']) & claimed:
                # Keep competing claims; never hide them to force a join.
                curve, accepted, stops = seed, {identity}, {'both': {'code': 'claimed'}}
            else:
                curve, accepted, stops = engine.ribbon_curve(result), set(result['curveIds']), result['stops']
            claimed.update(accepted)
            points = np.asarray(curve['points'], float)
            if _length(points) >= MIN_LENGTH:
                completed.append(dict(family=int(curve['family']), points=points, gaps=_gap_ranges(curve, points),
                                      stops=stops, chains=sorted(accepted)))
        if progress:
            progress(len(identities), len(identities))
        return completed + [dict(s, gaps=[], stops={}, chains=[]) for s in self.short]


def _gap_ranges(curve, points):
    """Point index ranges of the inferred gap bridges."""
    tree = cKDTree(points)
    ranges = []
    for gap in curve.get('gaps', []):
        if gap.get('kind') == 'gap' and len(gap.get('points', [])) >= 2:
            ranges.append(sorted(map(int, tree.query(np.asarray(gap['points'])[[0, -1]])[1])))
    return ranges
