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

RADIUS = 40
MIN_SCORE = .9
MIN_MARGIN = .15
MIN_LENGTH = 8.
FAMILIES = {"V": 0, "H": 1}


def _length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


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
            a, b = sorted(map(int, tree.query(np.asarray(gap['points'])[[0, -1]])[1]))
            if a != b:
                ranges.append([a, b])
    return ranges
