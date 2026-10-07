"""Every annotated fiber of one CT volume, for labelling identity losses as switches (``tube_scoring``).

All AFV splits and the Paris 4 manual fibers count when they share the evaluated source's CT and trace grid.
The trace's own fiber is excluded, and so are its duplicates: catalog fibers coincident with it along their whole
overlap (>= DUPLICATE_OVERLAP voxels within 4 voxels of it, median <= 1 and p90 <= 2 voxels). A neighbour that
only approaches closely somewhere is kept.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from ..shared.geometry import arclength, interp_at

SEARCH = 4.  # trace voxels; farther fibers are irrelevant to the switch label
CHUNK = 64
DUPLICATE_OVERLAP = 32.
AFV_CODE = 10**9  # fiber code: afv number * AFV_CODE + fiber id; manual fibers -(index+1)


def samples(points, step=.25):
    points = np.asarray(points, np.float64)
    arc = arclength(points)
    return interp_at(points, arc, np.arange(0., arc[-1]+1e-9, step)) if arc[-1] > 0 else points[:1]


class NeighborFibers:
    """Distances from trace points to the volume's other annotated fibers."""

    def __init__(self, afvs=(), manual=()):
        self.afvs = list(afvs)  # AFVFibers with split='all'
        self.manual = list(manual)
        pieces = [samples(f.points) for f in self.manual]
        self.manual_points = np.concatenate(pieces) if pieces else np.zeros((0, 3))
        self.manual_codes = np.concatenate([np.full(len(p), -(i+1)) for i, p in enumerate(pieces)]) if pieces else np.zeros(0, int)
        self.manual_tree = cKDTree(self.manual_points) if pieces else None
        self._duplicates = {}

    @classmethod
    def from_configs(cls, configs):
        """Catalog of the dataset-config source entries that share one CT and trace grid."""
        from ..data.afv import AFVFibers
        from ..data.data import load_fibers
        afvs, manual = [], []
        for config in configs:
            if config['kind'] == 'afv':
                afvs.append(AFVFibers(config['path'], float(config['grid_scale']), split='all'))
            else:
                manual.extend(load_fibers(config['fibers'], grid_scale=float(config.get('grid_scale', 8.))))
        return cls(afvs, manual)

    def code(self, name):
        """Fiber code of an annotation by its name (AFV catalog name or manual fiber file name)."""
        if not hasattr(self, '_codes'):
            self._codes = {f.name: -(i+1) for i, f in enumerate(self.manual)}
            for number, afv in enumerate(self.afvs):
                self._codes.update({row[1]: number*AFV_CODE+int(row[0]) for row in afv.catalog})
        return self._codes[name]

    def candidates(self, points):
        """Sample points and fiber codes of every catalog fiber within SEARCH of the bounding box of ``points``."""
        lo, hi = points.min(0)-SEARCH, points.max(0)+SEARCH
        pts, codes = [], []
        for number, afv in enumerate(self.afvs):
            for record in afv.nearby_blocks(np.stack([lo, hi]), -1, records=True):
                pts.append(record['samples'])
                codes.append(np.full(len(record['samples']), number*AFV_CODE+int(record['shard'].split(':')[1])))
        if self.manual_tree is not None:
            centre, radius = (lo+hi)/2, float(np.linalg.norm(hi-lo))/2
            ids = self.manual_tree.query_ball_point(centre, radius)
            if ids:
                pts.append(self.manual_points[ids])
                codes.append(self.manual_codes[ids])
        return (np.concatenate(pts), np.concatenate(codes)) if pts else (np.zeros((0, 3)), np.zeros(0, int))

    def distance(self, points, exclude):
        """Distance from each point to the nearest catalog fiber not in ``exclude`` (inf beyond SEARCH)."""
        points = np.asarray(points, np.float64).reshape(-1, 3)
        out = np.full(len(points), np.inf)
        for start in range(0, len(points), CHUNK):
            chunk = points[start:start+CHUNK]
            pts, codes = self.candidates(chunk)
            if not len(pts):
                continue
            keep = ~np.isin(codes, list(exclude))
            if keep.any():
                d, _ = cKDTree(pts[keep]).query(chunk, distance_upper_bound=SEARCH)
                out[start:start+CHUNK] = d
        return out

    def geometry(self, code):
        if code < 0:
            return np.asarray(self.manual[-code-1].points)
        afv = self.afvs[code//AFV_CODE]
        return np.asarray(afv.geometry(afv.id_to_index[code % AFV_CODE]).points)

    def duplicates(self, own, annotation):
        """Catalog fibers coincident with the annotation along their whole overlap (module docstring)."""
        if own not in self._duplicates:
            tree = cKDTree(annotation)
            found = set()
            probe = samples(annotation, 2.)
            for start in range(0, len(probe), CHUNK):
                pts, codes = self.candidates(probe[start:start+CHUNK])
                if len(pts):
                    d, _ = tree.query(pts, distance_upper_bound=1.5)
                    found.update(np.unique(codes[np.isfinite(d)]).tolist())
            found.discard(own)
            duplicate = set()
            for code in found:
                d, _ = tree.query(samples(self.geometry(code), 1.))
                overlap = d[d <= 4.]
                if len(overlap) >= DUPLICATE_OVERLAP and np.median(overlap) <= 1. and np.quantile(overlap, .9) <= 2.:
                    duplicate.add(code)
            self._duplicates[own] = duplicate
        return self._duplicates[own]

    def foreign(self, own, annotation):
        """``points -> distance`` to the nearest annotated fiber other than ``own`` and its duplicates."""
        exclude = {own} | self.duplicates(own, np.asarray(annotation, np.float64))
        return lambda points: self.distance(points, exclude)

