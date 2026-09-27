"""Coordinator-owned spatial coverage, independent of annotation and direction."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at


class PathCoverage:
    """Suppress paths whose aligned geometry is already covered by the bank.

    Equal-arclength samples make coverage length-weighted and invariant to
    reversal or vertex density. The index combines all annotations and shards.
    It identifies overlapping geometry, not biological fiber identities.
    """
    def __init__(self, *, distance=2., overlap=.8, sample_step=1., max_angle=25.):
        if (not np.isfinite([distance, overlap, sample_step, max_angle]).all()
                or distance <= 0 or not 0 < overlap <= 1 or sample_step <= 0 or not 0 < max_angle < 90):
            raise ValueError('Invalid path deduplication parameters')
        self.distance, self.overlap, self.sample_step = distance, overlap, sample_step
        self.alignment = np.cos(np.deg2rad(max_angle))
        self.points, self.tangents = np.empty((0,3)), np.empty((0,3))
        self.tree = None
        self.pending = []
        self.pending_tree = None

    def samples(self, path):
        path = np.asarray(path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.isfinite(path).all():
            raise ValueError('Invalid deduplication polyline')
        s = arclength(path)
        if s[-1] <= 0:
            raise ValueError('Deduplication polyline has zero length')
        edges = np.linspace(0., s[-1], int(np.ceil(s[-1]/self.sample_step))+1)
        points = interp_at(path, s, (edges[:-1]+edges[1:])/2)
        tangents = np.diff(interp_at(path, s, edges),axis=0)
        tangents /= np.maximum(np.linalg.norm(tangents,axis=1,keepdims=True),1e-12)
        return points, tangents

    def add(self, path):
        self.pending.append(self.samples(path))
        self.pending_tree = None
        if len(self.pending) >= 256:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        self.points = np.concatenate([self.points, *(p for p,_ in self.pending)])
        self.tangents = np.concatenate([self.tangents, *(t for _,t in self.pending)])
        self.tree = cKDTree(self.points)
        self.pending, self.pending_tree = [], None

    def coverage(self, path):
        points, tangents = self.samples(path)
        covered = np.zeros(len(points),bool)
        sources = [(self.tree,self.tangents)] if self.tree is not None else []
        if self.pending:
            if self.pending_tree is None:
                self.pending_tree = cKDTree(np.concatenate([p for p,_ in self.pending]))
            sources.append((self.pending_tree,np.concatenate([t for _,t in self.pending])))
        for tree, directions in sources:
            remaining = np.flatnonzero(~covered)
            for i, ids in zip(remaining,tree.query_ball_point(points[remaining],self.distance)):
                if ids and np.any(np.abs(directions[ids] @ tangents[i]) >= self.alignment):
                    covered[i] = True
        return float(covered.mean())

    def duplicate(self, path):
        return self.coverage(path) >= self.overlap
