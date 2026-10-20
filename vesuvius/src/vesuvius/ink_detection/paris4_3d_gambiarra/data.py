"""Labeled scribble samples, their CT parents and the labels visible in a plan view."""
import numpy as np
from scipy.spatial import cKDTree

from . import geometry as g
from . import tubes
from .ct import Parents
from .labels import Points
from .sampling import select_weak


class ClickData:
    """All VC3D scribble samples with a CT reader. weak=True also builds the weak tube sites over the
    training points (teacher stages 2-4)."""
    def __init__(self, store, weak=False, cache_gib=16., points_json=None):
        p = Points(points_json)
        self.q, self.y, self.parents, self.train, self.heldout = p.q, p.y, p.parents, p.train, p.heldout
        self.eval_parents = self.parents[self.heldout]
        self.tree = cKDTree(self.q[self.train])
        self.ct = Parents(store, lambda i: self.parents[i], cache_gib)
        self.weak = weak
        if weak:
            self.weak_q, self.weak_y, self.edges, self.weak_audit = tubes.build(self.q, self.y, self.train, self.heldout)
            self.weak_tree = cKDTree(self.weak_q)

    def transform(self, item):
        return g.construct(item['offset'], item['permutation'], item['flips'], item['angles'])

    def clicks(self, item, t):
        """Training points of the anchor's 320^3 context visible in the view: (ids, view coordinates float32)."""
        i = item['anchor']
        cand = self.train[self.tree.query_ball_point(self.parents[i] + 127.5, 159.5, p=np.inf)]
        ids, coords, _ = g.visible_ids(self.q[cand] - self.parents[i], cand, t)
        lo, hi = g.support_box(t, self.parents[i])
        assert i in ids and not g.intersects(lo, hi, self.eval_parents)
        return ids, coords

    def weak_sites(self, item, t):
        """At most 64 weak sites per class visible in the view: (site ids, view coordinates, labels)."""
        i = item['anchor']
        cand = np.asarray(self.weak_tree.query_ball_point(self.parents[i] + 127.5, 159.5, p=np.inf), np.int64)
        ids, coords, _ = g.visible_ids(self.weak_q[cand] - self.parents[i], cand, t)
        selected = select_weak(item, ids, self.weak_y)
        return ids[selected], coords[selected], self.weak_y[ids[selected]]

    def mask_clicks(self, item, t):
        """View coordinates of the training points inside the 128^3 view (the student's KD lattice avoids them)."""
        i = item['anchor']
        ids = self.train[self.tree.query_ball_point(self.parents[i] + 127.5, 160, p=np.inf)]
        q = g.points(self.q[ids] - self.parents[i], t)
        return q[np.all((q >= 0) & (q < 128), axis=1)].astype(np.float32)

    def limits(self, i):
        """(p1, p99) of point i's raw parent256."""
        return self.ct.limits(i)
