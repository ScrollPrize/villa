"""Weak "tube" sites for the teacher (stages 2-4): integer voxels within 1 voxel of each training point and of each
segment between two same-class training points at most 5 voxels apart.

These sites are derived from labeled points. Sites within 1 voxel of an opposite-class training
point, sites claimed by both classes, and exact training-point positions are dropped. Held-out points keep their
buffers: no site within Chebyshev distance 256 of a held-out point or in a held-out 512^3 cell.
"""
import itertools

import numpy as np
from scipy.spatial import cKDTree


def tube(a, b, radius=1.):
    """Integer voxel centres within `radius` of the segment a-b (a == b gives a ball)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    lo = np.ceil(np.minimum(a, b) - radius).astype(int)
    hi = np.floor(np.maximum(a, b) + radius).astype(int)
    q = np.array(list(itertools.product(*(range(x, y + 1) for x, y in zip(lo, hi)))), dtype=np.float64)
    if not len(q):
        return np.empty((0, 3), np.int64)
    ab = b - a
    den = float(ab @ ab)
    t = np.clip((q - a) @ ab / den, 0, 1) if den else np.zeros(len(q))
    return q[np.linalg.norm(q - (a + t[:, None] * ab), axis=1) <= radius + 1e-10].astype(np.int64)


def build(q, y, train, heldout):
    """(site coordinates float64 [N, 3], labels [N], segments [E, 3] = (id_a, id_b, label), audit dict).

    Only training points create or filter sites; held-out points only enforce the buffers.
    """
    q = np.asarray(q, np.float64)
    y = np.asarray(y)
    train = np.asarray(train)
    heldout = np.asarray(heldout, np.int64)
    assert not set(train.tolist()) & set(heldout.tolist())
    pools = [set(), set()]
    segments = []
    removed_near_opposite = [0, 0]
    for label in (0, 1):
        ids = train[y[train] == label]
        opposite = cKDTree(q[train[y[train] != label]])
        pairs = sorted(cKDTree(q[ids]).query_pairs(5.))
        edges = [(int(ids[a]), int(ids[b])) for a, b in pairs]
        for a, b in edges:
            assert y[a] == y[b] == label and np.linalg.norm(q[a] - q[b]) <= 5. + 1e-9
            segments.append((a, b, label))
        for a, b in [(int(i), int(i)) for i in ids] + edges:
            sites = tube(q[a], q[b])
            good = opposite.query(sites)[0] > 1.0 + 1e-9
            removed_near_opposite[label] += int((~good).sum())
            pools[label].update(map(tuple, sites[good].tolist()))
    conflict = pools[0] & pools[1]
    for label in (0, 1):
        pools[label] -= conflict
    held_cells = {tuple(v) for v in np.floor(q[heldout] / 512).astype(np.int64)}
    held_tree = cKDTree(q[heldout].reshape(-1, 3))
    exact_tree = cKDTree(q[train])
    coordinates, labels, excluded = [], [], []
    for label in (0, 1):
        sites = np.asarray(sorted(pools[label]), np.float64).reshape(-1, 3)
        keep = held_tree.query(sites, p=np.inf)[0] > 256
        keep &= np.array([tuple(v) not in held_cells for v in np.floor(sites / 512).astype(int)])
        keep &= exact_tree.query(sites)[0] > 1e-6
        excluded.append(int((~keep).sum()))
        coordinates.extend(sites[keep])
        labels.extend([label] * int(keep.sum()))
    coordinates = np.asarray(coordinates, np.float64).reshape(-1, 3)
    labels = np.asarray(labels, np.int64)
    assert len(set(map(tuple, coordinates))) == len(coordinates) and set(labels) == {0, 1}
    audit = dict(radius_voxels=1., max_edge_length_voxels=5., train_endpoints=len(train), edges=len(segments),
                 edges_by_class=[sum(s[2] == v for s in segments) for v in (0, 1)], weak_voxels=len(labels),
                 weak_voxels_by_class=[int((labels == v).sum()) for v in (0, 1)],
                 opposite_inferred_conflicts_removed=len(conflict), near_opposite_click_samples_removed=removed_near_opposite,
                 heldout_buffer_or_exact_duplicates_removed=excluded)
    return coordinates, labels, np.asarray(segments, np.int64), audit
