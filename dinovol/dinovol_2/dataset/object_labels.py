"""Pseudo-object labels (papyrus sheets and fibers) for thin source slabs.

Sheets come from a binary surface prediction, fibers from a fiber presence field with
an encoded fiber direction. Both are split into objects conservatively: an object may be
one sheet or fiber cut into several pieces, but never two sheets or two crossing fibers
merged into one. Pairs of objects are marked as different only when that is provable
(stacked sheet layers, fibers at an angle); every other pair is left unknown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

SIX_CONNECTED = ndi.generate_binary_structure(3, 1)
SHEET_KIND = 1
FIBER_KIND = 2


def _runs_per_column(column_key, depth, gap):
    order = np.lexsort((depth, column_key))
    key, depth = column_key[order], depth[order]
    new_column = np.r_[True, key[1:] != key[:-1]]
    new_run = new_column | np.r_[False, np.diff(depth) > gap]
    return np.add.reduceat(new_run.astype(np.int32), np.flatnonzero(new_column))


def _column_key(u, v, bin_size):
    return np.floor(u / bin_size).astype(np.int64) * 1_000_003 + np.floor(v / bin_size).astype(np.int64)


def multi_layer_fraction(coords: np.ndarray, bin_size: float = 3.0, gap: float = 3.0) -> float:
    """Fraction of normal-direction lines that cross a voxel set more than once.

    A single sheet is crossed once per column; sheets stacked and joined by a bridge are
    crossed several times. The normal is taken two ways and the worse result is kept:
    the least-variance PCA axis, and the least-variance axis within the y-x plane. The
    second guards thin z slabs, where the slab thickness itself can be the least-variance
    axis and would hide stacked vertical sheets.
    """
    centered = coords - coords.mean(0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    uvw = centered @ vectors[:, ::-1]
    pca = _runs_per_column(_column_key(uvw[:, 0], uvw[:, 1], bin_size), uvw[:, 2], gap)
    yx = centered[:, 1:]
    _, vectors = np.linalg.eigh(yx.T @ yx)
    tangent, normal = yx @ vectors[:, 1], yx @ vectors[:, 0]
    in_plane = _runs_per_column(_column_key(tangent, centered[:, 0], bin_size), normal, gap)
    return float(max((pca > 1).mean(), (in_plane > 1).mean()))


def sheet_objects(
    surface: np.ndarray,
    *,
    block: int = 48,
    min_piece: int = 100,
    min_object: int = 200,
    max_multi_layer: float = 0.05,
    min_contact: int = 20,
) -> np.ndarray:
    """Label single-layer sheet objects in a binary surface mask.

    Components are found inside blocks, where bridges between sheets are rare, and a
    piece is kept only if it is a single layer. Pieces in face-adjacent blocks are joined
    (largest contact first) only if the joined pieces in those two blocks are still a
    single layer and the group would not hold two pieces of the same block.
    """
    mask = surface > 0
    block_shape = [min(block, size) for size in mask.shape]
    counts = [size // edge for size, edge in zip(mask.shape, block_shape)]
    pieces = np.zeros(mask.shape, np.int32)
    piece_block = [None]
    for index in np.ndindex(*counts):
        region = tuple(slice(i * edge, (i + 1) * edge) for i, edge in zip(index, block_shape))
        labels, count = ndi.label(mask[region], structure=SIX_CONNECTED)
        if count == 0:
            continue
        for label, bounds in enumerate(ndi.find_objects(labels), 1):
            local = labels[bounds] == label
            if local.sum() < min_piece:
                continue
            if multi_layer_fraction(np.argwhere(local).astype(np.float32)) > max_multi_layer:
                continue
            piece_block.append(index)
            pieces[region][bounds][local] = len(piece_block) - 1
    n_pieces = len(piece_block) - 1
    if n_pieces == 0:
        return pieces

    contacts: dict[tuple[int, int], int] = {}
    for axis in range(3):
        for k in range(1, counts[axis]):
            lower = [slice(None)] * 3
            upper = [slice(None)] * 3
            lower[axis] = k * block_shape[axis] - 1
            upper[axis] = k * block_shape[axis]
            a, b = pieces[tuple(lower)], pieces[tuple(upper)]
            touching = (a > 0) & (b > 0)
            pairs, pair_counts = np.unique(np.stack([a[touching], b[touching]], 1), axis=0, return_counts=True)
            for (x, y), count in zip(pairs, pair_counts):
                key = (int(min(x, y)), int(max(x, y)))
                contacts[key] = contacts.get(key, 0) + int(count)

    flat = np.flatnonzero(pieces)
    ids = pieces.ravel()[flat]
    order = np.argsort(ids, kind="stable")
    flat, ids = flat[order], ids[order]
    bounds = np.searchsorted(ids, np.arange(1, n_pieces + 2))
    coords = [None] + [
        np.stack(np.unravel_index(flat[bounds[i - 1]:bounds[i]], mask.shape), -1).astype(np.float32)
        for i in range(1, n_pieces + 1)
    ]
    parent = list(range(n_pieces + 1))
    members = {i: {i} for i in range(1, n_pieces + 1)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (x, y), count in sorted(contacts.items(), key=lambda item: -item[1]):
        if count < min_contact:
            continue
        root_x, root_y = find(x), find(y)
        if root_x == root_y:
            continue
        if {piece_block[m] for m in members[root_x]} & {piece_block[m] for m in members[root_y]}:
            continue
        local_blocks = {piece_block[x], piece_block[y]}
        joined = np.concatenate([coords[m] for m in members[root_x] | members[root_y] if piece_block[m] in local_blocks])
        if multi_layer_fraction(joined) > max_multi_layer:
            continue
        parent[root_y] = root_x
        members[root_x] |= members.pop(root_y)

    roots = np.array([find(i) for i in range(n_pieces + 1)])
    labels = roots[pieces]
    sizes = np.bincount(labels.ravel())
    keep = [label for label in np.argsort(-sizes, kind="stable") if label > 0 and sizes[label] >= min_object]
    remap = np.zeros(sizes.size, np.int32)
    remap[keep] = np.arange(1, len(keep) + 1)
    return remap[labels]


def decode_fiber_directions(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Lasagna hemisphere encoding -> unit fiber directions in (z, y, x) order."""
    fx = (nx.astype(np.float32) - 128.0) / 127.0
    fy = (ny.astype(np.float32) - 128.0) / 127.0
    fz = np.sqrt(np.clip(1.0 - fx * fx - fy * fy, 0.0, 1.0))
    direction = np.stack([fz, fy, fx], -1)
    return direction / (np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-6)


def fiber_objects(
    present: np.ndarray,
    direction: np.ndarray,
    *,
    min_voxels: int = 30,
    min_cos: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """6-connected fiber components joining neighbours only if their directions agree.

    Returns the labels and each object's mean direction (row 0 unused).
    """
    index = np.full(present.shape, -1, np.int64)
    voxels = np.nonzero(present)
    count = voxels[0].size
    if count == 0:
        return np.zeros(present.shape, np.int32), np.zeros((1, 3), np.float32)
    index[voxels] = np.arange(count)
    rows, cols = [], []
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        lower, upper = tuple(lower), tuple(upper)
        agree = present[lower] & present[upper]
        agree &= np.abs((direction[lower] * direction[upper]).sum(-1)) >= min_cos
        rows.append(index[lower][agree])
        cols.append(index[upper][agree])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(count, count))
    _, component = connected_components(graph, directed=False)
    sizes = np.bincount(component)
    order = np.argsort(-sizes, kind="stable")
    kept = order[sizes[order] >= min_voxels]
    new_id = np.zeros(sizes.size, np.int32)
    new_id[kept] = np.arange(1, kept.size + 1)
    labels = np.zeros(present.shape, np.int32)
    labels[voxels] = new_id[component]
    # Directions are unoriented, so each object's direction is the principal axis of
    # its voxels' direction vectors rather than their (sign-dependent) mean.
    vectors = direction[voxels]
    object_ids = new_id[component]
    means = np.zeros((kept.size + 1, 3), np.float32)
    for label in range(1, kept.size + 1):
        v = vectors[object_ids == label]
        _, eigvecs = np.linalg.eigh(v.T @ v)
        means[label] = eigvecs[:, -1]
    return labels, means


def stacked_sheet_pairs(
    labels: np.ndarray,
    *,
    max_distance: float = 24.0,
    neighbourhood: int = 48,
    min_points: int = 50,
    single_layer: float = 0.05,
    stacked: float = 0.2,
    max_points: int = 1500,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    """Sheet pairs that are provably different layers.

    Near the closest approach of two single-layer objects, their union must clearly fail
    the single-layer test: the two pieces lie on top of each other along the normal.
    Pieces of one sheet split by the conservative stitching lie edge to edge and pass.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng() if rng is None else rng
    n = int(labels.max())
    if n < 2:
        return []
    flat = np.flatnonzero(labels)
    ids = labels.ravel()[flat]
    order = np.argsort(ids, kind="stable")
    flat, ids = flat[order], ids[order]
    bounds = np.searchsorted(ids, np.arange(1, n + 2))
    coords = {}
    for label in range(1, n + 1):
        points = flat[bounds[label - 1]:bounds[label]]
        if points.size > max_points:
            points = rng.choice(points, max_points, replace=False)
        coords[label] = np.stack(np.unravel_index(points, labels.shape), -1).astype(np.float32)
    trees = {label: cKDTree(points) for label, points in coords.items()}
    half = neighbourhood / 2.0
    pairs = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            distance, nearest = trees[b].query(coords[a], k=1, distance_upper_bound=max_distance)
            close = np.isfinite(distance)
            if not close.any():
                continue
            centre = coords[a][np.flatnonzero(close)[np.argmin(distance[close])]]
            local_a = coords[a][np.all(np.abs(coords[a] - centre) <= half, axis=1)]
            local_b = coords[b][np.all(np.abs(coords[b] - centre) <= half, axis=1)]
            if len(local_a) < min_points or len(local_b) < min_points:
                continue
            if multi_layer_fraction(local_a) > single_layer or multi_layer_fraction(local_b) > single_layer:
                continue
            if multi_layer_fraction(np.concatenate([local_a, local_b])) >= stacked:
                pairs.append((a, b))
    return pairs


def crossing_fiber_pairs(directions: np.ndarray, max_cos: float = 0.5) -> list[tuple[int, int]]:
    """Fiber pairs at a large angle (e.g. the crossing layers of a papyrus sheet)."""
    n = directions.shape[0] - 1
    if n < 2:
        return []
    cos = np.abs(directions[1:] @ directions[1:].T)
    a, b = np.nonzero(np.triu(cos < max_cos, 1))
    return [(int(i) + 1, int(j) + 1) for i, j in zip(a, b)]


@dataclass
class SlabObjects:
    """Objects of one slab: labels in slab coordinates, kinds and provable differences."""

    sheet_labels: np.ndarray
    fiber_labels: np.ndarray
    negative_pairs: list[tuple[int, int]]  # in combined ids: sheets 1..S, fibers S+1..S+F
    n_sheets: int
    n_fibers: int


def slab_objects(
    surface: np.ndarray | None,
    fiber_presence: np.ndarray | None,
    fiber_direction: np.ndarray | None,
    *,
    presence_threshold: float = 127.0,
    rng: np.random.Generator | None = None,
) -> SlabObjects:
    shape = surface.shape if surface is not None else fiber_presence.shape
    sheets = sheet_objects(surface) if surface is not None else np.zeros(shape, np.int32)
    n_sheets = int(sheets.max())
    if fiber_presence is not None:
        fibers, directions = fiber_objects(fiber_presence > presence_threshold, fiber_direction)
    else:
        fibers, directions = np.zeros(shape, np.int32), np.zeros((1, 3), np.float32)
    n_fibers = int(fibers.max())
    negatives = stacked_sheet_pairs(sheets, rng=rng) if n_sheets > 1 else []
    negatives += [(a + n_sheets, b + n_sheets) for a, b in crossing_fiber_pairs(directions)]
    return SlabObjects(sheets, fibers, negatives, n_sheets, n_fibers)
