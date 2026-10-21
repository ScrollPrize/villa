"""Pseudo-object labels (papyrus sheets and fibers) for thin source slabs.

Sheets come from a binary surface prediction, fibers from a fiber presence field with
an encoded fiber direction. Both are split into objects conservatively: an object may be
one sheet or fiber cut into several pieces, but never two sheets or two crossing fibers
merged into one. Pairs of objects are marked as different only when that is provable
(stacked sheet layers, fibers at an angle); every other pair is left unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import fsspec
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import tifffile
import zarr

from dinovol_2.dataset.point_annotations import map_scale0_voxel_centers, xyz_to_zyx

SIX_CONNECTED = ndi.generate_binary_structure(3, 1)
EIGHT_CONNECTED_2D = np.ones((3, 3), dtype=bool)
SHEET_KIND = 1
FIBER_KIND = 2
INK_KIND = 3


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
    min_span: float = 0.0,
    dedup_distance: float = 1.5,
    max_fibers: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """6-connected fiber components joining neighbours only if their directions agree.

    Components whose voxels fall on average within `dedup_distance` voxels of a larger
    component are dropped as duplicates. Returns the labels and each object's principal
    direction (row 0 unused).
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
    candidates = order[sizes[order] >= min_voxels]
    if candidates.size == 0:
        return np.zeros(present.shape, np.int32), np.zeros((1, 3), np.float32)

    voxel_coords = np.stack(voxels, axis=-1).astype(np.float32)
    vectors = direction[voxels]
    comp_order = np.argsort(component, kind="stable")
    comp_sorted = component[comp_order]
    comp_bounds = np.searchsorted(comp_sorted, np.arange(sizes.size + 1))

    kept_components: list[int] = []
    kept_trees: list[cKDTree] = []
    kept_coords: list[np.ndarray] = []
    kept_means: list[np.ndarray] = []

    for comp_id in candidates:
        if max_fibers > 0 and len(kept_components) >= max_fibers:
            break
        idx = comp_order[comp_bounds[comp_id]:comp_bounds[comp_id + 1]]
        pts = voxel_coords[idx]
        v = vectors[idx]
        _, eigvecs = np.linalg.eigh(v.T @ v)
        principal = eigvecs[:, -1]
        if min_span > 0.0:
            proj = pts @ principal
            if float(proj.max() - proj.min()) < min_span:
                continue
        if dedup_distance > 0.0:
            is_dup = False
            tree_b = None
            for prev_tree, prev_pts in zip(kept_trees, kept_coords):
                # Fast bounding-box precheck before KD-tree query
                if np.any(pts.min(0) - prev_pts.max(0) > dedup_distance * 4.0) or np.any(
                    prev_pts.min(0) - pts.max(0) > dedup_distance * 4.0
                ):
                    continue
                dist_b_to_a, _ = prev_tree.query(pts, k=1)
                if float(dist_b_to_a.mean()) <= dedup_distance:
                    is_dup = True
                    break
                if tree_b is None:
                    tree_b = cKDTree(pts)
                dist_a_to_b, _ = tree_b.query(prev_pts, k=1)
                if float(dist_a_to_b.mean()) <= dedup_distance:
                    is_dup = True
                    break
            if is_dup:
                continue
            kept_trees.append(cKDTree(pts) if tree_b is None else tree_b)
            kept_coords.append(pts)
        kept_components.append(int(comp_id))
        kept_means.append(principal)

    if not kept_components:
        return np.zeros(present.shape, np.int32), np.zeros((1, 3), np.float32)

    kept_arr = np.asarray(kept_components, dtype=np.int64)
    new_id = np.zeros(sizes.size, np.int32)
    new_id[kept_arr] = np.arange(1, kept_arr.size + 1)
    labels = np.zeros(present.shape, np.int32)
    labels[voxels] = new_id[component]
    means = np.zeros((kept_arr.size + 1, 3), np.float32)
    means[1:] = np.stack(kept_means, axis=0)
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
    max_sheets: int = 24,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int]]:
    """Sheet pairs that are provably different layers.

    Near the closest approach of two single-layer objects, their union must clearly fail
    the single-layer test: the two pieces lie on top of each other along the normal.
    Pieces of one sheet split by the conservative stitching lie edge to edge and pass.
    """
    rng = np.random.default_rng() if rng is None else rng
    n = int(labels.max())
    if max_sheets > 0:
        n = min(n, max_sheets)
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


def distinct_fiber_pairs(fibers_or_directions: int | np.ndarray) -> list[tuple[int, int]]:
    """All pairs of deduplicated fiber objects (1..n_fibers)."""
    n_fibers = (
        int(fibers_or_directions.shape[0] - 1)
        if isinstance(fibers_or_directions, np.ndarray)
        else int(fibers_or_directions)
    )
    if n_fibers < 2:
        return []
    return [(a, b) for a in range(1, n_fibers + 1) for b in range(a + 1, n_fibers + 1)]


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
    fiber_min_voxels: int = 30,
    fiber_min_span: float = 0.0,
    fiber_dedup_distance: float = 1.5,
    rng: np.random.Generator | None = None,
) -> SlabObjects:
    shape = surface.shape if surface is not None else fiber_presence.shape
    sheets = sheet_objects(surface) if surface is not None else np.zeros(shape, np.int32)
    n_sheets = int(sheets.max())
    if fiber_presence is not None:
        fibers, _ = fiber_objects(
            fiber_presence > presence_threshold,
            fiber_direction,
            min_voxels=fiber_min_voxels,
            min_span=fiber_min_span,
            dedup_distance=fiber_dedup_distance,
        )
    else:
        fibers = np.zeros(shape, np.int32)
    n_fibers = int(fibers.max())
    negatives = stacked_sheet_pairs(sheets, rng=rng) if n_sheets > 1 else []
    negatives += [(a + n_sheets, b + n_sheets) for a, b in distinct_fiber_pairs(n_fibers)]
    return SlabObjects(sheets, fibers, negatives, n_sheets, n_fibers)


@dataclass
class MeshSegment:
    """Pre-mapped 2D (u, v) papyrus surface mesh in selected volume_scale (z, y, x) coordinates."""

    segment_id: str
    zyx: np.ndarray        # [H, W, 3] float32 in selected volume_scale (z, y, x) coordinates
    valid: np.ndarray      # [H, W] bool
    ink_state: np.ndarray  # [H, W] int8: +1 confirmed ink, -1 confirmed non-ink surface, 0 unsupervised
    bbox_lo: np.ndarray    # [3] float32 (z_min, y_min, x_min)
    bbox_hi: np.ndarray    # [3] float32 (z_max, y_max, x_max)
    has_ink: bool
    tile_size: int = 64
    tile_lo: np.ndarray | None = None
    tile_hi: np.ndarray | None = None
    tile_valid: np.ndarray | None = None
    anchor_valid_zyx: np.ndarray | None = None
    anchor_ink_zyx: np.ndarray | None = None
    n_valid_points: int = 0
    n_ink_points: int = 0

    def ensure_spatial_index(self) -> None:
        if self.n_valid_points <= 0:
            self.n_valid_points = int(self.valid.sum())
        if self.has_ink and self.n_ink_points <= 0:
            self.n_ink_points = int((self.ink_state == 1).sum())
        if self.tile_lo is not None:
            return
        h, w = self.valid.shape
        ts = max(1, int(self.tile_size))
        u_cuts = np.arange(0, h, ts)
        v_cuts = np.arange(0, w, ts)
        with np.errstate(invalid="ignore"):
            t_lo = np.fmin.reduceat(np.fmin.reduceat(self.zyx, u_cuts, axis=0), v_cuts, axis=1)
            t_hi = np.fmax.reduceat(np.fmax.reduceat(self.zyx, u_cuts, axis=0), v_cuts, axis=1)
        self.tile_lo = t_lo.astype(np.float32, copy=False)
        self.tile_hi = t_hi.astype(np.float32, copy=False)
        self.tile_valid = np.isfinite(self.tile_lo[..., 0])

        step = max(1, int(math.sqrt((h * w) / 8192.0)))
        sub_valid = self.valid[::step, ::step]
        sub_zyx = self.zyx[::step, ::step]
        valid_pts = sub_zyx[sub_valid]
        if valid_pts.shape[0] == 0:
            valid_pts = self.zyx[self.valid]
        self.anchor_valid_zyx = valid_pts.astype(np.float32, copy=False)

        if self.has_ink:
            ink_step = max(1, int(math.sqrt((h * w) / 32768.0)))
            sub_ink = self.ink_state[::ink_step, ::ink_step] == 1
            ink_pts = self.zyx[::ink_step, ::ink_step][sub_ink]
            if ink_pts.shape[0] == 0:
                ink_pts = self.zyx[self.ink_state == 1]
            self.anchor_ink_zyx = ink_pts.astype(np.float32, copy=False)
        else:
            self.anchor_ink_zyx = np.empty((0, 3), dtype=np.float32)


@dataclass
class CropMeshSheet:
    """One single-layer 2D UV connected component of a mesh inside a 3D crop overlap."""

    coords: np.ndarray     # [N, 3] float32 in crop-overlap coordinates
    ink_state: np.ndarray  # [N] int8: +1 ink, -1 non-ink, 0 unsupervised
    uv_coords: np.ndarray  # [N, 2] int32 (u, v) grid indices


def parse_sample_and_volume_id(volume_path: str) -> tuple[str | None, str | None]:
    """Extract (sample_id, volume_id) from a standard Vesuvius volume path or URL."""
    path_str = str(volume_path).rstrip("/")
    sample_match = re.search(r"/(PHerc[^/]+|Scroll[^/]+|Frag[^/]+)/volumes/(\d{14})", path_str)
    if sample_match:
        return sample_match.group(1), sample_match.group(2)
    vol_match = re.search(r"/(\d{14})[^/]*\.zarr$", path_str)
    if vol_match:
        return None, vol_match.group(1)
    return None, None


def _read_tiff(url_or_path: str, storage_options: dict[str, Any] | None = None) -> np.ndarray:
    path_str = str(url_or_path)
    opts = dict(storage_options or {})
    if path_str.startswith("s3://"):
        opts.setdefault("anon", True)
    with fsspec.open(path_str, "rb", **opts) as f:
        data = f.read()
    return np.asarray(tifffile.imread(io.BytesIO(data)))


def _open_ink_zarr_level(url_or_path: str, level: str = "4", storage_options: dict[str, Any] | None = None) -> np.ndarray:
    path_str = str(url_or_path).rstrip("/")
    opts = dict(storage_options or {})
    if path_str.startswith("s3://"):
        opts.setdefault("anon", True)
        arr = zarr.open(path_str, path=level, mode="r", storage_options=opts)
    else:
        arr = zarr.open(path_str, path=level, mode="r")
    return np.asarray(arr[:])


def build_mesh_segment_from_arrays(
    segment_id: str,
    x_scale0: np.ndarray,
    y_scale0: np.ndarray,
    z_scale0: np.ndarray,
    scale0_shape: tuple[int, int, int] | None = None,
    selected_shape: tuple[int, int, int] | None = None,
    *,
    transform_matrix: np.ndarray | None = None,
    coordinate_scale: float = 1.0,
    inklabels: np.ndarray | None = None,
    supervision: np.ndarray | None = None,
) -> MeshSegment | None:
    """Convert scale-0 (x, y, z) tifxyz arrays and optional ink/supervision masks into a MeshSegment."""
    x = x_scale0.astype(np.float64)
    y = y_scale0.astype(np.float64)
    z = z_scale0.astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (x >= 0.0) & (y >= 0.0) & (z >= 0.0)
    valid &= ~((x == 0.0) & (y == 0.0) & (z == 0.0))
    if not valid.any():
        return None

    if coordinate_scale != 1.0:
        x = np.where(valid, x * coordinate_scale, np.nan)
        y = np.where(valid, y * coordinate_scale, np.nan)
        z = np.where(valid, z * coordinate_scale, np.nan)

    xyz = np.stack([x, y, z], axis=-1)
    if transform_matrix is not None:
        mat = np.asarray(transform_matrix, dtype=np.float64)
        valid_pts = xyz[valid]
        transformed = valid_pts @ mat[:, :3].T + mat[:, 3]
        xyz = np.full_like(xyz, np.nan)
        xyz[valid] = transformed

    valid_xyz = xyz[valid]
    if scale0_shape is None:
        max_zyx = xyz_to_zyx(valid_xyz).max(axis=0)
        scale0_shape = tuple(int(np.ceil(v)) + 2 for v in max_zyx)
    if selected_shape is None:
        selected_shape = scale0_shape
    mapped_zyx = map_scale0_voxel_centers(xyz_to_zyx(valid_xyz), scale0_shape, selected_shape).astype(np.float32)
    in_bounds = np.all((mapped_zyx >= 0.0) & (mapped_zyx < np.asarray(selected_shape, dtype=np.float32)), axis=-1)
    if not in_bounds.any():
        return None

    full_zyx = np.full(xyz.shape, np.nan, dtype=np.float32)
    valid_indices = np.argwhere(valid)
    kept_indices = valid_indices[in_bounds]
    kept_zyx = mapped_zyx[in_bounds]
    full_zyx[kept_indices[:, 0], kept_indices[:, 1]] = kept_zyx

    final_valid = np.zeros(x.shape, dtype=bool)
    final_valid[kept_indices[:, 0], kept_indices[:, 1]] = True

    h, w = x.shape
    ink_state = np.zeros((h, w), dtype=np.int8)
    if inklabels is not None and supervision is not None:
        sup = np.asarray(supervision)
        ink = np.asarray(inklabels)
        u_idx = np.clip((np.arange(h) * sup.shape[0]) // max(1, h), 0, sup.shape[0] - 1)
        v_idx = np.clip((np.arange(w) * sup.shape[1]) // max(1, w), 0, sup.shape[1] - 1)
        sup_grid = sup[u_idx[:, None], v_idx[None, :]] > 0
        ink_grid = ink[np.clip((np.arange(h) * ink.shape[0]) // max(1, h), 0, ink.shape[0] - 1)[:, None],
                       np.clip((np.arange(w) * ink.shape[1]) // max(1, w), 0, ink.shape[1] - 1)[None, :]] > 0
        ink_state = np.where(final_valid & sup_grid, np.where(ink_grid, 1, -1), 0).astype(np.int8)

    bbox_lo = kept_zyx.min(axis=0).astype(np.float32)
    bbox_hi = kept_zyx.max(axis=0).astype(np.float32)
    has_ink = bool((ink_state == 1).any())
    seg = MeshSegment(
        segment_id=str(segment_id),
        zyx=full_zyx,
        valid=final_valid,
        ink_state=ink_state,
        bbox_lo=bbox_lo,
        bbox_hi=bbox_hi,
        has_ink=has_ink,
    )
    seg.ensure_spatial_index()
    return seg


def save_mesh_segments_cache(cache_path: str | Path, segments: list[MeshSegment]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {"n_segments": np.asarray(len(segments), dtype=np.int32)}
    for idx, seg in enumerate(segments):
        arrays[f"seg_{idx}_id"] = np.asarray(seg.segment_id)
        arrays[f"seg_{idx}_zyx"] = seg.zyx.astype(np.float32)
        arrays[f"seg_{idx}_valid"] = np.packbits(seg.valid, axis=-1)
        arrays[f"seg_{idx}_shape"] = np.asarray(seg.valid.shape, dtype=np.int32)
        arrays[f"seg_{idx}_ink"] = seg.ink_state.astype(np.int8)
        arrays[f"seg_{idx}_bbox_lo"] = seg.bbox_lo.astype(np.float32)
        arrays[f"seg_{idx}_bbox_hi"] = seg.bbox_hi.astype(np.float32)
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_path, **arrays)
    tmp_path.replace(path)


def load_mesh_segments_cache(cache_path: str | Path) -> list[MeshSegment]:
    with np.load(cache_path, allow_pickle=False) as data:
        n_segments = int(data["n_segments"])
        segments: list[MeshSegment] = []
        for idx in range(n_segments):
            shape = tuple(int(v) for v in data[f"seg_{idx}_shape"])
            valid = np.unpackbits(data[f"seg_{idx}_valid"], axis=-1)[:, :shape[1]].astype(bool)
            zyx = data[f"seg_{idx}_zyx"].astype(np.float32)
            ink_state = data[f"seg_{idx}_ink"].astype(np.int8)
            bbox_lo = data[f"seg_{idx}_bbox_lo"].astype(np.float32)
            bbox_hi = data[f"seg_{idx}_bbox_hi"].astype(np.float32)
            seg = MeshSegment(
                segment_id=str(data[f"seg_{idx}_id"]),
                zyx=zyx,
                valid=valid,
                ink_state=ink_state,
                bbox_lo=bbox_lo,
                bbox_hi=bbox_hi,
                has_ink=bool((ink_state == 1).any()),
            )
            seg.ensure_spatial_index()
            segments.append(seg)
        return segments


_METADATA_CACHE: dict[str, dict[str, Any]] = {}


def _load_metadata_json(metadata_path: str | Path) -> dict[str, Any]:
    key = str(metadata_path)
    if key not in _METADATA_CACHE:
        with fsspec.open(key, "rt", encoding="utf-8") as f:
            _METADATA_CACHE[key] = json.load(f)
    return _METADATA_CACHE[key]


def load_or_build_volume_meshes(
    volume_path: str,
    volume_scale: int,
    selected_shape: tuple[int, int, int],
    *,
    sample_id: str | None = None,
    volume_id: str | None = None,
    metadata_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    mesh_cache_path: str | Path | None = None,
    s3_storage_options: dict[str, Any] | None = None,
) -> list[MeshSegment]:
    """Load pre-cached MeshSegments for a volume, or build & cache them from metadata.json."""
    if mesh_cache_path is not None and Path(mesh_cache_path).is_file():
        return load_mesh_segments_cache(mesh_cache_path)

    parsed_sample, parsed_vol = parse_sample_and_volume_id(volume_path)
    sample_id = sample_id or parsed_sample
    volume_id = volume_id or parsed_vol
    if not volume_id:
        return []

    cache_file: Path | None = None
    if mesh_cache_path is not None:
        cache_file = Path(mesh_cache_path)
    elif cache_dir is not None:
        prefix = f"{sample_id}_{volume_id}" if sample_id else volume_id
        cache_file = Path(cache_dir) / f"{prefix}_scale{int(volume_scale)}.npz"

    if cache_file is not None and cache_file.is_file():
        return load_mesh_segments_cache(cache_file)

    if metadata_path is None:
        candidates = []
        if cache_dir is not None:
            candidates.append(Path(cache_dir) / "metadata.json")
        candidates.extend([
            Path("/ephemeral/sean/dinovol_perf/mesh_cache_8_9um/metadata.json"),
            Path("/ephemeral/sean/dinovol_perf/tmp/vesuvius_metadata.json"),
            Path("/tmp/vesuvius_metadata.json"),
        ])
        for cand in candidates:
            if cand.is_file():
                metadata_path = cand
                break
        if metadata_path is None:
            return []

    path_str = str(metadata_path)
    if not path_str.startswith(("http://", "https://", "s3://")) and not Path(path_str).is_file():
        return []

    metadata = _load_metadata_json(metadata_path)
    raw_samples = metadata.get("samples", {})
    samples_iter = raw_samples.values() if isinstance(raw_samples, Mapping) else raw_samples

    target_sample = None
    if isinstance(raw_samples, Mapping) and sample_id and sample_id in raw_samples:
        target_sample = raw_samples[sample_id]
    else:
        for s in samples_iter:
            if not isinstance(s, Mapping):
                continue
            sid = s.get("sample", {}).get("id")
            if sample_id and sid == sample_id:
                target_sample = s
                break
            vols = s.get("volumes") or {}
            vols_iter = vols.values() if isinstance(vols, Mapping) else vols
            if (isinstance(vols, Mapping) and volume_id in vols) or any(
                isinstance(v, Mapping) and v.get("id") == volume_id for v in vols_iter
            ):
                target_sample = s
                break

    if target_sample is None:
        return []

    scale0_shape = None
    vols = target_sample.get("volumes") or {}
    vols_iter = vols.values() if isinstance(vols, Mapping) else vols
    for v in vols_iter:
        if isinstance(v, Mapping) and v.get("id") == volume_id:
            shape = v.get("properties", {}).get("shape")
            if shape and len(shape) == 3:
                scale0_shape = tuple(int(dim) for dim in shape)
            break
    if scale0_shape is None:
        factor = 2 ** int(volume_scale)
        scale0_shape = tuple(int(dim) * factor for dim in selected_shape)

    transforms_to_target: dict[str, np.ndarray] = {}
    vol_transforms = target_sample.get("sample", {}).get("properties", {}).get("volume_transforms") or []
    for vt in vol_transforms:
        if not isinstance(vt, Mapping):
            continue
        from_vid = str(vt.get("from_volume_id", ""))
        if vt.get("to_volume_id") == volume_id and "matrix" in vt:
            transforms_to_target[from_vid] = np.asarray(vt["matrix"], dtype=np.float64)
        for sub in vt.get("transforms") or []:
            if isinstance(sub, Mapping) and sub.get("to_volume_id") == volume_id and "matrix" in sub:
                transforms_to_target[from_vid] = np.asarray(sub["matrix"], dtype=np.float64)

    def _extract_s3_url(item: Mapping[str, Any]) -> str | None:
        if item.get("url"):
            u = str(item["url"]).rstrip("/")
            if u.startswith("https://vesuvius-challenge-open-data.s3.amazonaws.com/"):
                return "s3://vesuvius-challenge-open-data/" + u.split("amazonaws.com/", 1)[1]
            return u
        for origin in item.get("origins") or []:
            if not isinstance(origin, Mapping):
                continue
            rel_path = str(origin.get("path", "")).strip("/")
            if not rel_path:
                continue
            for root in origin.get("access_roots") or []:
                if isinstance(root, Mapping) and root.get("type") == "s3" and root.get("url"):
                    return f"{str(root['url']).rstrip('/')}/{rel_path}"
            return f"s3://vesuvius-challenge-open-data/{rel_path}"
        return None

    def _build_one_segment(seg: dict[str, Any]) -> MeshSegment | None:
        if not isinstance(seg, Mapping):
            return None
        seg_id = str(seg.get("id", "unknown"))
        orig_vol = str(seg.get("original_volume_id", ""))
        reps = list(seg.get("data") or seg.get("representations") or [])

        tif_url = None
        transform_matrix = None
        coord_scale = 1.0

        for rep in reps:
            if not isinstance(rep, Mapping):
                continue
            params = rep.get("parameters") or {}
            if rep.get("type") == "tifxyz-transformed" and params.get("target_volume") == volume_id:
                tif_url = _extract_s3_url(rep)
                if tif_url:
                    break
        if tif_url is None and (orig_vol == volume_id or orig_vol in transforms_to_target):
            for rep in reps:
                if isinstance(rep, Mapping) and rep.get("type") == "tifxyz-normalized":
                    tif_url = _extract_s3_url(rep)
                    if tif_url:
                        break
            if tif_url is None:
                for rep in reps:
                    if isinstance(rep, Mapping) and rep.get("type") == "tifxyz":
                        tif_url = _extract_s3_url(rep)
                        if tif_url:
                            coord_scale = float((seg.get("properties") or {}).get("original_volume_downscale", 1.0))
                            break
            if orig_vol != volume_id:
                transform_matrix = transforms_to_target[orig_vol]

        if tif_url is None:
            return None

        base_url = str(tif_url).rstrip("/")
        try:
            x_arr = _read_tiff(f"{base_url}/x.tif", s3_storage_options)
            y_arr = _read_tiff(f"{base_url}/y.tif", s3_storage_options)
            z_arr = _read_tiff(f"{base_url}/z.tif", s3_storage_options)
        except Exception:
            return None

        ink_arr = sup_arr = None
        for rep in reps:
            if isinstance(rep, Mapping) and rep.get("type") == "ink-labels":
                ink_base = _extract_s3_url(rep)
                if not ink_base:
                    continue
                try:
                    ink_arr = _open_ink_zarr_level(f"{ink_base}/inklabels.zarr", "4", s3_storage_options)
                    sup_arr = _open_ink_zarr_level(f"{ink_base}/supervision.zarr", "4", s3_storage_options)
                except Exception:
                    ink_arr = sup_arr = None
                if ink_arr is not None and sup_arr is not None:
                    break

        return build_mesh_segment_from_arrays(
            seg_id,
            x_arr,
            y_arr,
            z_arr,
            scale0_shape,
            selected_shape,
            transform_matrix=transform_matrix,
            coordinate_scale=coord_scale,
            inklabels=ink_arr,
            supervision=sup_arr,
        )

    raw_segs = target_sample.get("segments") or {}
    raw_segments = list(raw_segs.values()) if isinstance(raw_segs, Mapping) else list(raw_segs)
    segments: list[MeshSegment] = []
    if raw_segments:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(16, len(raw_segments))) as pool:
            for built in pool.map(_build_one_segment, raw_segments):
                if built is not None:
                    segments.append(built)

    if cache_file is not None:
        try:
            save_mesh_segments_cache(cache_file, segments)
        except OSError:
            pass
    return segments


def _sample_spread_indices(
    uv_coords: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample `count` indices spread across a 2D (u, v) component via stratified quantiles."""
    n = len(uv_coords)
    if count >= n:
        return np.arange(n, dtype=np.int64)
    if count <= 0:
        return np.empty((0,), dtype=np.int64)
    # Sort along a random 2D projection in (u, v) and draw one random index per stratum.
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    proj = uv_coords[:, 0] * np.cos(angle) + uv_coords[:, 1] * np.sin(angle)
    order = np.argsort(proj, kind="stable")
    edges = np.linspace(0, n, count + 1, dtype=np.int64)
    picks = [int(order[rng.integers(edges[i], max(edges[i] + 1, edges[i + 1]))]) for i in range(count)]
    return np.asarray(picks, dtype=np.int64)


def crop_mesh_objects(
    meshes: list[MeshSegment],
    crop_lo: np.ndarray,
    crop_hi: np.ndarray,
    *,
    min_points: int = 16,
    dedup_distance: float = 1.5,
    min_separation: float = 3.0,
    max_sheets: int = 6,
    points_per_sheet: int = 48,
    rng: np.random.Generator | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[int, int]]]:
    """Extract single-layer mesh sheet components and their negative pairs in [crop_lo, crop_hi).

    Each mesh is split into 2D (u, v) connected components inside the crop so multi-wrap
    spiral segments become separate sheet objects. Duplicate overlapping tracings (within
    `dedup_distance` voxels on average) are deduplicated, and sheet pairs separated by at
    least `min_separation` voxels in 3D are marked as negatives.

    Returns:
      - sampled_sheets: list of (coords_in_crop [K, 3] float32, ink_state [K] int8) for sheets 1..S
      - negative_pairs: list of 1-based (a, b) sheet pairs that are provably separate layers
    """
    rng = np.random.default_rng() if rng is None else rng
    crop_lo = np.asarray(crop_lo, dtype=np.float32)
    crop_hi = np.asarray(crop_hi, dtype=np.float32)

    candidates: list[CropMeshSheet] = []
    for seg in meshes:
        if np.any(seg.bbox_hi < crop_lo) or np.any(seg.bbox_lo >= crop_hi):
            continue
        seg.ensure_spatial_index()
        tile_hit = seg.tile_valid & np.all((seg.tile_hi >= crop_lo) & (seg.tile_lo < crop_hi), axis=-1)
        if not tile_hit.any():
            continue
        h, w = seg.valid.shape
        ts = max(1, int(seg.tile_size))
        t_labels, n_tcc = ndi.label(tile_hit, structure=EIGHT_CONNECTED_2D)
        for t_bounds in ndi.find_objects(t_labels):
            if t_bounds is None:
                continue
            u0 = t_bounds[0].start * ts
            u1 = min(h, t_bounds[0].stop * ts)
            v0 = t_bounds[1].start * ts
            v1 = min(w, t_bounds[1].stop * ts)
            sub_zyx = seg.zyx[u0:u1, v0:v1]
            sub_valid = seg.valid[u0:u1, v0:v1]
            with np.errstate(invalid="ignore"):
                inside = sub_valid & np.all((sub_zyx >= crop_lo) & (sub_zyx < crop_hi), axis=-1)
            if int(inside.sum()) < min_points:
                continue
            cc_labels, n_cc = ndi.label(inside, structure=EIGHT_CONNECTED_2D)
            if n_cc == 0:
                continue
            for label_id, bounds in enumerate(ndi.find_objects(cc_labels), 1):
                if bounds is None:
                    continue
                local_mask = cc_labels[bounds] == label_id
                if int(local_mask.sum()) < min_points:
                    continue
                u_local, v_local = np.nonzero(local_mask)
                u = u_local + (u0 + bounds[0].start)
                v = v_local + (v0 + bounds[1].start)
                coords = seg.zyx[u, v] - crop_lo
                ink = seg.ink_state[u, v]
                uv = np.stack([u, v], axis=-1).astype(np.int32)
                candidates.append(CropMeshSheet(coords=coords, ink_state=ink, uv_coords=uv))

    if not candidates:
        return [], []

    # Prioritize sheets with supervised ink (+1 or -1) and larger surface area, with random tie-breaking.
    perm = rng.permutation(len(candidates))
    candidates = [candidates[i] for i in perm]
    candidates.sort(key=lambda c: (int((c.ink_state == 1).any()), int((c.ink_state != 0).any()), len(c.coords)), reverse=True)

    kept_sheets: list[CropMeshSheet] = []
    kept_trees: list[cKDTree] = []
    for cand in candidates:
        if len(kept_sheets) >= max_sheets:
            break
        pts = cand.coords
        if len(pts) > 2000:
            sub_idx = rng.choice(len(pts), 2000, replace=False)
            pts_sub = pts[sub_idx]
        else:
            pts_sub = pts
        tree_cand = None
        is_dup = False
        if dedup_distance > 0.0:
            for prev_sheet, prev_tree in zip(kept_sheets, kept_trees):
                dist_c_to_p, _ = prev_tree.query(pts_sub, k=1)
                if float(dist_c_to_p.mean()) <= dedup_distance:
                    is_dup = True
                    break
                if tree_cand is None:
                    tree_cand = cKDTree(pts_sub)
                prev_pts = prev_tree.data
                dist_p_to_c, _ = tree_cand.query(prev_pts, k=1)
                if float(dist_p_to_c.mean()) <= dedup_distance:
                    is_dup = True
                    break
        if is_dup:
            continue
        kept_sheets.append(cand)
        kept_trees.append(cKDTree(pts_sub) if tree_cand is None else tree_cand)

    n_sheets = len(kept_sheets)
    negative_pairs: list[tuple[int, int]] = []
    for a in range(n_sheets):
        for b in range(a + 1, n_sheets):
            dist_ab, _ = kept_trees[b].query(kept_trees[a].data, k=1)
            dist_ba, _ = kept_trees[a].query(kept_trees[b].data, k=1)
            # Closest-approach separation (10th percentile of nearest-neighbour distances)
            sep = min(float(np.percentile(dist_ab, 10)), float(np.percentile(dist_ba, 10)))
            if sep >= min_separation:
                negative_pairs.append((a + 1, b + 1))

    sampled_sheets: list[tuple[np.ndarray, np.ndarray]] = []
    for sheet in kept_sheets:
        take = min(points_per_sheet, len(sheet.coords))
        ink_pos = np.flatnonzero(sheet.ink_state == 1)
        ink_neg = np.flatnonzero(sheet.ink_state == -1)
        if ink_pos.size > 0:
            n_pos = min(ink_pos.size, max(1, take // 2) if ink_neg.size > 0 else take)
            chosen_pos = ink_pos[_sample_spread_indices(sheet.uv_coords[ink_pos], n_pos, rng)]
            if ink_neg.size > 0 and take > n_pos:
                n_neg = min(ink_neg.size, take - n_pos)
                chosen_neg = ink_neg[_sample_spread_indices(sheet.uv_coords[ink_neg], n_neg, rng)]
                chosen = np.concatenate([chosen_pos, chosen_neg])
            else:
                chosen = chosen_pos
            if chosen.size < take:
                rem_mask = np.ones(len(sheet.coords), dtype=bool)
                rem_mask[chosen] = False
                rem_idx = np.flatnonzero(rem_mask)
                extra = rem_idx[_sample_spread_indices(sheet.uv_coords[rem_idx], take - chosen.size, rng)]
                chosen = np.concatenate([chosen, extra])
        else:
            chosen = _sample_spread_indices(sheet.uv_coords, take, rng)
        sampled_sheets.append((sheet.coords[chosen].astype(np.float32), sheet.ink_state[chosen].astype(np.int8)))

    return sampled_sheets, negative_pairs

