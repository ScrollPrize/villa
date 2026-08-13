#!/usr/bin/env python3
"""Find pairs of overlapping tifxyz patches via the surface patch spatial index.

Loads a folder of tifxyz patches (as fit_spiral.py does), builds a
``vc.surface_index.SurfacePatchIndex`` over all of them, then for each patch
queries the index with all of that patch's valid vertices in a batch. Every
*other* patch whose surface lies within ``--tolerance`` of those vertices is an
overlap candidate; the number of the patch's vertices that land on the other
patch is the (directed) overlap amount.

Both directions of each pair are measured, then combined: a pair is reported
when the mutual overlap (min of the two directed counts) is large enough AND each
patch has enough points NOT present on the other (min of the two one-sided
differences) -- the latter ensures both patches contribute unique area and
filters out near-duplicate patches.

Overlapping pairs are written to a JSONL file (one pair per line). For each
reported pair we also synthesise a connecting sequence of points -- a path that
starts on the part of one patch that does not overlap the other, crosses the
overlap, and ends on the part of the other patch that does not overlap the
first -- and write all of them as a single VC point-collection JSON file. Each
pair's JSONL row carries the ``collection_id`` of its synthesised sequence (or
null when no valid path could be found). See ``connect_pair`` for the method.
"""

import bisect
import colorsys
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import click
import numpy as np
import scipy.sparse as sp
import torch
from scipy.ndimage import distance_transform_edt
from scipy.sparse.csgraph import dijkstra as csgraph_dijkstra
from tqdm import tqdm

from tifxyz import Patch, load_tifxyz


def load_patches(patches_path, limit=None, segment_id_filter=lambda s: True):
    """Load every tifxyz patch under ``patches_path`` (mirrors fit_spiral.load_patches)."""
    patches = {}
    failed_count = 0
    for entry in sorted(os.listdir(patches_path)):
        if limit is not None and len(patches) >= limit:
            break
        segment_path = os.path.join(patches_path, entry)
        meta_path = os.path.join(segment_path, 'meta.json')
        if not os.path.isdir(segment_path) or not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            print(f'Warning: failed to read {meta_path}: {e}')
            continue
        if meta.get('format') != 'tifxyz' or not segment_id_filter(entry):
            continue
        try:
            patches[entry] = load_tifxyz(segment_path)
        except Exception as e:
            print(f'Failed to load segment {entry}: {e}')
            failed_count += 1
    if not patches:
        raise RuntimeError('No patches could be loaded')
    print(f'Loaded {len(patches)} patches, {failed_count} failed')
    return patches


def build_surface_index(patches, bbox_padding, sampling_stride):
    """Build a SurfacePatchIndex containing every patch, keyed by patch id."""
    from vc import surface_index

    surfaces = []
    for patch_id, patch in patches.items():
        zyx = patch.zyxs.detach().cpu().contiguous().numpy().astype(np.float32, copy=False)
        scale = patch.scale.detach().cpu().numpy() if hasattr(patch.scale, 'detach') else np.asarray(patch.scale)
        surfaces.append(surface_index.QuadSurface(patch_id, zyx, float(scale[0]), float(scale[1])))

    print(f'building surface patch index over {len(surfaces)} patches '
          f'(bbox_padding={bbox_padding}, sampling_stride={sampling_stride})')
    index = surface_index.SurfacePatchIndex()
    index.rebuild(surfaces, bbox_padding=bbox_padding, sampling_stride=sampling_stride)
    return index


def patch_query_points(patch, query_stride):
    """Return this patch's valid vertices as a contiguous (N, 3) float32 xyz array.

    query_stride subsamples the grid along both axes before dropping invalid
    vertices, so it gives a roughly quadratic reduction in query points.
    """
    zyx = patch.zyxs.detach().cpu().numpy().astype(np.float32, copy=False)
    if query_stride > 1:
        zyx = zyx[::query_stride, ::query_stride]
    zyx = zyx[np.any(zyx != -1.0, axis=-1)]  # keep valid vertices (matches Patch.valid_vertex_mask)
    # The index works in world xyz; Patch stores zyx, so reverse the last axis.
    return np.ascontiguousarray(zyx[:, ::-1])


def find_overlaps(patches, index, tolerance, query_stride, id_to_idx, surface_ids, workers=None):
    """For each patch, count how many of its query points land on every *other* patch.

    Queries every patch against the index, so both directions of a pair are
    measured independently (A's points onto B, and B's points onto A).

    ``id_to_idx``/``surface_ids`` map between patch ids and the integer surface
    indices used for bincounting hits.

    Returns:
      directed: dict (src_id, dst_id) -> {'count', 'dist_sum'}; ``count`` is the
                number of src's (strided) query points that hit dst's surface
                within ``tolerance``, counted at most once per point per patch.
      n_query:  dict src_id -> number of (strided) query points used for src.
      n_full:   dict src_id -> number of full-resolution valid vertices in src.

    The raw counts are in units of strided query points; main() scales them back
    to full-resolution-equivalent vertices (via n_full / n_query) so the reported
    overlap/diff and their thresholds are invariant to ``query_stride``.

    Patches are queried concurrently: index.locate_all_xyz_batch releases the GIL
    and takes only a shared (read) lock on the index, and each patch is an
    independent object touched by one worker, so the per-patch work runs in
    parallel. The (tiny) result dicts are populated in the main thread.
    """
    n_surf = len(surface_ids)

    def query_patch(item):
        patch_id, patch = item
        n_full = int(patch.valid_vertex_mask.sum())
        xyz = patch_query_points(patch, query_stride)
        n_query = xyz.shape[0]
        hits = []  # (other_id, count, dist_sum) for each other patch this one hit
        if n_query:
            self_idx = id_to_idx[patch_id]
            # CSR-packed hits over all query points; surf_idx already indexes
            # surface_ids() (-1 for any unknown surface), so no per-hit id lookup.
            _offsets, surf_idx, distance, _ij = index.locate_all_xyz_batch(xyz, tolerance)

            # locate_all yields at most one hit per surface per query point, so a
            # per-surface bincount over all hits is the number of this patch's
            # points that landed on each other patch (and the summed distance).
            # Drop hits on the patch's own surface (distance ~0) and any unknown.
            keep = (surf_idx != self_idx) & (surf_idx >= 0)
            oi = surf_idx[keep].astype(np.int64)
            if oi.size:
                counts = np.bincount(oi, minlength=n_surf)
                dist_sum = np.bincount(oi, weights=distance[keep].astype(np.float64), minlength=n_surf)
                hits = [(surface_ids[other_idx], int(counts[other_idx]), float(dist_sum[other_idx]))
                        for other_idx in np.nonzero(counts)[0]]
        return patch_id, n_full, n_query, hits

    directed = {}
    n_query = {}
    n_full = {}
    with ThreadPoolExecutor(max_workers=workers or (os.cpu_count() or 4)) as pool:
        futures = [pool.submit(query_patch, item) for item in patches.items()]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='querying patches'):
            patch_id, nf, nq, hits = fut.result()
            n_full[patch_id] = nf
            n_query[patch_id] = nq
            for other_id, count, dist_sum in hits:
                directed[(patch_id, other_id)] = {'count': count, 'dist_sum': dist_sum}
    return directed, n_query, n_full


# ---------------------------------------------------------------------------
# Synthesising connecting point sequences for each overlapping pair.
#
# For a pair {A, B} we build a path that starts at a point of A that does *not*
# lie on B, crosses the overlap region, and ends at a point of B that does not
# lie on A. The path is constructed on each patch's *quad* grid (the cells of
# the tifxyz grid whose four corners are all valid), so every point we sample
# lifts cleanly through Patch.ij_to_zyx and never crosses an invalid region.
# The two quad grids are stitched together by "bridge" edges in the overlap
# (an overlapping A quad is joined to the B quad it projects onto), and we run a
# weighted Dijkstra over the union. Edge weights penalise being near a boundary
# (low distance-transform value) so the path hugs the perpendicular middle of
# each strip rather than its edges.
# ---------------------------------------------------------------------------

# 8-connectivity on the quad grid. Diagonal moves stay on valid quads because
# the segment between two diagonally-adjacent quad centres only ever passes
# through those two quads (it crosses their shared corner, nothing else).
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class PatchGraph:
    """Cached per-patch structures for path finding on the quad grid."""
    patch: Patch
    valid_quad: np.ndarray       # (Hq, Wq) bool
    dt: np.ndarray               # (Hq, Wq) float32, distance to nearest invalid/border quad
    ids: np.ndarray              # (Hq, Wq) int64, local node id per valid quad else -1
    node_qi: np.ndarray          # (N,) int64, quad row per node
    node_qj: np.ndarray          # (N,) int64, quad col per node
    quad_center_xyz: np.ndarray  # (N, 3) float32, quad centre in world xyz (for index queries)
    csr: sp.csr_matrix           # (N, N) directed weighted adjacency among valid quads
    n: int


def build_patch_graph(patch, medial_weight):
    """Build (and weight) the quad-grid graph for one patch.

    Nodes are valid quads (cells with four valid corners); the node's position
    in patch (i, j) space is the quad centre (qi + 0.5, qj + 0.5). Directed edge
    u -> v carries weight ``base_len * (1 + medial_weight / max(dt_v, 1))`` where
    base_len is 1 (orthogonal) or sqrt(2) (diagonal): steps into shallow,
    near-the-boundary quads cost more, so shortest paths drift to the middle.
    """
    vq = patch.valid_quad_mask.detach().cpu().numpy()
    hq, wq = vq.shape

    # Distance transform of the valid-quad region; pad with a False border so
    # quads at the very edge of the grid are correctly treated as boundary.
    padded = np.pad(vq, 1, mode='constant', constant_values=False)
    dt = distance_transform_edt(padded)[1:-1, 1:-1].astype(np.float32)

    n = int(vq.sum())
    ids = np.full((hq, wq), -1, dtype=np.int64)
    ids[vq] = np.arange(n)
    node_qi, node_qj = (a.astype(np.int64) for a in np.nonzero(vq))

    # Lift quad centres to world xyz for surface-index queries. Centres of valid
    # quads are always interior to a valid quad, so ij_to_zyx never rejects them.
    ij = torch.stack([
        torch.from_numpy(node_qi).to(torch.float32) + 0.5,
        torch.from_numpy(node_qj).to(torch.float32) + 0.5,
    ], dim=-1)
    zyx, _ = patch.ij_to_zyx(ij)
    quad_center_xyz = np.ascontiguousarray(zyx.cpu().numpy()[:, ::-1], dtype=np.float32)

    rows, cols, data = [], [], []
    for di, dj in _DIRS8:
        base = float(np.hypot(di, dj))
        si0, si1 = max(0, -di), hq - max(0, di)
        sj0, sj1 = max(0, -dj), wq - max(0, dj)
        if si1 <= si0 or sj1 <= sj0:
            continue
        src = vq[si0:si1, sj0:sj1]
        dst = vq[si0 + di:si1 + di, sj0 + dj:sj1 + dj]
        both = src & dst
        if not both.any():
            continue
        su = ids[si0:si1, sj0:sj1][both]
        sv = ids[si0 + di:si1 + di, sj0 + dj:sj1 + dj][both]
        dtv = dt[si0 + di:si1 + di, sj0 + dj:sj1 + dj][both]
        w = base * (1.0 + medial_weight / np.maximum(dtv, 1.0))
        rows.append(su)
        cols.append(sv)
        data.append(w.astype(np.float64))

    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.concatenate(data)
    else:
        rows = cols = np.empty(0, dtype=np.int64)
        data = np.empty(0, dtype=np.float64)
    csr = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    return PatchGraph(patch, vq, dt, ids, node_qi, node_qj, quad_center_xyz, csr, n)


def patch_overlap_to(index, src_graph, dst_idx, tolerance):
    """Which of ``src_graph``'s quads lie on surface index ``dst_idx`` (within tolerance).

    Returns ``(overlap_mask, bridge_src_locals, bridge_dst_ij)`` where
    ``overlap_mask`` is a (Hq, Wq) bool over src quads, ``bridge_src_locals`` is
    an array of the source node ids that hit dst, and ``bridge_dst_ij`` is the
    (K, 2) array of matching (i, j) locations on dst (used to stitch the two
    grids together).
    """
    offsets, surf_idx, _distance, ij = index.locate_all_xyz_batch(src_graph.quad_center_xyz, tolerance)
    # quad_center_xyz is one point per node in node order, so a hit's query-point
    # index *is* its source node id; locate_all yields at most one hit per surface
    # per point, so each node appears at most once among the dst hits.
    node_of_hit = np.repeat(np.arange(offsets.size - 1), np.diff(offsets))
    mask = surf_idx == dst_idx
    bridge_src = node_of_hit[mask].astype(np.int64)
    bridge_dst_ij = ij[mask].astype(np.float64).reshape(-1, 2)  # (grid_y, grid_x) per hit
    overlap = np.zeros(src_graph.valid_quad.shape, dtype=bool)
    overlap[src_graph.node_qi[bridge_src], src_graph.node_qj[bridge_src]] = True
    return overlap, bridge_src, bridge_dst_ij


def _reconstruct(pred, start, end):
    """Walk scipy's predecessor array back from ``end`` to ``start``."""
    if end == start:
        return [start]
    path = [end]
    cur = end
    while True:
        p = int(pred[cur])
        if p < 0:  # -9999 == no predecessor (unreachable)
            return None
        path.append(p)
        if p == start:
            break
        cur = p
    path.reverse()
    return path


def resample_path(nodes, spacing):
    """Resample a node polyline at ``spacing`` (grid-vertex units) of arc length.

    ``nodes`` is a list of (patch_label, i, j) at quad centres. Consecutive
    same-patch nodes are joined by a real segment (length 1 or sqrt2); a change
    of patch_label is a bridge of ~zero length (the two quad centres coincide in
    3D) and contributes no arc length. Returns a list of (patch_label, i, j)
    samples including both endpoints.
    """
    n = len(nodes)
    if n <= 1:
        return list(nodes)

    seg = [0.0] * (n - 1)
    for k in range(n - 1):
        a, b = nodes[k], nodes[k + 1]
        if a[0] == b[0]:
            seg[k] = float(np.hypot(b[1] - a[1], b[2] - a[2]))
    cum = [0.0]
    for s in seg:
        cum.append(cum[-1] + s)
    total = cum[-1]
    if total <= 0:
        return [nodes[0]]

    positions = []
    p = 0.0
    while p < total - 1e-9:
        positions.append(p)
        p += spacing
    positions.append(total)

    out = []
    for pos in positions:
        k = min(max(bisect.bisect_right(cum, pos) - 1, 0), n - 2)
        if seg[k] <= 0:  # landed on a zero-length bridge; snap to a real segment
            kk = k
            while kk < n - 1 and seg[kk] <= 0:
                kk += 1
            if kk > n - 2:
                kk = k
                while kk >= 0 and seg[kk] <= 0:
                    kk -= 1
            if kk < 0 or kk > n - 2:
                continue
            k = kk
        pos_c = min(max(pos, cum[k]), cum[k + 1])
        t = (pos_c - cum[k]) / seg[k]
        a, b = nodes[k], nodes[k + 1]
        out.append((a[0], a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t))
    return out


def lift_samples_to_xyz(samples, ga, gb):
    """Lift (patch_label, i, j) samples to world xyz via each patch's ij_to_zyx."""
    out = [None] * len(samples)
    for label, g in ((0, ga), (1, gb)):
        idxs = [k for k, s in enumerate(samples) if s[0] == label]
        if not idxs:
            continue
        ij = torch.tensor([[samples[k][1], samples[k][2]] for k in idxs], dtype=torch.float32)
        zyx, mask = g.patch.ij_to_zyx(ij)
        zyx = zyx.cpu().numpy()
        mask = mask.cpu().numpy()
        for n_, k in enumerate(idxs):
            if mask[n_]:
                z, y, x = zyx[n_]
                out[k] = [float(x), float(y), float(z)]  # point collections store xyz
    return [p for p in out if p is not None]


def _rasterize_targets(shared, g, dst_ij):
    """Mark the quads of ``g`` that the fractional (i, j) projections fall in."""
    if len(dst_ij) == 0:
        return
    arr = np.floor(np.asarray(dst_ij, dtype=np.float64)).astype(np.int64)
    ii, jj = arr[:, 0], arr[:, 1]
    h, w = g.valid_quad.shape
    ok = (ii >= 0) & (ii < h) & (jj >= 0) & (jj < w)
    ii, jj = ii[ok], jj[ok]
    valid = g.valid_quad[ii, jj]
    shared[ii[valid], jj[valid]] = True


def _pick_margin_endpoint(only_mask, over_dist, dt, margin, reachable=None):
    """Pick a quad in the non-overlapping region, preferably ``margin`` in.

    Among quads that are non-overlapping (``only_mask``) and -- when given --
    reachable: if any lie at least ``margin`` grid vertices from the overlap
    (``over_dist``), choose the one *closest* to the margin (we want a fixed
    margin in, not the farthest-possible point), breaking ties toward the more
    central (higher ``dt``) quad. Otherwise fall back to an arbitrary
    non-overlapping quad rather than giving up. Returns the chosen (qi, qj), or
    ``None`` only if there is no (reachable) non-overlapping quad at all.
    """
    avail = only_mask & reachable if reachable is not None else only_mask
    if not avail.any():
        return None
    cand = avail & (over_dist >= margin)
    if cand.any():
        score = np.where(cand, over_dist - 1e-3 * dt, np.inf)
        qi, qj = np.unravel_index(int(np.argmin(score)), score.shape)
    else:
        # No point is `margin` into the region; take an arbitrary one.
        qi, qj = np.unravel_index(int(np.argmax(avail)), avail.shape)
    return int(qi), int(qj)


def connect_pair(ga, gb, index, a_idx, b_idx, tolerance, medial_weight, spacing, margin):
    """Build the connecting xyz point sequence for one overlapping pair.

    ``a_idx``/``b_idx`` are the two patches' integer surface indices (into
    ``index.surface_ids()``). The two endpoints are placed ``margin`` grid
    vertices into each patch's non-overlapping region (measured from the
    overlap), as close to that margin as possible rather than as deep as
    possible; when no point is that far in, an arbitrary non-overlapping point is
    used instead. Returns the list of xyz points, or ``None`` if no valid path
    exists (the patches don't share a bridge, one is entirely contained in the
    other, or the endpoints are unreachable through valid quads).
    """
    overlap_a, a_src, a_dst_ij = patch_overlap_to(index, ga, b_idx, tolerance)
    overlap_b, b_src, b_dst_ij = patch_overlap_to(index, gb, a_idx, tolerance)

    na, nb = ga.n, gb.n
    brows, bcols, bdata = [], [], []

    def add_bridge(u_global, v_global, dst_dt):
        brows.append(u_global)
        bcols.append(v_global)
        bdata.append(1.0 * (1.0 + medial_weight / max(float(dst_dt), 1.0)))

    # A-quads that hit B -> the B-quad they project onto (and the reverse edge).
    for local, ij in zip(a_src, a_dst_ij):
        bi, bj = int(np.floor(ij[0])), int(np.floor(ij[1]))
        if 0 <= bi < gb.valid_quad.shape[0] and 0 <= bj < gb.valid_quad.shape[1] and gb.valid_quad[bi, bj]:
            v = int(gb.ids[bi, bj])
            add_bridge(local, na + v, gb.dt[bi, bj])
            add_bridge(na + v, local, ga.dt[ga.node_qi[local], ga.node_qj[local]])
    # B-quads that hit A -> the A-quad they project onto.
    for local, ij in zip(b_src, b_dst_ij):
        ai, aj = int(np.floor(ij[0])), int(np.floor(ij[1]))
        if 0 <= ai < ga.valid_quad.shape[0] and 0 <= aj < ga.valid_quad.shape[1] and ga.valid_quad[ai, aj]:
            u = int(ga.ids[ai, aj])
            add_bridge(na + local, u, ga.dt[ai, aj])
            add_bridge(u, na + local, gb.dt[gb.node_qi[local], gb.node_qj[local]])

    if not brows:
        return None  # no overlap bridge => the two grids can't be joined

    # The shared region as seen from each side: a quad is "shared" if its centre
    # lands on the other patch, or the other patch projects a quad onto it.
    shared_a = overlap_a.copy()
    _rasterize_targets(shared_a, ga, b_dst_ij)
    shared_b = overlap_b.copy()
    _rasterize_targets(shared_b, gb, a_dst_ij)

    a_only = ga.valid_quad & ~shared_a
    b_only = gb.valid_quad & ~shared_b
    if not a_only.any() or not b_only.any() or not shared_a.any() or not shared_b.any():
        return None  # no non-overlapping region (or no overlap) on one side

    # Distance, in grid vertices, from the overlap into each patch's own region.
    over_a = distance_transform_edt(~shared_a).astype(np.float32)
    over_b = distance_transform_edt(~shared_b).astype(np.float32)

    # Start: a point ~margin vertices into A's non-overlapping region.
    start_qij = _pick_margin_endpoint(a_only, over_a, ga.dt, margin)
    if start_qij is None:
        return None
    start = int(ga.ids[start_qij])

    combined = sp.bmat([[ga.csr, None], [None, gb.csr]], format='csr')
    bridge_mat = sp.coo_matrix((bdata, (brows, bcols)), shape=(na + nb, na + nb)).tocsr()
    combined = (combined + bridge_mat).tocsr()

    dist, pred = csgraph_dijkstra(combined, directed=True, indices=start, return_predecessors=True)

    # End: same margin into B's non-overlapping region, reachable from the start.
    reachable_b = np.zeros(gb.valid_quad.shape, dtype=bool)
    reachable_b[gb.valid_quad] = np.isfinite(dist[na + gb.ids[gb.valid_quad]])
    end_qij = _pick_margin_endpoint(b_only, over_b, gb.dt, margin, reachable=reachable_b)
    if end_qij is None:
        return None
    end = int(na + gb.ids[end_qij])

    path = _reconstruct(pred, start, end)
    if path is None:
        return None

    nodes = []
    for gnode in path:
        if gnode < na:
            nodes.append((0, ga.node_qi[gnode] + 0.5, ga.node_qj[gnode] + 0.5))
        else:
            local = gnode - na
            nodes.append((1, gb.node_qi[local] + 0.5, gb.node_qj[local] + 0.5))

    samples = resample_path(nodes, spacing)
    return lift_samples_to_xyz(samples, ga, gb)


@click.command(help=__doc__)
@click.argument('patches_path', type=click.Path(exists=True, file_okay=False))
@click.option('--tolerance', type=float, default=2.0, show_default=True,
              help='max distance (voxels) from a vertex to another patch surface to count as overlapping')
@click.option('--min-overlap-points', type=int, default=16, show_default=True,
              help='only report a pair when BOTH patches have at least this many vertices '
                   'landing on the other (min of the two directed counts); full-resolution '
                   'vertex units, invariant to --query-stride')
@click.option('--min-diff-points', type=int, default=64, show_default=True,
              help='each patch must have at least this many vertices NOT present on the other '
                   '(min of the two one-sided differences); ensures both patches contribute '
                   'unique area and filters out near-duplicate patches; full-resolution '
                   'vertex units, invariant to --query-stride')
@click.option('--query-stride', type=int, default=1, show_default=True,
              help='subsample every Nth valid vertex when querying (speeds up large patches)')
@click.option('--index-stride', type=int, default=1, show_default=True,
              help='sampling_stride passed to the index (>1 trades accuracy for speed/memory)')
@click.option('--limit-patches', type=int, default=None,
              help='load at most this many patches (for quick iteration)')
@click.option('--limit-pcls', type=int, default=None,
              help='synthesise connecting point collections for at most (roughly) this many '
                   'pairs -- the highest-overlap ones; all pairs are still written to the JSONL, '
                   'but pairs beyond this count get no collection (slightly fewer than this may '
                   'actually be produced, since some of those pairs yield no valid path)')
@click.option('--pairs-output', '-o', type=click.Path(dir_okay=False), default='overlap_pairs.jsonl',
              show_default=True, help='write overlapping pairs to this JSONL file (one pair per line)')
@click.option('--collection-output', '-c', type=click.Path(dir_okay=False), default='overlap_connections.json',
              show_default=True, help='write the synthesised connecting point collection to this JSON file')
@click.option('--point-spacing', type=float, default=4.0, show_default=True,
              help='spacing between consecutive points along each connecting path, in grid vertex units')
@click.option('--medial-weight', type=float, default=4.0, show_default=True,
              help='how strongly paths are pulled toward the perpendicular middle of each patch '
                   '(0 = shortest path; larger keeps the path away from valid-region boundaries)')
@click.option('--endpoint-margin', type=float, default=13.0, show_default=True,
              help='prefer placing each path endpoint this many grid vertices into its patch\'s '
                   'non-overlapping region (measured from the overlap), if no point is that far in, an '
                   'arbitrary non-overlapping point is used instead')
@click.option('--workers', '-j', type=int, default=None,
              help='worker threads for the querying and connecting-paths steps (default: CPU count)')
def main(patches_path, tolerance, min_overlap_points, min_diff_points, query_stride, index_stride, limit_patches,
         limit_pcls, pairs_output, collection_output, point_spacing, medial_weight, endpoint_margin, workers):
    patches = load_patches(patches_path, limit=limit_patches)
    index = build_surface_index(patches, bbox_padding=tolerance, sampling_stride=index_stride)
    surface_ids = index.surface_ids()                       # idx -> patch id
    id_to_idx = {sid: i for i, sid in enumerate(surface_ids)}
    directed, n_query, n_full = find_overlaps(patches, index, tolerance, query_stride, id_to_idx, surface_ids,
                                              workers=workers)

    # Scale a patch's strided probe counts back to full-resolution-equivalent
    # vertices: count_xy is x's own (strided) sampling, so it scales by x's own
    # full/strided ratio. This makes the reported counts -- and therefore the
    # thresholds below -- invariant to --query-stride (scale == 1 at stride 1).
    def scale_of(pid):
        nq = n_query.get(pid, 0)
        return (n_full.get(pid, 0) / nq) if nq else 0.0

    # Pair the two directed measurements for each unordered patch pair. For pair
    # {a, b}: count_ab = a's (rescaled) vertices on b, count_ba = b's on a.
    #   overlap = min(count_ab, count_ba)             -- mutual overlap region
    #   only_a = n_a - count_ab, only_b = n_b - count_ba  -- one-sided differences
    # We want BOTH one-sided differences large (each patch holds unique area), so
    # the filter is on min(only_a, only_b), not their sum.
    pair_keys = {((a, b) if a <= b else (b, a)) for (a, b) in directed}

    rows = []
    for a, b in pair_keys:
        info_ab, info_ba = directed.get((a, b)), directed.get((b, a))
        n_hits = (info_ab['count'] if info_ab else 0) + (info_ba['count'] if info_ba else 0)
        dist_sum = (info_ab['dist_sum'] if info_ab else 0.0) + (info_ba['dist_sum'] if info_ba else 0.0)
        # Rescale directed hit counts to full-resolution-equivalent vertices.
        count_ab = (info_ab['count'] if info_ab else 0) * scale_of(a)
        count_ba = (info_ba['count'] if info_ba else 0) * scale_of(b)
        n_a, n_b = n_full.get(a, 0), n_full.get(b, 0)
        rows.append({
            'a': a, 'b': b,
            'count_ab': count_ab, 'count_ba': count_ba,
            'n_a': n_a, 'n_b': n_b,
            'overlap': min(count_ab, count_ba),
            'only_a': n_a - count_ab,
            'only_b': n_b - count_ba,
            'min_diff': min(n_a - count_ab, n_b - count_ba),
            'mean_dist': dist_sum / n_hits if n_hits else float('nan'),
        })

    # Lazily-built, reused per-patch quad graphs (one patch appears in many
    # pairs). Guarded for concurrent access; each patch is built at most once,
    # and different patches can build concurrently.
    graph_cache = {}
    cache_lock = threading.Lock()
    build_locks = {}

    def get_graph(pid):
        with cache_lock:
            g = graph_cache.get(pid)
            if g is not None:
                return g
            lock = build_locks.setdefault(pid, threading.Lock())
        with lock:
            g = graph_cache.get(pid)
            if g is None:
                g = build_patch_graph(patches[pid], medial_weight)
                with cache_lock:
                    graph_cache[pid] = g
            return g

    def connect_row(r):
        try:
            return connect_pair(
                get_graph(r['a']), get_graph(r['b']), index,
                id_to_idx[r['a']], id_to_idx[r['b']], tolerance, medial_weight, point_spacing, endpoint_margin,
            )
        except Exception as e:  # never let one bad pair abort the whole run
            print(f"Warning: failed to connect {r['a']} <-> {r['b']}: {e}")
            return None

    # Highest-overlap pairs first, so collection ids end up sorted that way too.
    pairs = [r for r in sorted(rows, key=lambda r: -r['overlap'])
             if r['overlap'] >= min_overlap_points and r['min_diff'] >= min_diff_points]

    # Fan the (heavy) path finding out across threads -- the dominant work
    # (distance transforms, Dijkstra, numpy/torch) releases the GIL -- then do
    # the lightweight bookkeeping serially below so output stays deterministic.
    # Only synthesise connecting collections for the highest-overlap pairs (pairs
    # is sorted by descending overlap). Every pair is still written to the JSONL;
    # those beyond --limit-pcls simply get no collection. "Roughly" because some
    # of these top pairs may still yield no valid path.
    n_connect = len(pairs) if limit_pcls is None else min(limit_pcls, len(pairs))
    path_points = [None] * len(pairs)
    with ThreadPoolExecutor(max_workers=workers or (os.cpu_count() or 4)) as pool:
        futures = {pool.submit(connect_row, r): i for i, r in enumerate(pairs[:n_connect])}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='connecting pairs'):
            path_points[futures[fut]] = fut.result()

    collections = {}      # collection_id (str) -> collection dict
    next_collection_id = 0
    with open(pairs_output, 'w') as f:
        for r, points_xyz in zip(pairs, path_points):
            collection_id = None
            if points_xyz is not None and len(points_xyz) >= 2:
                collection_id = next_collection_id
                next_collection_id += 1
                hue = (collection_id * 0.6180339887) % 1.0
                color = list(colorsys.hsv_to_rgb(hue, 0.65, 0.95))
                points = {
                    str(pid): {'p': p, 'creation_time': 0, 'wind_a': 0}
                    for pid, p in enumerate(points_xyz)
                }
                collections[str(collection_id)] = {
                    'name': f"between_patches__{r['a']}__{r['b']}",
                    'points': points,
                    'metadata': {'winding_is_absolute': False},
                    'color': color,
                }

            f.write(json.dumps({**r, 'tolerance': tolerance, 'collection_id': collection_id}) + '\n')

    reported = len(pairs)
    print(f'wrote {reported} overlapping pairs to {pairs_output} '
          f'(overlap >= {min_overlap_points} pts and each-side diff >= {min_diff_points} pts), '
          f'out of {len(rows)} candidate pairs across {len(patches)} patches')

    with open(collection_output, 'w') as cf:
        json.dump({
            'vc_pointcollections_json_version': '1',
            'collections': collections,
        }, cf)
    total_points = sum(len(c['points']) for c in collections.values())
    limited = reported - n_connect          # never attempted, capped by --limit-pcls
    no_path = n_connect - len(collections)  # attempted but produced no valid path
    print(f'wrote {len(collections)} connecting point sequences ({total_points} points) '
          f'to {collection_output}; of {reported} reported pairs, {no_path} were skipped '
          f'(no valid path)' + (f' and {limited} beyond --limit-pcls' if limited else ''))


if __name__ == '__main__':
    main()
