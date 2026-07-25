# Geodesic strip-path computation for the 'dijkstra' patch-strip sampling mode (see losses.py).
# Deliberately torch-free and project-import-free: losses.py runs these functions in
# ProcessPoolExecutor worker processes (scipy's Dijkstra holds the GIL, so threads can't
# provide background freshness), and keeping the import set to numpy + scipy keeps worker
# startup cheap and CUDA-free. Masks cross the process boundary bit-packed (pack_mask).

import time

import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import dijkstra as _csgraph_dijkstra


def warmup():
    # Submitted at configure time to force worker processes to spawn (and import numpy/scipy)
    # while the trainer is still loading data, before the first step needs them.
    time.sleep(0.1)


def pack_mask(valid_quad_mask):
    return np.packbits(valid_quad_mask.reshape(-1)), valid_quad_mask.shape


def _unpack_mask(packed_mask, shape):
    n = shape[0] * shape[1]
    return np.unpackbits(packed_mask, count=n).astype(bool).reshape(shape)


def build_quad_graph(valid_quad_mask):
    # 8-connected CSR adjacency over the (flattened) quad grid; edges join valid cells with
    # weight 1 (orthogonal) or sqrt(2) (diagonal), so Dijkstra distances are geometric path
    # lengths in quad units.
    h, w = valid_quad_mask.shape
    n = h * w
    flat_idx = np.arange(n).reshape(h, w)
    rows, cols, weights = [], [], []
    for di, dj, weight in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2.)), (1, -1, np.sqrt(2.))):
        src_i = slice(max(0, -di), h - max(0, di))
        src_j = slice(max(0, -dj), w - max(0, dj))
        dst_i = slice(max(0, di), h - max(0, -di))
        dst_j = slice(max(0, dj), w - max(0, -dj))
        edge_valid = valid_quad_mask[src_i, src_j] & valid_quad_mask[dst_i, dst_j]
        rows.append(flat_idx[src_i, src_j][edge_valid])
        cols.append(flat_idx[dst_i, dst_j][edge_valid])
        weights.append(np.full(int(edge_valid.sum()), weight, dtype=np.float32))
    return scipy.sparse.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )


def choose_distant_endpoints(dist_flat, num, rng, candidate_mask_flat=None):
    # Sample `num` 'distant but reachable' endpoint cells from a single-source geodesic
    # distance field: restrict to cells at >= half the max reachable distance (within the
    # candidate set) and sample with probability proportional to distance^2. Returns flat
    # indices, or None if nothing (beyond the source itself) is reachable.
    reachable = np.isfinite(dist_flat) & (dist_flat > 0)
    if candidate_mask_flat is not None:
        candidates = reachable & candidate_mask_flat
        if not candidates.any():
            candidates = reachable
    else:
        candidates = reachable
    candidate_idx = np.flatnonzero(candidates)
    if candidate_idx.size == 0:
        return None
    d = dist_flat[candidate_idx]
    far = d >= 0.5 * d.max()
    candidate_idx, d = candidate_idx[far], d[far]
    p = d ** 2
    return rng.choice(candidate_idx, num, replace=candidate_idx.size < num, p=p / p.sum())


def backtrack_path_ij(predecessors, end_flat, width):
    # Walk the Dijkstra predecessor tree from `end_flat` back to the source; returns the path
    # as an int32 [path_len, 2] ij array ordered source -> end.
    nodes = [int(end_flat)]
    while predecessors[nodes[-1]] >= 0:
        nodes.append(int(predecessors[nodes[-1]]))
    nodes.reverse()
    flat = np.asarray(nodes, dtype=np.int64)
    return np.stack([flat // width, flat % width], axis=-1).astype(np.int32)


def compute_patch_path_pool(valid_quad_mask, num_starts, endpoints_per_start, seed):
    # Path pool for the patch radius/DT losses: `num_starts` uniform-random start cells, one
    # Dijkstra each, `endpoints_per_start` distant endpoints per start. Returns a list of
    # int32 ij path arrays.
    rng = np.random.default_rng(seed)
    width = valid_quad_mask.shape[1]
    valid_ij = np.argwhere(valid_quad_mask)
    graph = build_quad_graph(valid_quad_mask)
    pool = []
    for _ in range(num_starts):
        start_i, start_j = valid_ij[rng.integers(len(valid_ij))]
        dist, predecessors = _csgraph_dijkstra(
            graph, directed=False, indices=int(start_i) * width + int(start_j), return_predecessors=True,
        )
        ends = choose_distant_endpoints(dist, endpoints_per_start, rng)
        if ends is None:  # isolated start cell
            pool.append(np.array([[start_i, start_j]], dtype=np.int32))
        else:
            pool.extend(backtrack_path_ij(predecessors, end, width) for end in ends)
    return pool


def compute_patch_path_pool_packed(packed_mask, shape, num_starts, endpoints_per_start, seed):
    return compute_patch_path_pool(_unpack_mask(packed_mask, shape), num_starts, endpoints_per_start, seed)


def compute_anchor_path_pools(valid_quad_mask, i_q, j_q, paths_per_cone, seed):
    # Path pools for the rel/abs winding losses: one Dijkstra from the annotated cell, then
    # `paths_per_cone` paths per cardinal cone (right/left/down/up by dominant ij direction,
    # mirroring the 4 L-shape primaries), each to a distant endpoint reached as directly as
    # the valid region allows. Returns a list of 4 pools (lists of ij path arrays, every path
    # starting at the anchor). Caller guarantees valid_quad_mask[i_q, j_q].
    rng = np.random.default_rng(seed)
    height, width = valid_quad_mask.shape
    graph = build_quad_graph(valid_quad_mask)
    dist, predecessors = _csgraph_dijkstra(
        graph, directed=False, indices=i_q * width + j_q, return_predecessors=True,
    )
    cell_i, cell_j = np.divmod(np.arange(height * width), width)
    di = cell_i - i_q
    dj = cell_j - j_q
    cone_masks = [dj >= np.abs(di), -dj >= np.abs(di), di >= np.abs(dj), -di >= np.abs(dj)]

    pools = []
    for cone_mask in cone_masks:
        ends = choose_distant_endpoints(dist, paths_per_cone, rng, candidate_mask_flat=cone_mask)
        if ends is None:  # isolated anchor cell
            pools.append([np.array([[i_q, j_q]], dtype=np.int32)])
        else:
            pools.append([backtrack_path_ij(predecessors, end, width) for end in ends])
    return pools


def compute_anchor_path_pools_packed(packed_mask, shape, i_q, j_q, paths_per_cone, seed):
    return compute_anchor_path_pools(_unpack_mask(packed_mask, shape), i_q, j_q, paths_per_cone, seed)
