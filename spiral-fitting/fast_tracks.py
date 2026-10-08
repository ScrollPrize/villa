"""Interjoint track extraction without networkx, with the same output.

``extract_surface_tracks.get_skeleton_tracks`` (``path_mode = 'interjoint'``) turns every
skeleton into tracks as follows:

    graph = nx.Graph(); graph.add_edges_from(skeleton.edges)
    for path in extract_inter_branch_paths(graph):          # maximal degree-2 chains
        if len(path) < 10: continue
        tracks.append((skeleton.vertices[path].astype(np.int64) + offset).astype(np.int32))

This module computes the same list, in the same order and with the same dtypes, without
building networkx graphs. ``extract_inter_branch_paths`` walks the graph in networkx's
iteration order, so this module reproduces that order exactly:

  nodes      ``graph.nodes()`` yields nodes in order of first appearance in the edge list, as
             ``add_edges_from`` inserts them (u, then v, edge by edge).
  neighbors  ``graph.neighbors(n)`` yields n's neighbours in the order their edge was first
             added. A repeated edge (in either orientation) keeps its first position; a
             self-loop lists n among its own neighbours.
  degree     ``graph.degree(n)`` is the number of distinct neighbours, plus one for a
             self-loop (a self-loop counts 2).
  walk       for every node of degree != 2, in node order, and every neighbour, in neighbour
             order: skip the edge if it was visited; else follow degree-2 nodes, always to the
             first neighbour that is not the previous node, marking edges visited, until a node
             of degree != 2 or an already visited edge. Components whose nodes all have
             degree 2 (pure cycles) produce no path.

Three engines compute the same paths:

  'numba'   the walk above, compiled (O(edges), no Python objects per vertex). Default when
            numba is importable. Compiled on first use and cached on disk (``cache=True``).
  'numpy'   vectorized: chains are found as the connected components of the degree-2 nodes
            (``scipy.sparse.csgraph``), each chain is kept in the direction the walk takes it
            (the end whose (node order, neighbour order) comes first), and its vertices are
            ordered by a breadth-first search from that end. Exact for every ``nx.Graph`` (a
            graph without multi-edges): there, a walk never meets a visited edge or a dead end
            mid-chain. Default when numba is missing.
  'python'  the numba kernel run as plain Python. Slow; a readable specification for tests.

The numba kernel works on the packed layout directly (per-skeleton local edge ids and vertex /
edge offsets), checks every edge, and uses int32 index arrays when the sizes allow.

All skeletons of one slab are processed in one call, as the disjoint union of their graphs:
the union's node order is the skeletons' node orders one after the other, so its paths come
out skeleton by skeleton, in the order ``get_skeleton_tracks`` iterates
``skeletons.values()``.

API
  extract_inter_branch_paths_fast(edges)    the paths of ``extract_inter_branch_paths`` for
                                            ``nx.Graph(edges)``, as arrays of node ids
  inter_branch_paths(eu, ev, n_nodes)       the same for one graph, as flat node ids and offsets
  tracks_from_skeletons(skeletons, offset)  the tracks of ``get_skeleton_tracks`` from its
                                            skeletons (objects with ``.vertices`` and
                                            ``.edges``, in dict order)
  tracks_from_packed(v_off, e_off, vertices, edges, offset)
                                            the same from ``brook.PackedSkeletons`` arrays
  warmup()                                  compile the numba kernel now

Tracks are independent C-contiguous int32 (L, 3) arrays, like those of ``get_skeleton_tracks``
(``copy=False`` returns views into one array: same values, same pickles, less time).
Edges must be non-negative vertex indices below the skeleton's vertex count; other inputs raise
``ValueError`` (the networkx version would raise ``IndexError`` or wrap negative indices).
"""
from __future__ import annotations

import threading
from collections.abc import Iterable

import numpy as np

__all__ = ['MIN_TRACK_VERTICES', 'ENGINES', 'default_engine', 'warmup', 'extract_inter_branch_paths_fast',
           'inter_branch_paths', 'tracks_from_skeletons', 'tracks_from_packed']

MIN_TRACK_VERTICES = 10          # get_skeleton_tracks drops paths of fewer vertices

# numba is optional: `uv sync` installs it on Linux x86-64, and the 'numpy' engine covers
# environments set up without it.
try:
    import numba as _numba
except ImportError:              # pragma: no cover - depends on the environment
    _numba = None

ENGINES = ('numba', 'numpy', 'python')


def default_engine():
    """``'numba'`` when numba is importable, else ``'numpy'``."""
    return 'numba' if _numba is not None else 'numpy'


# ------------------------------------------------------------------------------------------------
# The walk (numba kernel; also the plain-Python specification)

def _walk(edges, e_off, v_off, min_len, proto):
    """Paths of extract_inter_branch_paths(nx.Graph(edges of skeleton k)), skeleton after skeleton,
    with >= min_len nodes, as global vertex ids.

    Skeleton k has edges[e_off[k]:e_off[k + 1]] (local vertex ids, < v_off[k + 1] - v_off[k]) and
    global ids v_off[k] + local id. proto: an empty array whose dtype is used for all index arrays
    (int32 when the sizes allow). Returns (flat node ids, offsets): path i is
    flat[offsets[i]:offsets[i + 1]].
    """
    K = e_off.shape[0] - 1
    n = v_off[K]
    it = proto.dtype
    # pass 1, edges: node order (first appearance, u then v, as add_edges_from inserts them) and
    # the number of adjacency insertions per node
    seen = np.zeros(n, np.bool_)
    order = np.empty(n, it)
    raw_start = np.zeros(n + 1, it)
    n_order = 0
    for k in range(K):
        b = v_off[k]
        nv = v_off[k + 1] - b
        for i in range(e_off[k], e_off[k + 1]):
            a0 = edges[i, 0]
            a1 = edges[i, 1]
            if a0 < 0 or a1 < 0 or a0 >= nv or a1 >= nv:
                raise ValueError('an edge refers to a vertex outside its skeleton')
            u = b + a0
            v = b + a1
            if not seen[u]:
                seen[u] = True
                order[n_order] = u
                n_order += 1
            if not seen[v]:
                seen[v] = True
                order[n_order] = v
                n_order += 1
            raw_start[u + 1] += 1
            if u != v:
                raw_start[v + 1] += 1
    for x in range(n):
        raw_start[x + 1] += raw_start[x]
    # pass 2, edges: every insertion in edge order (_adj[u][v] = ..., then _adj[v][u] = ...)
    fill = raw_start[:n].copy()
    raw = np.empty(raw_start[n], it)
    for k in range(K):
        b = v_off[k]
        for i in range(e_off[k], e_off[k + 1]):
            u = b + edges[i, 0]
            v = b + edges[i, 1]
            raw[fill[u]] = v
            fill[u] += 1
            if u != v:
                raw[fill[v]] = u
                fill[v] += 1
    # pass 3, nodes: dict semantics, a repeated neighbour keeps its first position; raw[p] becomes
    # the slot of insertion p in the deduplicated adjacency nbr[start[x]:start[x + 1]]
    mark = np.full(n, -1, it)                    # mark[y] = y's slot in the current node's list
    start = np.empty(n + 1, it)
    nbr = np.empty(raw_start[n], it)
    deg = np.empty(n, it)
    w = 0
    for x in range(n):
        s = w
        start[x] = s
        for p in range(raw_start[x], raw_start[x + 1]):
            y = raw[p]
            if mark[y] >= s:
                raw[p] = mark[y]
            else:
                mark[y] = w
                nbr[w] = y
                raw[p] = w
                w += 1
        deg[x] = (w - s) + (1 if mark[x] >= s else 0)     # graph.degree: a self-loop counts 2
    start[n] = w
    # pass 4, edges (replayed): rev[slot of x -> y] = slot of y -> x
    rev = np.empty(w, it)
    fill[:] = raw_start[:n]
    for k in range(K):
        b = v_off[k]
        for i in range(e_off[k], e_off[k + 1]):
            u = b + edges[i, 0]
            v = b + edges[i, 1]
            p = fill[u]
            fill[u] += 1
            q = p
            if u != v:
                q = fill[v]
                fill[v] += 1
            sp = raw[p]
            sq = raw[q]
            rev[sp] = sq
            rev[sq] = sp
    # the walk of extract_inter_branch_paths
    visited = np.zeros(w, np.bool_)
    out = np.empty(2 * w + 2, it)
    offs = np.empty(w + 2, it)
    offs[0] = 0
    n_paths = 0
    wp = 0
    for j in range(n_order):
        x = order[j]
        if deg[x] == 2:
            continue
        for p in range(start[x], start[x + 1]):
            if visited[p]:                       # edge in visited_edges
                continue
            base = wp
            out[wp] = x
            out[wp + 1] = nbr[p]
            wp += 2
            visited[p] = True
            visited[rev[p]] = True
            prev = x
            cur = nbr[p]
            while deg[cur] == 2:
                found = -1                       # first neighbour != path[-2]
                for q in range(start[cur], start[cur + 1]):
                    if nbr[q] != prev:
                        found = q
                        break
                if found < 0:
                    break
                if visited[found]:
                    break
                nxt = nbr[found]
                out[wp] = nxt
                wp += 1
                visited[found] = True
                visited[rev[found]] = True
                prev = cur
                cur = nxt
            if wp - base < min_len:              # the minimum track length
                wp = base
            else:
                n_paths += 1
                offs[n_paths] = wp
    return out[:wp].copy(), offs[:n_paths + 1].copy()


_walk_numba = None
_walk_numba_lock = threading.Lock()


def _get_walk_numba():
    # Track threads may call this concurrently; the lock makes them share one dispatcher (and one
    # compilation) instead of racing to create several.
    global _walk_numba
    if _walk_numba is None:
        with _walk_numba_lock:
            if _walk_numba is None:
                if _numba is None:
                    raise ImportError("engine 'numba' needs numba")
                _walk_numba = _numba.njit(cache=True, nogil=True)(_walk)
    return _walk_numba


def warmup(engine=None):
    """Compile (or load from the cache) the numba kernel now instead of on first use (for uint32,
    int32 and int64 edges)."""
    if (engine or default_engine()) == 'numba':
        off = np.array([0, 1], np.int64)
        for dt in (np.uint32, np.int32, np.int64):
            _get_walk_numba()(np.array([[0, 1]], dt), off, np.array([0, 2], np.int64), 0,
                              np.zeros(0, np.int32))


# ------------------------------------------------------------------------------------------------
# The vectorized engine

def _paths_numpy(eu, ev, n_nodes, min_len):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    E = eu.shape[0]
    empty = (np.zeros(0, np.int64), np.zeros(1, np.int64))
    if E == 0:
        return empty
    if E >= 2 ** 31:
        raise ValueError('too many edges')
    # first appearance of every node in u0 v0 u1 v1 ... (graph.nodes() order)
    flat = np.empty(2 * E, np.int64)
    flat[0::2] = eu
    flat[1::2] = ev
    first = np.full(n_nodes, 2 * E, np.int64)
    np.minimum.at(first, flat, np.arange(2 * E, dtype=np.int64))
    present = first < 2 * E
    # distinct edges, each with the index of its first occurrence (its neighbour position)
    lo, hi = np.minimum(eu, ev), np.maximum(eu, ev)
    code = lo * n_nodes + hi
    ucode, t = np.unique(code, return_index=True)
    a, b = ucode // n_nodes, ucode % n_nodes
    loop = a == b
    deg = np.bincount(a, minlength=n_nodes) + np.bincount(b, minlength=n_nodes)   # a self-loop counts 2
    has_loop = np.zeros(n_nodes, np.bool_)
    has_loop[a[loop]] = True
    critical = present & (deg != 2)
    interior = present & (deg == 2) & ~has_loop
    # half-edges src -> dst (one per self-loop); the walk may start on those leaving a critical node
    nl = ~loop
    src = np.concatenate([a[nl], b[nl], a[loop]])
    dst = np.concatenate([b[nl], a[nl], a[loop]])
    tt = np.concatenate([t[nl], t[nl], t[loop]])
    pair = np.concatenate([np.flatnonzero(nl), np.flatnonzero(nl), np.flatnonzero(loop)])
    st = critical[src]
    src, dst, tt, pair = src[st], dst[st], tt[st], pair[st]
    key = first[src] * E + tt                     # (node order, neighbour order) of the start
    # (i) edges between critical nodes and self-loops: one-edge paths [src, dst]; a critical-
    #     critical edge is walked from the end that comes first
    single = critical[dst]
    s_idx = np.flatnonzero(single)
    o = np.lexsort((key[s_idx], pair[s_idx]))
    s_idx = s_idx[o]
    keep = np.ones(s_idx.size, np.bool_)
    keep[1:] = pair[s_idx[1:]] != pair[s_idx[:-1]]
    s_idx = s_idx[keep]
    # (ii) chains through degree-2 nodes: components of the degree-2 subgraph with two boundary
    #      half-edges each (pure cycles have none); the walk takes the first boundary half-edge
    #      and ends at the other one's source
    c_idx = np.flatnonzero(~single)
    ii = interior[a] & interior[b] & nl
    g = csr_matrix((np.ones(int(ii.sum()), np.int8), (a[ii], b[ii])), shape=(n_nodes, n_nodes))
    _, comp = connected_components(g, directed=False)
    ccomp = comp[dst[c_idx]]
    o = np.lexsort((key[c_idx], ccomp))
    c_idx, ccomp = c_idx[o], ccomp[o]
    if c_idx.size % 2 or np.any(ccomp[0::2] != ccomp[1::2]):
        raise AssertionError('a degree-2 chain without exactly two ends (not a simple graph?)')
    c_keep, c_end = c_idx[0::2], src[c_idx[1::2]]
    csize = np.bincount(comp, weights=interior, minlength=comp.max() + 1).astype(np.int64)
    c_len = csize[comp[dst[c_keep]]] + 2
    # all kept paths, in walk order, filtered by length
    p_src = np.concatenate([src[s_idx], src[c_keep]])
    p_dst = np.concatenate([dst[s_idx], dst[c_keep]])
    p_end = np.concatenate([dst[s_idx], c_end])
    p_len = np.concatenate([np.full(s_idx.size, 2, np.int64), c_len])
    p_key = np.concatenate([key[s_idx], key[c_keep]])
    p_chain = np.concatenate([np.zeros(s_idx.size, np.bool_), np.ones(c_keep.size, np.bool_)])
    f = p_len >= min_len
    o = np.argsort(p_key[f], kind='stable')
    p_src, p_dst, p_end, p_len, p_chain = (x[f][o] for x in (p_src, p_dst, p_end, p_len, p_chain))
    offs = np.zeros(p_len.size + 1, np.int64)
    np.cumsum(p_len, out=offs[1:])
    out = np.empty(offs[-1], np.int64)
    out[offs[:-1]] = p_src
    out[offs[1:] - 1] = p_end
    if p_chain.any():
        # chain vertices: breadth-first from a super source joined to every kept chain's first
        # vertex; within a chain (a path) that is the walk order. Group by chain, stably.
        heads = p_dst[p_chain]
        S = n_nodes
        rows = np.concatenate([np.full(heads.size, S, np.int64), a[ii], b[ii]])
        cols = np.concatenate([heads, b[ii], a[ii]])
        gd = csr_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(n_nodes + 1, n_nodes + 1))
        bfs = breadth_first_order(gd, S, directed=True, return_predecessors=False)[1:]
        rank = np.full(comp.max() + 1, -1, np.int64)
        rank[comp[heads]] = np.arange(heads.size)
        r = rank[comp[bfs]]
        bfs = bfs[r >= 0]
        mids = bfs[np.argsort(r[r >= 0], kind='stable')]
        is_mid = np.ones(out.size, np.bool_)
        is_mid[offs[:-1]] = False
        is_mid[offs[1:] - 1] = False
        if mids.size != int(is_mid.sum()):
            raise AssertionError('chain lengths disagree')
        out[is_mid] = mids
    return out, offs


# ------------------------------------------------------------------------------------------------
# Paths

_KERNEL_DTYPES = (np.dtype(np.uint32), np.dtype(np.int32), np.dtype(np.int64))


def _as_edge_array(edges):
    e = np.asarray(edges)
    if e.size == 0:
        return np.zeros((0, 2), np.int64), e.dtype
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError(f'edges must have shape (E, 2), not {e.shape}')
    if not np.issubdtype(e.dtype, np.integer):
        raise ValueError(f'edges must be integers, not {e.dtype}')
    return e, e.dtype


def _paths(edges, e_off, v_off, min_len, engine):
    """(flat global node ids, offsets) of the paths of every skeleton, skeleton after skeleton.

    edges: (E, 2) integer local ids; skeleton k: edges[e_off[k]:e_off[k+1]], ids < v_off[k+1] - v_off[k]."""
    engine = engine or default_engine()
    e_off = np.ascontiguousarray(e_off, np.int64)
    v_off = np.ascontiguousarray(v_off, np.int64)
    if e_off.shape != v_off.shape or e_off.ndim != 1 or e_off.size < 1:
        raise ValueError('e_off and v_off must be 1-D with one entry per skeleton plus one')
    if edges.dtype.kind not in 'iu':
        raise ValueError(f'edges must be integers, not {edges.dtype}')
    if edges.dtype not in _KERNEL_DTYPES:
        edges = edges.astype(np.int64)
    edges = np.ascontiguousarray(edges)
    n, n_edges = int(v_off[-1]), int(e_off[-1] - e_off[0])
    if engine in ('numba', 'python'):
        proto = np.zeros(0, np.int32 if max(n + 2, 2 * n_edges + 4) < 2 ** 31 - 1 else np.int64)
        walk = _get_walk_numba() if engine == 'numba' else _walk
        return walk(edges, e_off, v_off, int(min_len), proto)
    if engine == 'numpy':
        e = edges[e_off[0]:e_off[-1]]
        ne, nv = np.diff(e_off), np.diff(v_off)
        if len(e) and ((e.dtype.kind == 'i' and e.min() < 0)
                       or (e.max(axis=1).astype(np.int64) >= np.repeat(nv, ne)).any()):
            raise ValueError('an edge refers to a vertex outside its skeleton')
        base = np.repeat(v_off[:-1], ne)
        return _paths_numpy(e[:, 0].astype(np.int64) + base, e[:, 1].astype(np.int64) + base, n,
                            int(min_len))
    raise ValueError(f'unknown engine {engine!r}; one of {ENGINES}')


def inter_branch_paths(eu, ev, n_nodes, min_len=0, engine=None):
    """(flat node ids, offsets) of the interjoint paths of one graph, with >= min_len nodes.

    eu, ev: integer node ids in [0, n_nodes)."""
    edges = np.stack([np.asarray(eu, np.int64), np.asarray(ev, np.int64)], axis=1)
    return _paths(edges, np.array([0, len(edges)]), np.array([0, int(n_nodes)]), min_len, engine)


def extract_inter_branch_paths_fast(edges, engine=None, min_len=0):
    """``extract_inter_branch_paths(G)`` for ``G = nx.Graph(); G.add_edges_from(edges)``.

    edges: (E, 2) integers (any values; ids are compacted internally). Returns a list of 1-D
    arrays of node ids, in the edges' dtype, equal element by element to the networkx paths."""
    e, dtype = _as_edge_array(edges)
    if e.shape[0] == 0:
        return []
    flat = e.reshape(-1)
    lo, hi = int(flat.min()), int(flat.max())
    if lo < 0 or hi >= 4 * flat.size + 1024:            # compact sparse or negative ids
        ids, inv = np.unique(flat, return_inverse=True)
        nodes, offs = _paths(inv.reshape(-1, 2), np.array([0, len(e)]), np.array([0, ids.size]),
                             min_len, engine)
        nodes = ids[nodes]
    else:
        nodes, offs = _paths(e, np.array([0, len(e)]), np.array([0, hi + 1]), min_len, engine)
    nodes = nodes.astype(dtype, copy=False)
    return [nodes[i:j] for i, j in zip(offs[:-1].tolist(), offs[1:].tolist())]


# ------------------------------------------------------------------------------------------------
# Tracks

def _tracks(vertices, nodes, offs, offset_zyx, copy):
    offset = np.asarray(offset_zyx, dtype=np.int64)
    coords = (vertices[nodes].astype(np.int64) + offset).astype(np.int32)
    b = offs.tolist()
    if copy:
        return [coords[i:j].copy() for i, j in zip(b[:-1], b[1:])]
    return [coords[i:j] for i, j in zip(b[:-1], b[1:])]


def tracks_from_packed(v_off, e_off, vertices, edges, offset_zyx, min_len=MIN_TRACK_VERTICES, engine=None,
                       copy=True):
    """The tracks of ``get_skeleton_tracks`` for the skeletons of ``brook.PackedSkeletons`` arrays.

    Skeleton k has vertices[v_off[k]:v_off[k+1]] and edges[e_off[k]:e_off[k+1]] (indices local
    to the skeleton), in the order ``get_skeleton_tracks`` iterates ``skeletons.values()``."""
    e_off = np.asarray(e_off, np.int64)
    if len(e_off) < 2 or e_off[-1] == e_off[0]:
        return []
    nodes, offs = _paths(np.asarray(edges).reshape(-1, 2), e_off, v_off, min_len, engine)
    return _tracks(vertices, nodes, offs, offset_zyx, copy)


def tracks_from_skeletons(skeletons: Iterable, offset_zyx, min_len=MIN_TRACK_VERTICES, engine=None,
                          copy=True):
    """The tracks of ``get_skeleton_tracks`` for its skeletons (e.g. ``skeletonize(...).values()``,
    in order)."""
    sk = list(skeletons)
    if not sk:
        return []
    verts = [np.asarray(s.vertices) for s in sk]
    edges = [np.asarray(s.edges).reshape(-1, 2) for s in sk]
    v_off = np.zeros(len(sk) + 1, np.int64)
    np.cumsum([len(v) for v in verts], out=v_off[1:])
    e_off = np.zeros(len(sk) + 1, np.int64)
    np.cumsum([len(x) for x in edges], out=e_off[1:])
    if e_off[-1] == 0:
        return []
    if len({v.dtype for v in verts}) > 1:          # convert per skeleton, as get_skeleton_tracks does
        verts = [v.astype(np.int64) for v in verts]
    kinds = {x.dtype for x in edges if len(x)}
    e = np.concatenate([x.astype(np.int64) if len(kinds) > 1 else x for x in edges])
    return tracks_from_packed(v_off, e_off, np.concatenate(verts), e, offset_zyx, min_len, engine, copy)
