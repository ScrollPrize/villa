import json
import glob
import os

import numpy as np
import scipy.ndimage
import torch
from tqdm import tqdm

from sample_spiral import get_spiral_yxs, get_theta_and_radii
from tifxyz import load_tifxyz, save_tifxyz, save_combined_tifxyz
from vc3d_fiber_format_adapter import (
    parse_vc3d_fiber_format,
)


def scale_patch(patch, downsample_factor):
    patch.scale *= downsample_factor
    patch.zyxs /= downsample_factor
    patch.area /= downsample_factor ** 2
    patch.release_derived_caches()


def patch_intersects_z_roi(patch, z_begin, z_end):
    zs = patch.valid_zyxs[..., 0]
    if zs.numel() == 0:
        return False
    return bool(((zs >= z_begin) & (zs < z_end)).any().item())


def scale_counts_for_z_range(
    config,
    z_begin,
    z_end,
    reference_z_range_num_slices,
    z_range_scaled_count_keys,
    floors=None,
):
    """Scale per-step sample counts with the z-range, respecting floors.

    Floors exist for losses whose per-sample information is sparse: the
    phase bundle sees ~6 winding gradient sites per pair, so
    volume-proportional scaling starves it on narrow windows (a 300-slice
    session got ~380 pairs from the 12k default and corrected at half
    grad_mag's rate - 2026-07-17 sampling-scale probes).
    """
    num_slices = z_end - z_begin
    scale = num_slices / reference_z_range_num_slices
    for key in z_range_scaled_count_keys:
        floor = 1 if floors is None else int(floors.get(key, 1))
        config[key] = max(floor, round(config[key] * scale))
    return scale, num_slices


SAMPLING_COUNT_FLOORS = {
    'sample_count_dense_spacing_pairs': 8_000,
    'sample_count_dense_spacing_density_extra_pairs': 16_000,
}

# All per-step sample-count defaults are tuned for a 9500-slice z-range;
# scale_counts_for_z_range() scales them relative to this reference.
REFERENCE_Z_RANGE_NUM_SLICES = 9500


def scale_and_split_counts(config, z_begin, z_end, count_keys, world_size=None):
    """Scale per-step counts for the z-range, then split them across ranks.

    The shared scale-then-split sequence used by both the headless CLI
    (fit_spiral.__main__) and the interactive runtime (spiral_runtime).
    The callers pass their own key sets: the CLI keeps its historical
    tuple while the runtime derives keys from the Config catalog's
    scale_with_z fields. ``world_size`` is passed explicitly by callers that
    hold a DistributedContext; None falls back to the installed process
    context. Returns (scale, num_slices, split_divisor) for caller-side
    reporting.
    """
    from ddp_helpers import split_counts_across_ranks

    scale, num_slices = scale_counts_for_z_range(
        config, z_begin, z_end,
        REFERENCE_Z_RANGE_NUM_SLICES, count_keys,
        floors=SAMPLING_COUNT_FLOORS,
    )
    split_divisor = split_counts_across_ranks(
        config, count_keys, world_size=world_size)
    return scale, num_slices, split_divisor


def _decimate_ordered_points_min_spacing(points, min_spacing, return_indices=False,
                                         force_keep=None):
    # force_keep: original indices that must survive regardless of spacing (used to
    # keep cross-fiber link endpoints exact). Keeping a few extra, more closely
    # spaced points is harmless to the strip/winding losses.
    force_keep = set(force_keep or ())
    if min_spacing <= 0 or len(points) <= 1:
        keep = list(range(len(points)))
        return (points, np.asarray(keep, dtype=np.int64)) if return_indices else points

    keep = [0]
    last_kept = points[0]
    for i in range(1, len(points)):
        if i in force_keep or np.linalg.norm(points[i] - last_kept) >= min_spacing:
            keep.append(i)
            last_kept = points[i]
    if return_indices:
        return points[keep], np.asarray(keep, dtype=np.int64)
    return points[keep]


def load_fiber_point_collection(path, collection_id, coordinate_scale=0.25, min_point_spacing=20.0):
    # Fiber JSONs are stored as one vc3d_fiber per file. Their control_points and
    # line_points are x/y/z coordinates at 4x the scale used by the regular PCL
    # JSONs. line_points is the dense traced polyline; the control points lie
    # exactly on it, and the tracer extends it ~150 points past the first and
    # last control point. We fit against the dense polyline, trimmed to the
    # first-to-last control point span so the dangling ends don't act as
    # constraints.
    with open(path, 'r') as f:
        data = json.load(f)

    if data.get('version', 1) == 1 and not data.get('control_points'):
        print(f'WARNING: fiber {path} has no control_points; skipping')
        return None
    parsed = parse_vc3d_fiber_format(data, path=path)

    control_xyz = np.asarray(parsed.control_points_xyz, dtype=np.float32)
    if control_xyz.ndim != 2 or control_xyz.shape[1] != 3:
        print(f'WARNING: fiber {path} control points have shape {control_xyz.shape}; expected (N, 3); skipping')
        return None
    num_control_points = len(control_xyz)

    line_xyz = np.asarray(parsed.line_points_xyz, dtype=np.float32)
    if len(line_xyz) >= num_control_points and num_control_points >= 1:
        # Map each control point to its nearest line point (exact coincidence in
        # practice), then trim the polyline to the control-point span.
        squared_distances = (
            (line_xyz[None, :, :] - control_xyz[:, None, :]) ** 2).sum(axis=-1)
        control_line_indices = squared_distances.argmin(axis=1)
        span_begin = int(control_line_indices.min())
        span_end = int(control_line_indices.max())
        points_xyz = line_xyz[span_begin:span_end + 1]
        control_line_indices = control_line_indices - span_begin
    else:
        # No usable dense polyline (legacy or degenerate fiber): fall back to
        # the control points themselves.
        points_xyz = control_xyz
        control_line_indices = np.arange(num_control_points)

    points_xyz = points_xyz * coordinate_scale
    original_num_points = len(points_xyz)
    # Force-keep this fiber's link endpoint control points through decimation so
    # cross-fiber links (which reference control_point indices) resolve to the
    # exact point rather than a surviving neighbour.
    link_endpoint_indices = set()
    for br in (data.get('branches') or []):
        control_index = int(br.get('control_point_index', -1))
        if 0 <= control_index < num_control_points:
            link_endpoint_indices.add(int(control_line_indices[control_index]))
    # Keep the surviving points' pre-decimation (trimmed line) indices so
    # cross-fiber links resolve exactly after decimation.
    points_xyz, kept_orig_indices = _decimate_ordered_points_min_spacing(
        points_xyz, min_point_spacing, return_indices=True,
        force_keep=link_endpoint_indices)
    name = data.get('name') or os.path.splitext(os.path.basename(path))[0]
    thing = data.get('webknossos', {}).get('thing', {})
    wk_color = thing.get('color', {})
    color = [
        float(wk_color.get('r', 0.0)),
        float(wk_color.get('g', 0.0)),
        float(wk_color.get('b', 0.0)),
    ]

    collection = {
        'id': collection_id,
        'name': name,
        'points': {},
        'metadata': {
            'source_format': data.get('type', 'vc3d_fiber'),
            'fiber_version': parsed.version,
            'fiber_generation': parsed.generation,
            'fiber_sequence': data.get('sequence'),
            'fiber_started_at': data.get('started_at'),
            'fiber_tags': data.get('tags', []),
            'hv_classification': data.get('hv_classification', {}),
            'input_coordinate_scale': coordinate_scale,
            'pcl_fiber_min_point_spacing': min_point_spacing,
            'fiber_original_num_points': original_num_points,
            'fiber_num_control_points': num_control_points,
        },
        'color': color,
    }
    # Two-step index mapping for cross-fiber links (which reference control_point
    # indices): control_line_indices maps a control_point index to its trimmed
    # line-point index, kept_orig_indices maps that to the retained point id
    # (0..k-1 in kept order, i.e. the position within kept_orig_indices).
    collection['kept_orig_indices'] = kept_orig_indices
    collection['control_line_indices'] = control_line_indices
    for point_id, (p, orig_index) in enumerate(zip(points_xyz, kept_orig_indices)):
        collection['points'][point_id] = {
            'id': point_id,
            'collectionId': collection_id,
            'p': p.tolist(),
            'winding_annotation': float('nan'),
            'creation_time': 0,
            'orig_index': int(orig_index),
        }

    # Cross-fiber links ("branches"): each entry connects one of this fiber's
    # control points to a control point on another fiber (by explicit original
    # control_point index), expressing a same-winding continuation across the
    # junction (delta 0). Stored reciprocally in both fibers' JSONs (see VC3D
    # LineAnnotationController). Resolved to retained point ids downstream via
    # kept_orig_indices (see resolve_fiber_links).
    branches = []
    for br in (data.get('branches') or []):
        branch_file = br.get('branch_file')
        local_index = int(br.get('control_point_index', -1))
        branch_index = int(br.get('branch_control_point_index', -1))
        if not branch_file or local_index < 0 or branch_index < 0:
            continue
        branches.append({
            'local_index': local_index,
            'branch_file': os.path.basename(branch_file),
            'branch_index': branch_index,
            'pending': bool(br.get('pending', False)),
        })
    collection['branches'] = branches
    collection['file_basename'] = os.path.basename(path)
    return collection


def load_fiber_point_collections(path, next_id, min_point_spacing=20.0):
    if not path:
        return {}, next_id
    fiber_paths = sorted(glob.glob(os.path.join(path, '*.json')))
    if not fiber_paths:
        print(f'no fiber point collections found in {path}')
        return {}, next_id

    point_collections = {}
    total_points = 0
    skipped = 0
    for fiber_path in fiber_paths:
        try:
            pcl = load_fiber_point_collection(fiber_path, next_id, min_point_spacing=min_point_spacing)
        except Exception as e:
            print(f'WARNING: failed to load fiber {fiber_path}: {e}')
            skipped += 1
            continue
        if pcl is None:
            skipped += 1
            continue
        pcl['source_file'] = fiber_path
        point_collections[next_id] = pcl
        total_points += len(pcl['points'])
        next_id += 1

    print(
        f'Loaded {len(point_collections)} fiber point collections '
        f'({total_points} points, min spacing {min_point_spacing:g} vx) from {path}'
        + (f'; skipped {skipped}' if skipped else '')
    )
    return point_collections, next_id


def _point_id_for_orig_index(pcl, orig_index):
    """Map an original control_point index to the retained point id.

    Fiber collections hold decimated line points, so this is a two-step map:
    control_point index -> trimmed line-point index (control_line_indices),
    then to the position of the nearest kept index within kept_orig_indices
    (exact when that index survived decimation; its nearest surviving
    neighbour otherwise).
    """
    if orig_index < 0:
        return None
    control_line = pcl.get('control_line_indices')
    if control_line is not None and len(control_line):
        if orig_index >= len(control_line):
            return None
        orig_index = int(control_line[orig_index])
    kept = pcl.get('kept_orig_indices')
    if kept is None or len(kept) == 0:
        return None
    return int(np.argmin(np.abs(np.asarray(kept) - orig_index)))


def resolve_fiber_links(point_collections, include_pending=False):
    """Resolve stored branch metadata into concrete point-to-point links.

    Fibers/PCLs carry raw 'branches' (see load_fiber_point_collection), each
    naming the linked collection by 'branch_file' and the two endpoints by their
    explicit original control_point indices. We map those indices to retained
    point ids (via kept_orig_indices) and dedupe the reciprocal entries so each
    undirected link appears once.

    Returns a list of dicts:
        {'a_coll', 'a_point', 'b_coll', 'b_point', 'pending'}
    """
    by_basename = {}
    for cid, pcl in point_collections.items():
        basename = pcl.get('file_basename')
        if basename is not None:
            by_basename.setdefault(basename, cid)

    # Links are same-winding statements between unannotated collections; drop any
    # link touching an explicitly-annotated collection here, before the component
    # decomposition, so every consumer of the link graph (the cross-patch merge,
    # the unattached walk sampling) agrees on membership. Must run before
    # normalise_pcl_winding_annotations 0-fills unannotated pcls.
    annotated_cids = {
        cid for cid, pcl in point_collections.items()
        if any(np.isfinite(p['winding_annotation']) for p in pcl['points'].values())
    }

    links = []
    seen = set()
    for cid, pcl in point_collections.items():
        for br in pcl.get('branches', []):
            if br.get('pending') and not include_pending:
                continue
            target_cid = by_basename.get(br['branch_file'])
            if target_cid is None or target_cid == cid:
                continue
            if cid in annotated_cids or target_cid in annotated_cids:
                print(f'WARNING: dropping fiber link {pcl.get("name")} -> '
                      f'{br["branch_file"]}: links must join unannotated '
                      f'(same-winding) collections')
                continue
            a_point = _point_id_for_orig_index(pcl, br['local_index'])
            b_point = _point_id_for_orig_index(
                point_collections[target_cid], br['branch_index'])
            if a_point is None or b_point is None:
                continue
            key = tuple(sorted([(cid, a_point), (target_cid, b_point)]))
            if key in seen:
                continue
            seen.add(key)
            links.append({
                'a_coll': cid, 'a_point': a_point,
                'b_coll': target_cid, 'b_point': b_point,
                'pending': bool(br.get('pending')),
            })
    return links


def build_link_components(resolved_links):
    """Connected components of collections joined by cross-fiber links.

    Returns a list of (member_cids, member_links): member_cids in BFS order from
    the component's first-seen collection, member_links the resolved links whose
    endpoints both lie in the component."""
    adjacency = {}
    for link in resolved_links:
        adjacency.setdefault(link['a_coll'], []).append(link)
        adjacency.setdefault(link['b_coll'], []).append(link)
    components = []
    seen = set()
    for seed in adjacency:
        if seed in seen:
            continue
        member_cids = []
        member_links = []
        seen_links = set()
        queue = [seed]
        seen.add(seed)
        while queue:
            cid = queue.pop(0)
            member_cids.append(cid)
            for link in adjacency[cid]:
                if id(link) not in seen_links:
                    seen_links.add(id(link))
                    member_links.append(link)
                other = link['b_coll'] if link['a_coll'] == cid else link['a_coll']
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        components.append((member_cids, member_links))
    return components


class Chain:
    """Traversal interface every pcl carries as pcl['chain'].

    Consumers go through this and never assume the pcl's id-sorted point order
    is chain-valid (it is not for merged fiber-link components, whose chains
    route through their fiber graph). Two entry points:
      - zyxs_between(p1, p2): ordered (N, 3) [z, y, x] chain from p1 to p2 with
        every consecutive pair |dtheta| < pi apart, for sequential theta=0
        unwrapping;
      - iter_chain(): a chain-valid point sequence covering the whole pcl (may
        revisit points; consumers wanting each adjacency once must dedupe)."""

    def zyxs_between(self, p1, p2):
        raise NotImplementedError

    def iter_chain(self):
        raise NotImplementedError

    @staticmethod
    def _zyxs(points):
        return np.stack([p['zyx'] for p in points], axis=0).astype(np.float32)


class SequenceChain(Chain):
    """Chain for an ordinary (single-sequence) pcl: id-sorted point order.

    Holds a reference to the pcl dict; the sorted view is cached against the
    identity and length of pcl['points'], so both replacing the points dict
    wholesale (as the z-roi trims do) and adding/removing keys in place
    invalidate it. Same-size in-place key replacement would go undetected --
    don't do that; replace the dict instead."""

    def __init__(self, pcl):
        self.pcl = pcl
        self._source = None
        self._source_len = -1
        self._ordered = None
        self._index_of = None

    def _sorted(self):
        points = self.pcl['points']
        if self._source is not points or self._source_len != len(points):
            self._source = points
            self._source_len = len(points)
            self._ordered = [p for _, p in sorted(points.items(), key=lambda kv: int(kv[0]))]
            self._index_of = {id(p): k for k, p in enumerate(self._ordered)}
        return self._ordered, self._index_of

    def zyxs_between(self, p1, p2):
        ordered, index_of = self._sorted()
        i1, i2 = index_of[id(p1)], index_of[id(p2)]
        if i1 <= i2:
            chain = ordered[i1:i2 + 1]
        else:
            chain = list(reversed(ordered[i2:i1 + 1]))
        return self._zyxs(chain)

    def iter_chain(self):
        return self._sorted()[0]


class ComponentChain(Chain):
    """Chain for a merged fiber-link component: routes through the fiber graph.

    member_sorted[m] is member m's points in int-id order; pos_of maps
    id(point) -> (member, position); tree_parent[m] is None for the root else
    (parent_member, pos_in_parent, pos_in_m) for the link joining m to its
    spanning-tree parent; extra_edges are the loop-closing link edges not in the
    spanning tree, as (m_a, pos_a, m_b, pos_b). zyxs_between routes through the
    tree only (any route is winding-equivalent when the annotations are
    consistent); iter_chain also crosses every extra edge once, so tour-walking
    consumers (holonomy detection in find_inconsistent_windings) see each link
    cycle's constraint. Junction hop endpoints are nearly coincident, so every
    consecutive chain pair satisfies the |dtheta| < pi unwrap assumption."""

    def __init__(self, member_sorted, pos_of, tree_parent, extra_edges=()):
        self.member_sorted = member_sorted
        self.pos_of = pos_of
        self.tree_parent = tree_parent
        self.extra_edges = list(extra_edges)

    def _path_to_root(self, m):
        path = [m]
        while self.tree_parent[m] is not None:
            m = self.tree_parent[m][0]
            path.append(m)
        return path

    def _hop_positions(self, m_from, m_to):
        # Positions (leave in m_from, arrive in m_to) of the tree link between
        # two adjacent members, whichever of the two is the tree child.
        tree_parent = self.tree_parent
        if tree_parent[m_to] is not None and tree_parent[m_to][0] == m_from:
            _, pos_parent, pos_child = tree_parent[m_to]
            return pos_parent, pos_child
        assert tree_parent[m_from] is not None and tree_parent[m_from][0] == m_to
        _, pos_parent, pos_child = tree_parent[m_from]
        return pos_child, pos_parent

    def _member_segment(self, m, pos_from, pos_to):
        points = self.member_sorted[m]
        if pos_from <= pos_to:
            return points[pos_from:pos_to + 1]
        return list(reversed(points[pos_to:pos_from + 1]))

    def zyxs_between(self, p1, p2):
        # Within-member index ranges concatenated across the spanning-tree path
        # from p1's member to p2's, hopping fibers at each junction.
        m1, i1 = self.pos_of[id(p1)]
        m2, i2 = self.pos_of[id(p2)]
        if m1 == m2:
            member_path = [m1]
        else:
            up1 = self._path_to_root(m1)
            up2 = self._path_to_root(m2)
            in_up2 = {m: k for k, m in enumerate(up2)}
            lca_idx1 = next(k for k, m in enumerate(up1) if m in in_up2)
            lca = up1[lca_idx1]
            member_path = up1[:lca_idx1 + 1] + list(reversed(up2[:in_up2[lca]]))
        chain = []
        pos = i1
        for m_from, m_to in zip(member_path, member_path[1:]):
            leave, arrive = self._hop_positions(m_from, m_to)
            chain.extend(self._member_segment(m_from, pos, leave))
            pos = arrive
        chain.extend(self._member_segment(member_path[-1], pos, i2))
        return self._zyxs(chain)

    def iter_chain(self):
        # Euler tour of the member tree: walk each member end-to-end and back,
        # detouring into each child at its junction position (and returning), so
        # every consecutive pair of tour points is either an adjacent same-fiber
        # pair or a nearly-coincident junction hop. Loop-closing (non-tree) link
        # edges are also crossed once each, at the first visit of either
        # endpoint: hop across, re-walk the far member without further detours
        # (its own subtree and edges are covered by the main tour), and hop
        # back. Adjustment-accumulating consumers thereby see every link
        # cycle's constraint, not just the spanning tree's.
        children = {m: {} for m in self.tree_parent}
        root = None
        for m, parent in self.tree_parent.items():
            if parent is None:
                root = m
            else:
                parent_m, pos_in_parent, pos_in_child = parent
                children[parent_m].setdefault(pos_in_parent, []).append((m, pos_in_child))
        extra_at = {}
        for k, (m_a, pos_a, m_b, pos_b) in enumerate(self.extra_edges):
            extra_at.setdefault((m_a, pos_a), []).append((k, m_b, pos_b))
            extra_at.setdefault((m_b, pos_b), []).append((k, m_a, pos_a))
        crossed = set()
        out = []

        def tour(m, enter_pos, detour=True):
            points = self.member_sorted[m]
            n = len(points)
            kids = children[m]
            seen_pos = set()
            walk = (list(range(enter_pos, -1, -1)) + list(range(1, n))
                    + list(range(n - 2, enter_pos - 1, -1)))
            for pos in walk:
                out.append(points[pos])
                if not detour or pos in seen_pos:
                    continue
                seen_pos.add(pos)
                for child, child_pos in kids.get(pos, []):
                    tour(child, child_pos)
                    out.append(points[pos])  # hop back through the junction
                for k, other, other_pos in extra_at.get((m, pos), []):
                    if k in crossed:
                        continue
                    crossed.add(k)
                    tour(other, other_pos, detour=False)
                    out.append(points[pos])  # hop back through the junction

        tour(root, 0)
        return out


def _index_component_members(members):
    """Per-member id-sorted point lists and the id(point) -> (member, pos) map."""
    member_sorted = [
        sorted(pcl['points'].values(), key=lambda point: int(point['id']))
        for _, pcl in members
    ]
    pos_of = {}
    for m, points in enumerate(member_sorted):
        for pos, point in enumerate(points):
            pos_of[id(point)] = (m, pos)
    return member_sorted, pos_of


def _component_link_edges(members, member_links, point_collections, pos_of):
    """Member-indexed undirected link edges (m_a, pos_a, m_b, pos_b), deduped;
    links whose endpoints fall outside `members` (or within one member) drop."""
    member_index = {cid: m for m, (cid, _) in enumerate(members)}
    edges = []
    seen = set()
    for link in member_links:
        m_a = member_index.get(link['a_coll'])
        m_b = member_index.get(link['b_coll'])
        if m_a is None or m_b is None or m_a == m_b:
            continue
        pa = point_collections[link['a_coll']]['points'][link['a_point']]
        pb = point_collections[link['b_coll']]['points'][link['b_point']]
        edge = (m_a, pos_of[id(pa)][1], m_b, pos_of[id(pb)][1])
        key = frozenset([(edge[0], edge[1]), (edge[2], edge[3])])
        if key in seen:
            continue
        seen.add(key)
        edges.append(edge)
    return edges


def _link_spanning_tree(num_members, link_edges):
    """BFS spanning tree over members from member 0 along the link edges.

    Returns (tree_parent, extra_edges): tree_parent as ComponentChain expects it
    (covering only the members reachable from 0), extra_edges the loop-closing
    non-tree edges whose members are both reachable."""
    adjacency = {m: [] for m in range(num_members)}
    for k, (m_a, pos_a, m_b, pos_b) in enumerate(link_edges):
        adjacency[m_a].append((k, m_b, pos_a, pos_b))
        adjacency[m_b].append((k, m_a, pos_b, pos_a))
    tree_parent = {0: None}
    tree_edge_ids = set()
    queue = [0]
    while queue:
        m = queue.pop(0)
        for k, neighbour, pos_m, pos_n in adjacency[m]:
            if neighbour not in tree_parent:
                tree_parent[neighbour] = (m, pos_m, pos_n)
                tree_edge_ids.add(k)
                queue.append(neighbour)
    extra_edges = [edge for k, edge in enumerate(link_edges)
                   if k not in tree_edge_ids
                   and edge[0] in tree_parent and edge[2] in tree_parent]
    return tree_parent, extra_edges


def merge_linked_point_collections(point_collections, link_components,
                                   cross_patch_point_collections):
    """Fold link-connected collections into merged cross-patch component pcls.

    link_components is the shared decomposition from build_link_components
    (annotated collections never appear: resolve_fiber_links drops their links).
    For each link component, removes the member collections from the cross-patch
    pool and (when the union holds >= 2 attached points) adds one merged pcl
    whose points are the union of all member points, renumbered member-major.
    The merged pcl carries the same uniform chain interface as ordinary pcls
    (pcl['chain'], see Chain) -- a ComponentChain routing through the fiber
    graph, since its id-sorted point order is NOT chain-valid across members.
    'link_member_cids' records the member collection ids for diagnostics.

    Returns (cross_patch_point_collections, num_merged); the input dict is
    modified in place and also returned."""
    num_merged = 0
    for member_cids, member_links in link_components:
        members = [(cid, point_collections[cid]) for cid in member_cids
                   if cid in point_collections]
        if len(members) < 2:
            continue
        member_sorted, pos_of = _index_component_members(members)
        link_edges = _component_link_edges(members, member_links, point_collections, pos_of)
        tree_parent, extra_edges = _link_spanning_tree(len(members), link_edges)
        if len(tree_parent) < len(members):
            # Members whose links resolved against collections outside
            # point_collections can leave the component disconnected; keep only
            # the part reachable from the first member merged and leave the rest
            # as-is. Rebuild the indices/edges/tree from the subset rather than
            # remapping in place.
            print(f'WARNING: fiber-link component {member_cids} not fully '
                  f'connected after loading; merging only the reachable part')
            members = [members[m] for m in sorted(tree_parent)]
            member_sorted, pos_of = _index_component_members(members)
            link_edges = _component_link_edges(members, member_links, point_collections, pos_of)
            tree_parent, extra_edges = _link_spanning_tree(len(members), link_edges)
        # Merged keys are fresh member-major ordinals, but the point dicts are the
        # members' own (shared, not copied -- they also live on in the unattached
        # pool) and keep their member-local 'id', which stays meaningful for
        # tracing a point back to its source fiber. So on a merged pcl key !=
        # point['id']: never index merged_points by a point's 'id'.
        merged_points = {}
        for points in member_sorted:
            for point in points:
                merged_points[len(merged_points)] = point
        for cid, _ in members:
            cross_patch_point_collections.pop(cid, None)
        num_attached = sum(1 for point in merged_points.values()
                           if 'on_patch' in point)
        if num_attached < 2:
            continue
        merged_id = f'fibercomp:{num_merged}'
        rep = members[0][1]
        if extra_edges:
            print(f'fiber-link component {merged_id} '
                  f'({[cid for cid, _ in members]}): {len(extra_edges)} '
                  f'loop-closing link(s) beyond the spanning tree')
        cross_patch_point_collections[merged_id] = {
            'id': merged_id,
            'name': merged_id,
            'sampling_group': rep.get('sampling_group', 'fibers'),
            'metadata': {'winding_is_absolute': False,
                         'input_role': 'fiber_link_component'},
            'points': merged_points,
            'link_member_cids': [cid for cid, _ in members],
            'chain': ComponentChain(member_sorted, pos_of, tree_parent, extra_edges),
        }
        num_merged += 1
    return cross_patch_point_collections, num_merged


def _huber_abs(residual, delta):
    abs_residual = residual.abs()
    return torch.where(
        abs_residual <= delta,
        0.5 * residual ** 2 / delta,
        abs_residual - 0.5 * delta,
    )


def _cauchy_abs(residual, c):
    """Cauchy (Lorentzian) robust loss, applied element-wise.

    Formula:  ρ(r; c) = (c²/2) · log(1 + (r/c)²)
    Gradient: dρ/dr  = r / (1 + (r/c)²)

    For |r| >> c the gradient decays toward zero, so outlier samples
    (e.g. track points straddling two windings) exert vanishing pull on the
    optimiser.  For |r| << c the loss is approximately quadratic (r²/2),
    identical to the L2/Huber near-zero regime.

    The operating point c = dr_per_winding / 2 (half the winding spacing in
    canonical radius units) puts a path one half-winding off firmly in the
    log regime: its gradient is suppressed by ~80%, preventing the spiral
    from averaging between two adjacent windings.

    Args:
        residual: Tensor of per-point residuals (already hinged / margined
            by the caller when appropriate).
        c: Scale parameter.  Must be > 0.  Controls the transition from
            quadratic to logarithmic regime.

    Returns:
        Element-wise Cauchy loss values, same shape as *residual*.
    """
    return (c ** 2 / 2.0) * torch.log1p((residual / c) ** 2)


def _get_patch_valid_points(patch, device, z_begin, z_end, max_points=None, fixed_num_points=None):
    valid_mask = patch.valid_vertex_mask
    z_in_roi = (patch.zyxs[..., 0] >= z_begin) & (patch.zyxs[..., 0] < z_end)
    valid_indices = torch.where(valid_mask & z_in_roi)
    if len(valid_indices[0]) == 0:
        valid_indices = torch.where(valid_mask)
    n = len(valid_indices[0])
    if fixed_num_points is not None:
        sel = np.random.choice(n, fixed_num_points, replace=(n < fixed_num_points))
        valid_indices = (valid_indices[0][sel], valid_indices[1][sel])
    elif max_points is not None and n > max_points:
        sel = np.random.choice(n, max_points, replace=False)
        valid_indices = (valid_indices[0][sel], valid_indices[1][sel])
    return patch.zyxs[valid_indices[0], valid_indices[1], :].to(device=device, dtype=torch.float32)


def get_face_indices(h, w):
    indices = torch.arange(h * w).view(h, w)
    top_left = indices[:-1, :-1].flatten()
    top_right = indices[:-1, 1:].flatten()
    bottom_left = indices[1:, :-1].flatten()
    bottom_right = indices[1:, 1:].flatten()
    return torch.cat([
        torch.stack([bottom_left, top_left, top_right], dim=1),
        torch.stack([bottom_left, top_right, bottom_right], dim=1)
    ], dim=0)


@torch.inference_mode()
def compute_winding_range_and_input_extents(
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    unattached_pcl_strips,
    cfg,
    z_begin,
    z_end,
    get_or_build_unattached_pcl_flat,
    authoritative_zyx_lines=(),
):
    """Compute output winding range plus max observed radius/winding per patch and PCL."""
    device = dr_per_winding.device
    dr = dr_per_winding.detach()
    min_w = None
    max_w = None

    def update_from_winding_indices(winding_indices):
        nonlocal min_w, max_w
        local_min = int(winding_indices.min().item())
        local_max = int(winding_indices.max().item())
        min_w = local_min if min_w is None else min(min_w, local_min)
        max_w = local_max if max_w is None else max(max_w, local_max)

    chunk = 65536

    def transform_in_chunks(zyxs):
        spiral_pieces = []
        for start in range(0, zyxs.shape[0], chunk):
            spiral_pieces.append(slice_to_spiral_transform(zyxs[start:start + chunk]))
        return torch.cat(spiral_pieces, dim=0) if len(spiral_pieces) > 1 else spiral_pieces[0]

    patch_extents = [(None, None)] * len(patches)
    for patch_index, patch in enumerate(patches):
        patch_zyxs = patch.zyxs.to(device=device, dtype=torch.float32)
        if patch_zyxs.shape[0] < 2 or patch_zyxs.shape[1] < 2:
            continue
        valid_quad_mask = patch.valid_quad_mask.to(device=device)
        quad_center_zyxs = (
            patch_zyxs[:-1, :-1]
            + patch_zyxs[1:, :-1]
            + patch_zyxs[:-1, 1:]
            + patch_zyxs[1:, 1:]
        ) / 4
        quad_zs = torch.stack([
            patch_zyxs[:-1, :-1, 0],
            patch_zyxs[1:, :-1, 0],
            patch_zyxs[:-1, 1:, 0],
            patch_zyxs[1:, 1:, 0],
        ], dim=0)
        quad_touches_roi = (quad_zs.amax(dim=0) >= z_begin) & (quad_zs.amin(dim=0) < z_end)
        mask = valid_quad_mask & quad_touches_roi
        if not mask.any():
            continue
        spiral_zyxs = transform_in_chunks(quad_center_zyxs[mask])
        _, radius, shifted_radius = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
        winding_indices = (shifted_radius / dr).round().to(torch.int64).clamp_min(0)
        update_from_winding_indices(winding_indices)
        patch_extents[patch_index] = (
            float(radius.max().item()),
            int(winding_indices.max().item()),
        )

    pcl_extents = [(None, None)] * len(unattached_pcl_strips)
    if unattached_pcl_strips:
        flat = get_or_build_unattached_pcl_flat(unattached_pcl_strips, device)
        if flat is not None and flat['total'] > 0:
            zyxs = flat['zyxs']
            strip_id = flat['strip_id']
            num_strips = flat['num_strips']
            in_roi = (zyxs[:, 0] >= z_begin) & (zyxs[:, 0] < z_end)
            if in_roi.any():
                zyxs_roi = zyxs[in_roi]
                strip_id_roi = strip_id[in_roi]
                spiral_zyxs = transform_in_chunks(zyxs_roi)
                _, radius, shifted_radius = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
                winding_indices = (shifted_radius / dr).round().to(torch.int64).clamp_min(0)
                update_from_winding_indices(winding_indices)
                per_strip_max_r = torch.zeros(num_strips, dtype=torch.float32, device=device)
                per_strip_max_w = torch.full((num_strips,), -1, dtype=torch.int64, device=device)
                strip_has_roi = torch.zeros(num_strips, dtype=torch.bool, device=device)
                per_strip_max_r.scatter_reduce_(0, strip_id_roi, radius.to(torch.float32), reduce='amax')
                per_strip_max_w.scatter_reduce_(0, strip_id_roi, winding_indices, reduce='amax')
                strip_has_roi.scatter_(0, strip_id_roi, torch.ones_like(strip_id_roi, dtype=torch.bool))
                per_strip_max_r_cpu = per_strip_max_r.cpu().tolist()
                per_strip_max_w_cpu = per_strip_max_w.cpu().tolist()
                strip_has_roi_cpu = strip_has_roi.cpu().tolist()
                for k in range(num_strips):
                    if strip_has_roi_cpu[k]:
                        pcl_extents[k] = (per_strip_max_r_cpu[k], per_strip_max_w_cpu[k])

    # Tracks are authoritative fit geometry too.  Including them makes
    # tracks-only/disable-patches sessions derive the same output upper bound
    # instead of silently producing no preview.  Track DBMs can contain millions
    # of short lines, so batch their points before moving them to the GPU.  A
    # transform call per line makes preview generation dominated by CUDA launch
    # overhead (and took hours for multi-million-track datasets).
    pending_track_points = []
    pending_track_point_count = 0

    def update_from_track_points(zyxs):
        # Reduce each transformed chunk immediately.  In addition to bounding
        # memory, this lets callers reuse an already-flat GPU track tensor
        # without concatenating a second full-dataset transform result.
        for start in range(0, zyxs.shape[0], chunk):
            spiral_zyxs = slice_to_spiral_transform(zyxs[start:start + chunk])
            _, _, shifted_radius = get_theta_and_radii(spiral_zyxs[..., 1:], dr_per_winding)
            update_from_winding_indices(
                (shifted_radius / dr).round().to(torch.int64).clamp_min(0)
            )

    def flush_pending_track_points():
        nonlocal pending_track_point_count
        if not pending_track_points:
            return
        points = (pending_track_points[0] if len(pending_track_points) == 1
                  else np.concatenate(pending_track_points, axis=0))
        update_from_track_points(torch.as_tensor(points, device=device, dtype=torch.float32))
        pending_track_points.clear()
        pending_track_point_count = 0

    for line in authoritative_zyx_lines or ():
        if torch.is_tensor(line):
            # Keep tensor callers supported without trying to convert CUDA data
            # through NumPy.  Flush host-side points first to preserve the bound.
            flush_pending_track_points()
            zyxs = line.to(device=device, dtype=torch.float32).reshape(-1, 3)
            in_roi = (zyxs[:, 0] >= z_begin) & (zyxs[:, 0] < z_end)
            if in_roi.any():
                update_from_track_points(zyxs[in_roi])
            continue

        points = np.asarray(line, dtype=np.float32).reshape(-1, 3)
        if points.shape[0] == 0:
            continue
        in_roi = (points[:, 0] >= z_begin) & (points[:, 0] < z_end)
        if not in_roi.any():
            continue
        points = points[in_roi]
        pending_track_points.append(points)
        pending_track_point_count += points.shape[0]
        if pending_track_point_count >= chunk:
            flush_pending_track_points()
    flush_pending_track_points()

    first_winding = cfg['output_first_winding']
    if min_w is None:
        output_winding_range = (first_winding, first_winding)
    else:
        margin = cfg['output_winding_margin']
        output_winding_range = (max(min_w - margin, first_winding), max_w + 1 + margin)
    return output_winding_range, patch_extents, pcl_extents


def _infer_shell_outer_winding_idx(
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    unattached_pcl_strips,
    cfg,
    z_begin,
    z_end,
    get_or_build_unattached_pcl_flat,
):
    _, patch_extents, pcl_extents = compute_winding_range_and_input_extents(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        unattached_pcl_strips,
        cfg,
        z_begin,
        z_end,
        get_or_build_unattached_pcl_flat,
    )
    observed_max = None
    for _, max_w in patch_extents + pcl_extents:
        if max_w is None:
            continue
        observed_max = max_w if observed_max is None else max(observed_max, max_w)
    if observed_max is None:
        observed_max = cfg['output_first_winding']
    return int(observed_max + cfg['shell_outer_winding_margin'])


def _resolve_shell_outer_winding_idx(cfg):
    # Winding bound shared by every sampler that integrates over the spiral
    # cylinder: the dense lasagna losses, the symmetric Dirichlet
    # regulariser and the phase bundle (incl. min_spacing). Resolved once per
    # run from the config; the shell branch may still override it with the
    # inferred value. None disables those samplers (they early-return zero),
    # whatever their weights: _structurally_disabled_dense_weight_keys
    # reports that combination so the run can warn instead of staying silent.
    configured = cfg['shell_outer_winding_idx']
    if configured is None:
        return None
    try:
        resolved = int(configured)
    except (TypeError, ValueError):
        raise ValueError(
            f'shell_outer_winding_idx must be an integer >= 2 or None, '
            f'got {configured!r}')
    if resolved < 2:
        # sample_spiral_surface_frame draws windings from [1, idx); 0 and 1
        # crash multinomial/arange at the first step with an opaque error.
        raise ValueError(
            f'shell_outer_winding_idx must be an integer >= 2 or None, '
            f'got {configured!r}')
    return resolved


def resolve_outer_winding_idx_and_notes(cfg, shell_active, infer_outer_winding_idx):
    """Resolve the run's outer winding index and the lines to print for it.

    ``infer_outer_winding_idx`` is a zero-argument callable, invoked only
    when shell losses are active and the config key is None (the historical
    inference path). Returns ``(index_or_none, notes)``.
    """
    idx = _resolve_shell_outer_winding_idx(cfg)
    notes = []
    if shell_active:
        if idx is None:
            idx = int(infer_outer_winding_idx())
            notes.append(f'inferred shell_outer_winding_idx = {idx}')
        else:
            notes.append(f'using configured shell_outer_winding_idx = {idx}')
    elif idx is not None:
        notes.append(
            'no outer-shell losses; using configured shell_outer_winding_idx '
            f'= {idx} for the dense and regularisation losses')
    if idx is not None:
        min_gap = idx + 3
        gap_windings = cfg['model_gap_expander_num_windings']
        if gap_windings < min_gap:
            notes.append(
                f'WARNING: shell_outer_winding_idx {idx} requires '
                f'model_gap_expander_num_windings >= {min_gap}, got '
                f'model_gap_expander_num_windings {gap_windings}; '
                'increase model_gap_expander_num_windings or lower '
                'shell_outer_winding_idx')
    return idx, notes


# Every loss weight whose term samples out to shell_outer_winding_idx and is
# therefore silently zero while that index is unresolved.
_DENSE_WEIGHT_KEYS_NEEDING_OUTER_WINDING_IDX = (
    'loss_weight_dense_normals',
    'loss_weight_dense_spacing',
    'loss_weight_dense_spacing_count',
    'loss_weight_dense_spacing_density',
    'loss_weight_dense_attachment',
    'loss_weight_min_spacing',
    'loss_weight_sym_dirichlet',
)


def _structurally_disabled_dense_weight_keys(cfg, shell_outer_winding_idx):
    """Nonzero weights that cannot produce a loss because no outer winding
    index could be resolved (config key is None and no outer shell inferred
    one)."""
    if shell_outer_winding_idx is not None:
        return ()
    return tuple(
        key for key in _DENSE_WEIGHT_KEYS_NEEDING_OUTER_WINDING_IDX
        if cfg.get(key, 0.0) > 0
    )


def _warn_if_inputs_exceed_flow_bounds(
    patch_ids,
    patch_extents,
    unattached_pcl_strips,
    pcl_extents,
    flow_field_radius,
    cfg,
):
    gap_expander_num_windings = cfg['model_gap_expander_num_windings']

    over_radius_patches = []
    over_winding_patches = []
    for pid, (max_r, max_w) in zip(patch_ids, patch_extents):
        if max_r is None:
            continue
        if max_r > flow_field_radius:
            over_radius_patches.append((pid, max_r))
        if max_w >= gap_expander_num_windings:
            over_winding_patches.append((pid, max_w))

    over_radius_pcls = []
    over_winding_pcls = []
    for k, (strip, (max_r, max_w)) in enumerate(zip(unattached_pcl_strips, pcl_extents)):
        if max_r is None:
            continue
        name = strip.get('name') or strip.get('id') or strip.get('source_file') or f'#{k}'
        if max_r > flow_field_radius:
            over_radius_pcls.append((name, max_r))
        if max_w >= gap_expander_num_windings:
            over_winding_pcls.append((name, max_w))

    def _print_offenders(kind, value_label, threshold_label, patches, pcls, fmt):
        if not (patches or pcls):
            return
        print(f'WARNING: {len(patches)} patch(es) and {len(pcls)} unattached pcl(s) have {value_label} exceeding {threshold_label}:')
        for pid, v in sorted(patches, key=lambda e: -e[1])[:10]:
            print(f'  patch {pid}: max {kind} {fmt(v)}')
        if len(patches) > 10:
            print(f'  ... and {len(patches) - 10} more patches')
        for name, v in sorted(pcls, key=lambda e: -e[1])[:10]:
            print(f'  pcl {name}: max {kind} {fmt(v)}')
        if len(pcls) > 10:
            print(f'  ... and {len(pcls) - 10} more pcls')

    _print_offenders(
        'spiral radius', 'spiral-space radius', f'flow_bounds_radius ({flow_field_radius})',
        over_radius_patches, over_radius_pcls, lambda v: f'{v:.1f}',
    )
    _print_offenders(
        'winding idx', 'winding index', f'gap_expander_num_windings ({gap_expander_num_windings})',
        over_winding_patches, over_winding_pcls, lambda v: f'{v}',
    )


def _rasterize_triangles_into_mesh(
    tri_uvs,
    tri_scrolls,
    tri_target_w,
    scroll_zyxs,
    winding_offsets_t,
    num_thetas_t,
):
    device = scroll_zyxs.device
    T = tri_uvs.shape[0]
    if T == 0:
        return

    num_zs = scroll_zyxs.shape[0]
    v_lim_per_tri = num_thetas_t[tri_target_w]

    u_min = torch.floor(tri_uvs[..., 0].min(dim=-1).values).to(torch.long)
    u_max = torch.ceil(tri_uvs[..., 0].max(dim=-1).values).to(torch.long)
    v_min = torch.floor(tri_uvs[..., 1].min(dim=-1).values).to(torch.long)
    v_max = torch.ceil(tri_uvs[..., 1].max(dim=-1).values).to(torch.long)
    u_min = u_min.clamp(min=0, max=num_zs - 1)
    u_max = u_max.clamp(min=0, max=num_zs - 1)
    v_min = v_min.clamp(min=0)
    v_max = torch.minimum(v_max, v_lim_per_tri - 1)
    valid_bbox = (u_min <= u_max) & (v_min <= v_max)

    bbox_h = (u_max - u_min + 1).clamp(min=1)
    bbox_w = (v_max - v_min + 1).clamp(min=1)

    chunk_size = 16384
    for s in range(0, T, chunk_size):
        e = min(s + chunk_size, T)
        valid_c = valid_bbox[s:e]
        if not valid_c.any():
            continue
        u_min_c = u_min[s:e]
        v_min_c = v_min[s:e]
        bbox_h_c = bbox_h[s:e]
        bbox_w_c = bbox_w[s:e]
        max_h = int(bbox_h_c[valid_c].max().item())
        max_w = int(bbox_w_c[valid_c].max().item())

        du_grid, dv_grid = torch.meshgrid(
            torch.arange(max_h, device=device),
            torch.arange(max_w, device=device),
            indexing='ij',
        )
        us = u_min_c[:, None, None] + du_grid[None]
        vs = v_min_c[:, None, None] + dv_grid[None]
        in_bbox = (
            (du_grid[None] < bbox_h_c[:, None, None])
            & (dv_grid[None] < bbox_w_c[:, None, None])
            & valid_c[:, None, None]
        )

        tri_uvs_c = tri_uvs[s:e]
        pts = torch.stack([us.float(), vs.float()], dim=-1)
        a = tri_uvs_c[:, 0]
        b = tri_uvs_c[:, 1]
        c = tri_uvs_c[:, 2]
        v0 = b - a
        v1 = c - a
        v2 = pts - a[:, None, None]
        d00 = (v0 * v0).sum(-1)
        d01 = (v0 * v1).sum(-1)
        d11 = (v1 * v1).sum(-1)
        d20 = (v2 * v0[:, None, None]).sum(-1)
        d21 = (v2 * v1[:, None, None]).sum(-1)
        denom = d00 * d11 - d01 * d01
        nonzero = denom.abs() >= 1e-9
        denom_safe = torch.where(nonzero, denom, torch.ones_like(denom))
        beta = (d11[:, None, None] * d20 - d01[:, None, None] * d21) / denom_safe[:, None, None]
        gamma = (d00[:, None, None] * d21 - d01[:, None, None] * d20) / denom_safe[:, None, None]
        alpha = 1 - beta - gamma
        inside = (alpha >= -1e-6) & (beta >= -1e-6) & (gamma >= -1e-6) & nonzero[:, None, None]
        mask = in_bbox & inside
        if not mask.any():
            continue

        tri_scrolls_c = tri_scrolls[s:e]
        sa = tri_scrolls_c[:, 0][:, None, None, :]
        sb = tri_scrolls_c[:, 1][:, None, None, :]
        sc = tri_scrolls_c[:, 2][:, None, None, :]
        interp = alpha[..., None] * sa + beta[..., None] * sb + gamma[..., None] * sc

        sel = mask.reshape(-1)
        target_u = us.reshape(-1)[sel]
        target_v_local = vs.reshape(-1)[sel]
        target_w_flat = tri_target_w[s:e][:, None, None].expand(-1, max_h, max_w).reshape(-1)[sel]
        target_v_global = winding_offsets_t[target_w_flat] + target_v_local
        scroll_zyxs[target_u, target_v_global] = interp.reshape(-1, 3)[sel]


@torch.inference_mode()
def _build_spliced_overlay(
    scroll_zyxs,
    num_thetas_by_winding,
    z0,
    grid_spacing,
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    satisfied_patches,
    boundary_satisfied_patches,
    target_winding_idx_per_patch,
):
    device = scroll_zyxs.device
    dr = dr_per_winding.detach()
    num_windings = len(num_thetas_by_winding)
    winding_offsets_t = torch.cat([
        torch.zeros([1], dtype=torch.long, device=device),
        torch.cumsum(torch.tensor(num_thetas_by_winding, dtype=torch.long, device=device), dim=0),
    ])
    num_thetas_t = torch.tensor(num_thetas_by_winding, dtype=torch.long, device=device)

    all_quad_uvs = []
    all_quad_scrolls = []
    all_target_w = []
    chunk = 65536

    for patch_idx, patch in enumerate(patches):
        if not (bool(satisfied_patches[patch_idx]) or bool(boundary_satisfied_patches[patch_idx])):
            continue
        target_winding_idx = target_winding_idx_per_patch[patch_idx].to(device)
        quad_mask = (target_winding_idx >= 0) & (target_winding_idx < num_windings)
        if not quad_mask.any():
            continue

        patch_zyxs = patch.zyxs.to(device=device, dtype=torch.float32)
        Hv, Wv = patch_zyxs.shape[:2]

        def chunked_transform(flat):
            pieces = []
            for s in range(0, flat.shape[0], chunk):
                pieces.append(slice_to_spiral_transform(flat[s : s + chunk]))
            return torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]

        vertex_spiral = chunked_transform(patch_zyxs.reshape(-1, 3)).reshape(Hv, Wv, 3)
        v_theta, _, v_shifted = get_theta_and_radii(vertex_spiral[..., 1:], dr_per_winding)
        v_winding_raw = (v_shifted / dr).round().to(torch.int64)

        quad_center_scroll = (
            patch_zyxs[:-1, :-1]
            + patch_zyxs[1:, :-1]
            + patch_zyxs[:-1, 1:]
            + patch_zyxs[1:, 1:]
        ) / 4
        center_spiral = chunked_transform(quad_center_scroll.reshape(-1, 3)).reshape(*quad_center_scroll.shape)
        c_theta, _, _ = get_theta_and_radii(center_spiral[..., 1:], dr_per_winding)

        qi, qj = torch.where(quad_mask)
        w_target = target_winding_idx[qi, qj].to(torch.float32)
        ref_full = c_theta[qi, qj] + w_target * (2 * np.pi)

        vi = torch.stack([qi, qi, qi + 1, qi + 1], dim=-1)
        vj = torch.stack([qj, qj + 1, qj, qj + 1], dim=-1)
        vert_spiral = vertex_spiral[vi, vj]
        vert_scroll = patch_zyxs[vi, vj]
        vert_theta = v_theta[vi, vj]
        vert_w_raw = v_winding_raw[vi, vj].to(torch.float32)
        vert_full = vert_theta + vert_w_raw * (2 * np.pi)
        diff = vert_full - ref_full[:, None]
        vert_full_snapped = vert_full - torch.round(diff / (2 * np.pi)) * (2 * np.pi)

        u_coords = (vert_spiral[..., 0] - z0) / grid_spacing
        theta_step_per_quad = grid_spacing / ((w_target + 0.5) * dr)
        v_coords = (vert_full_snapped - w_target[:, None] * (2 * np.pi)) / theta_step_per_quad[:, None]

        all_quad_uvs.append(torch.stack([u_coords, v_coords], dim=-1))
        all_quad_scrolls.append(vert_scroll)
        all_target_w.append(target_winding_idx[qi, qj])

    if not all_quad_uvs:
        return

    quad_uvs = torch.cat(all_quad_uvs, dim=0)
    quad_scrolls = torch.cat(all_quad_scrolls, dim=0)
    quad_target_w = torch.cat(all_target_w, dim=0)
    Nq = quad_uvs.shape[0]

    tri_local = torch.tensor([[0, 1, 3], [0, 3, 2]], device=device, dtype=torch.long)
    quad_repeat = torch.arange(Nq, device=device).repeat_interleave(2)
    tri_local_flat = tri_local.unsqueeze(0).expand(Nq, -1, -1).reshape(-1, 3)
    tri_uvs = quad_uvs[quad_repeat[:, None].expand(-1, 3), tri_local_flat]
    tri_scrolls = quad_scrolls[quad_repeat[:, None].expand(-1, 3), tri_local_flat]
    tri_target_w = quad_target_w[quad_repeat]

    _rasterize_triangles_into_mesh(
        tri_uvs, tri_scrolls, tri_target_w,
        scroll_zyxs, winding_offsets_t, num_thetas_t,
    )


@torch.inference_mode()
def save_mesh(
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    unattached_pcl_strips,
    out_path,
    cfg,
    z_begin,
    z_end,
    voxel_size_um,
    get_or_build_unattached_pcl_flat,
    get_patch_satisfied_areas,
    tracks=(),
    run_tag=None,
    name='mesh',
    progress=None,
):
    (min_winding_idx, max_winding_idx), _, _ = compute_winding_range_and_input_extents(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        unattached_pcl_strips,
        cfg,
        z_begin,
        z_end,
        get_or_build_unattached_pcl_flat,
        authoritative_zyx_lines=tracks,
    )
    if cfg['shell_outer_winding_idx'] is not None:
        max_winding_idx = min(max_winding_idx, cfg['shell_outer_winding_idx'])
    print(f'save_mesh {name}: winding range [{min_winding_idx}, {max_winding_idx})')
    grid_spacing = cfg['output_step_size']
    z_margin = cfg['model_flow_bounds_z_margin']
    spiral_yxs_by_winding = get_spiral_yxs(max_winding_idx, dr_per_winding, grid_spacing, group_by_winding=True)
    num_thetas_by_winding = [len(yxs_for_winding) for yxs_for_winding in spiral_yxs_by_winding]
    spiral_yxs = torch.cat(spiral_yxs_by_winding, dim=0)
    z0 = z_begin - z_margin
    spiral_zs = torch.arange(z0, z_end + z_margin, grid_spacing, dtype=torch.float32, device=spiral_yxs.device)
    spiral_zyxs = torch.cat([spiral_zs[:, None, None].expand(-1, spiral_yxs.shape[0], 1), spiral_yxs[None, :, :].expand(spiral_zs.shape[0], -1, 2)], dim=-1)
    chunk = 65536
    flat_spiral_zyxs = spiral_zyxs.reshape(-1, 3)
    scroll_pieces = []
    transform_chunk_total = (
        flat_spiral_zyxs.shape[0] + chunk - 1) // chunk
    if progress is not None:
        progress.begin(
            'finalizing', 'Transforming final mesh',
            step=0, total_steps=transform_chunk_total, unit='chunks')
    for chunk_number, start in enumerate(
            range(0, flat_spiral_zyxs.shape[0], chunk), start=1):
        scroll_pieces.append(slice_to_spiral_transform.inv(flat_spiral_zyxs[start : start + chunk]))
        if progress is not None:
            progress.update(chunk_number)
    scroll_zyxs = torch.cat(scroll_pieces, dim=0).reshape(*spiral_zyxs.shape)

    out_of_roi = (scroll_zyxs[..., 0] < z_begin) | (scroll_zyxs[..., 0] >= z_end)
    scroll_zyxs[out_of_roi] = -1.0

    spliced_scroll_zyxs = scroll_zyxs.clone()
    # Splicing is deliberately more permissive than the reported satisfaction
    # metrics: it should accept a mostly aligned patch without relabelling that
    # patch as fully satisfied in the user-facing metrics.
    splicing_metrics_overrides = {
        'satisfaction_radius_tolerance': 0.495,
        'satisfaction_distance_tolerance': 12.0,
        'satisfied_patch_quad_fraction': 0.90,
    }
    satisfied_patches, _, _, _, boundary_satisfied_patches, target_winding_idx_per_patch = get_patch_satisfied_areas(
        slice_to_spiral_transform, dr_per_winding, patches,
        metrics_overrides=splicing_metrics_overrides,
    )
    _build_spliced_overlay(
        spliced_scroll_zyxs, num_thetas_by_winding, z0, grid_spacing,
        slice_to_spiral_transform, dr_per_winding,
        patches,
        satisfied_patches, boundary_satisfied_patches, target_winding_idx_per_patch,
    )

    step_size = grid_spacing
    tag_suffix = f'_{run_tag}' if run_tag else ''
    out_dir = f'{out_path}/meshes/{name}{tag_suffix}'
    os.makedirs(out_dir, exist_ok=True)
    output_total = 2 * len(num_thetas_by_winding)
    output_done = 0
    if progress is not None:
        progress.begin(
            'finalizing', 'Writing final mesh windings',
            step=0, total_steps=output_total, unit='windings')
    for uuid_suffix, variant_zyxs in [('', scroll_zyxs), ('_spliced', spliced_scroll_zyxs)]:
        offset = 0
        for winding_idx, num_thetas in enumerate(tqdm(
                num_thetas_by_winding,
                desc=f'saving winding patches ({name}{uuid_suffix})',
                disable=progress is not None)):
            if num_thetas >= 2 and winding_idx >= min_winding_idx:
                winding_slice = variant_zyxs[:, offset:offset + num_thetas]
                invalid_mask = (winding_slice == -1.0).all(dim=-1).cpu().numpy()
                winding_zyxs = winding_slice.cpu().numpy().astype(np.float32)
                winding_zyxs[invalid_mask] = -1.0
                save_tifxyz(
                    winding_zyxs,
                    out_dir,
                    uuid=f'w{winding_idx:03d}{uuid_suffix}{tag_suffix}',
                    step_size=step_size,
                    voxel_size_um=voxel_size_um,
                    source=f'fit_spiral {name}{uuid_suffix}',
                )
            offset += num_thetas
            output_done += 1
            if progress is not None:
                progress.update(
                    output_done, detail=f'winding {winding_idx}{uuid_suffix}')


@torch.inference_mode()
def save_combined_preview(
    slice_to_spiral_transform,
    dr_per_winding,
    patches,
    unattached_pcl_strips,
    generation_path,
    cfg,
    z_begin,
    z_end,
    voxel_size_um,
    get_or_build_unattached_pcl_flat,
    tracks=(),
    *,
    surface_id,
    progress=None,
):
    """Write the authoritative connected preview used by VC3D and Lasagna."""
    (_, derived_upper), _, _ = compute_winding_range_and_input_extents(
        slice_to_spiral_transform,
        dr_per_winding,
        patches,
        unattached_pcl_strips,
        cfg,
        z_begin,
        z_end,
        get_or_build_unattached_pcl_flat,
        authoritative_zyx_lines=tracks,
    )
    configured_outer = cfg.get('shell_outer_winding_idx')
    exclusive_upper = derived_upper
    if configured_outer is not None:
        configured_upper = int(configured_outer) + 1
        exclusive_upper = (configured_upper if derived_upper <= int(cfg['output_first_winding'])
                           else min(exclusive_upper, configured_upper))
    first_winding = 10
    last_winding = int(exclusive_upper) - 1
    if last_winding < first_winding:
        raise RuntimeError(
            f'No preview winding is at or above {first_winding}; derived last winding is {last_winding}'
        )

    grid_spacing = int(cfg['output_step_size'])
    z_margin = int(cfg['model_flow_bounds_z_margin'])
    spiral_yxs_by_winding = get_spiral_yxs(
        last_winding + 1,
        dr_per_winding,
        grid_spacing,
        group_by_winding=True,
    )
    z0 = z_begin - z_margin
    spiral_zs = torch.arange(
        z0,
        z_end + z_margin,
        grid_spacing,
        dtype=torch.float32,
        device=dr_per_winding.device,
    )
    winding_grids = {}
    total_windings = last_winding - first_winding + 1
    if progress is not None:
        progress.begin(
            'exporting_preview', 'Transforming preview windings',
            step=0, total_steps=total_windings, unit='windings')
    for winding_number, winding in enumerate(
            range(first_winding, last_winding + 1), start=1):
        yxs = spiral_yxs_by_winding[winding]
        if yxs.shape[0] < 2:
            raise RuntimeError(f'Preview winding {winding} has fewer than two theta samples')
        spiral = torch.cat([
            spiral_zs[:, None, None].expand(-1, yxs.shape[0], 1),
            yxs[None, :, :].expand(spiral_zs.shape[0], -1, 2),
        ], dim=-1)
        flat = spiral.reshape(-1, 3)
        pieces = []
        for start in range(0, flat.shape[0], 65536):
            pieces.append(slice_to_spiral_transform.inv(flat[start:start + 65536]))
        scroll = torch.cat(pieces, dim=0).reshape_as(spiral)
        outside = (scroll[..., 0] < z_begin) | (scroll[..., 0] >= z_end)
        scroll[outside] = -1.0
        winding_grids[winding] = scroll.cpu().numpy().astype(np.float32)
        if progress is not None:
            progress.update(winding_number, detail=f'winding {winding}')

    if progress is not None:
        progress.begin(
            'exporting_preview', 'Writing preview surface',
            detail=f'{total_windings} windings')
    manifest = save_combined_tifxyz(
        winding_grids,
        generation_path,
        surface_id,
        grid_spacing,
        voxel_size_um,
        source='fit_spiral interactive preview',
        first_winding=first_winding,
        cleanup_erosion_cells=3,
    )
    return manifest


def load_patches(patches_path, segment_id_filter=lambda s: True):
    patches = {}
    failed_count = 0
    for entry in sorted(os.listdir(patches_path)):
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


def erode_patch_valid_region(patch, num_cells):
    """Erode the patch's valid-vertex region inward by `num_cells` grid cells."""
    valid = patch.valid_vertex_mask.cpu().numpy()
    eroded = scipy.ndimage.binary_erosion(valid, iterations=num_cells, border_value=0)
    remove = valid & ~eroded
    if not remove.any():
        return True
    patch.zyxs[torch.from_numpy(remove)] = -1.0
    new_valid_vertex = torch.any(patch.zyxs != -1, dim=-1)
    new_valid_quad = (
        new_valid_vertex[:-1, :-1] & new_valid_vertex[1:, :-1]
        & new_valid_vertex[:-1, 1:] & new_valid_vertex[1:, 1:]
    )
    if not bool(new_valid_quad.any()):
        return False
    patch.__post_init__()
    return True


def _segmented_median_per_strip(ctx):
    # Segmented median: sort the flat values with a composite key
    # (strip_id-major, normalised_radii-minor) so values for each strip end
    # up contiguous and sorted within their range.
    normalised_radii = ctx['normalised_radii']
    strip_id = ctx['strip_id']
    starts = ctx['starts']
    lengths = ctx['lengths']
    S = ctx['S']
    device = ctx['device']
    if normalised_radii.numel() == 0:
        return torch.zeros(S, dtype=normalised_radii.dtype, device=device)

    val_min = normalised_radii.min().to(torch.float64)
    val_max = normalised_radii.max().to(torch.float64)
    val_range = (val_max - val_min) + 1.0
    composite = (
        strip_id.to(torch.float64) * val_range
        + (normalised_radii.to(torch.float64) - val_min)
    )
    order = torch.argsort(composite)
    sorted_norm = normalised_radii[order]
    median_indices = starts[:-1] + (lengths - 1) // 2
    return sorted_norm[median_indices]
