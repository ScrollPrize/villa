"""Grow and connect integer winding regions seeded by fiber components.

Run with --config configs/inconsistent_wind_config.json --output /tmp/growth.json.
The old diagnostic's input/linking settings are read, but its first-contact
filter, BFS, and minimum-edge repair are not run. See docs/winding_growth.md.
"""

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import heapq
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.csgraph import breadth_first_order, connected_components

import find_inconsistent_windings as fiw
from point_collection import (
    _build_surface_patch_index, _side_measurements, _window_hit_statistics,
    PatchLinkOptions, SIDE_FRONT, SIDE_BEHIND,
)
from spiral_helpers import load_fiber_point_collections, resolve_fiber_links, fiber_collection_hv_tag
from winding_linking import AnalysisLinkingConfig, model_inward_direction


@dataclass(frozen=True)
class GrowthConfig:
    min_contact_length: float = 10.0
    support_length: float = 40.0
    fit_scale: float = 1.0
    min_agreement: float = 1.0  # fixed: every eligible connection must agree
    overlaps: bool = True
    overlap_distance: float = 1.0
    overlap_min_area: float = 400.0
    overlap_normal_cos: float = 0.95
    coincidence_tolerance: float = 3.0
    coincidence_metric: str = 'xyz'
    coincidence_grid_size: int = 3
    coincidence_neighborhood_voxels: float | None = None
    seed_mode: str = 'largest_component'
    max_steps: int | None = None

    @property
    def overlap_search_distance(self):
        return max(self.overlap_distance, self.coincidence_tolerance) if self.coincidence_metric == 'normal_grid' else self.overlap_distance

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, dict):
            raise ValueError('growth must be a JSON object')
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'unknown growth settings: {sorted(unknown)}')
        cfg = cls(**data)
        for name in ('min_contact_length', 'support_length', 'fit_scale',
                     'overlap_distance', 'overlap_min_area'):
            value = getattr(cfg, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'growth.{name} must be finite and positive')
        for name in ('min_agreement', 'overlap_normal_cos'):
            value = getattr(cfg, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0.5 < value <= 1:
                raise ValueError(f'growth.{name} must be in (0.5, 1]')
        if cfg.min_agreement != 1.0:
            raise ValueError('growth.min_agreement must be 1.0: strict growth requires all connections to agree')
        if (type(cfg.coincidence_tolerance) not in (float, int)
                or not math.isfinite(cfg.coincidence_tolerance)
                or cfg.coincidence_tolerance <= 0):
            raise ValueError('growth.coincidence_tolerance must be finite and positive')
        if cfg.coincidence_metric not in ('xyz', 'normal_grid'):
            raise ValueError('growth.coincidence_metric must be xyz or normal_grid')
        if type(cfg.coincidence_grid_size) is not int or cfg.coincidence_grid_size not in (2, 3):
            raise ValueError('growth.coincidence_grid_size must be 2 or 3')
        if cfg.coincidence_neighborhood_voxels is not None:
            value = cfg.coincidence_neighborhood_voxels
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('growth.coincidence_neighborhood_voxels must be finite and positive')
            if cfg.coincidence_metric != 'normal_grid':
                raise ValueError('Physical neighborhoods require normal_grid metric')
        if cfg.coincidence_metric == 'normal_grid' and cfg.coincidence_tolerance != 3.:
            raise ValueError('normal_grid growth requires coincidence_tolerance=3 (stages 1, 2, 3)')
        if type(cfg.overlaps) is not bool:
            raise ValueError('growth.overlaps must be boolean')
        if cfg.seed_mode not in ('largest_component', 'all_components'):
            raise ValueError('growth.seed_mode must be largest_component or all_components')
        if cfg.max_steps is not None and (type(cfg.max_steps) is not int or cfg.max_steps < 1):
            raise ValueError('growth.max_steps must be a positive integer or null')
        return cfg


def branch_step(a, b):
    """Integer lift increment for fractional theta in [0, 1)."""
    difference = np.asarray(b) - np.asarray(a)
    return (difference < -0.5).astype(np.int64) - (difference > 0.5).astype(np.int64)


def tree_lifts(theta, predecessors):
    """Exact integer path sums on the same BFS tree, using pointer doubling.

    Roots point to themselves and contribute zero. Each round doubles the
    represented ancestor path; all additions remain int64, with no floating
    point solve or change to the chosen traversal tree.
    """
    parent = np.asarray(predecessors, dtype=np.int64).copy()
    roots = parent < 0
    parent[roots] = np.flatnonzero(roots)
    lift = branch_step(theta[parent], theta)
    while np.any(parent != parent[parent]):
        lift = lift + lift[parent]
        parent = parent[parent]
    return lift


class ThetaFrame:
    def __init__(self, transform, dr):
        self.transform, self.dr = transform, dr

    def __call__(self, points):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        values = []
        with torch.no_grad():
            for start in range(0, len(points), 262144):
                x = torch.as_tensor(points[start:start + 262144], device=self.dr.device)
                spiral = self.transform(x)
                theta, radius, _ = fiw.get_theta_and_radii(spiral[:, 1:], self.dr)
                if not bool(torch.isfinite(theta).all()) or bool((radius < 1e-5).any()):
                    raise ValueError('theta is undefined at a nonfinite point or on the axis')
                values.append(theta.double().cpu().numpy() / (2 * np.pi))
        return np.concatenate(values) if values else np.empty(0)


def solve_integer_graph(nodes, edges):
    """Solve b-a=delta; check every chord, including parallel/self edges."""
    adjacent = defaultdict(list)
    for a, b, delta, provenance in edges:
        adjacent[a].append((b, int(delta), provenance))
        adjacent[b].append((a, -int(delta), provenance))
    components = []
    seen = set()
    for root in sorted(nodes):
        if root in seen:
            continue
        offsets = {root: 0}
        queue = deque([root])
        while queue:
            a = queue.popleft()
            for b, delta, _ in sorted(adjacent[a], key=lambda e: (e[0], e[1])):
                if b not in offsets:
                    offsets[b] = offsets[a] + delta
                    queue.append(b)
        seen.update(offsets)
        conflicts = [dict(a=a, b=b, delta=int(d), residual=offsets[b] - offsets[a] - int(d),
                          provenance=p)
                     for a, b, d, p in edges if a in offsets
                     and offsets[b] - offsets[a] != d]
        components.append(dict(root=root, offsets=offsets, conflicts=conflicts))
    return components


def prepare_fibers(collections, frame, z_range):
    """Retain every contiguous in-domain run; never bridge a clipped-out gap."""
    links = resolve_fiber_links(collections)
    endpoint_ids = defaultdict(set)
    for link in links:
        endpoint_ids[link['a_coll']].add(link['a_point'])
        endpoint_ids[link['b_coll']].add(link['b_point'])
    members, point_location = {}, {}
    for cid, pcl in sorted(collections.items()):
        items = sorted(pcl['points'].items(), key=lambda item: int(item[0]))
        if not items:
            continue
        xyz = np.asarray([p['p'][::-1] for _, p in items], dtype=np.float32)
        valid = np.isfinite(xyz).all(1) & (xyz[:, 0] >= z_range[0]) & (xyz[:, 0] < z_range[1])
        indices = np.flatnonzero(valid)
        for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
            if len(run) < 2:
                continue
            ids = np.asarray([int(items[i][0]) for i in run])
            key = f'{pcl.get("source_file", pcl.get("name", cid))}:{ids[0]}'
            theta = frame(xyz[run])
            lift = np.r_[0, np.cumsum(branch_step(theta[:-1], theta[1:]))]
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz[run], axis=0), axis=1))]
            members[key] = dict(key=key, collection_id=cid, name=pcl.get('name'),
                                source_file=pcl.get('source_file'), ids=ids, zyx=xyz[run],
                                theta=theta, lift=lift, arc=arc, pcl=pcl)
            # Only explicit branch endpoints are queried below. Indexing every
            # dense point added millions of otherwise unused dictionary entries.
            for point_id in endpoint_ids[cid]:
                j = int(np.searchsorted(ids, point_id))
                if j < len(ids) and ids[j] == point_id:
                    point_location[cid, int(point_id)] = (key, j)
    edges, excluded = [], []
    for link in links:
        a = point_location.get((link['a_coll'], link['a_point']))
        b = point_location.get((link['b_coll'], link['b_point']))
        if a is None or b is None:
            excluded.append(link)
            continue
        ma, mb = members[a[0]], members[b[0]]
        delta = (ma['lift'][a[1]] + branch_step(ma['theta'][a[1]], mb['theta'][b[1]])
                 - mb['lift'][b[1]])
        edges.append((a[0], b[0], int(delta), link))
    components = solve_integer_graph(members, edges)
    for component in components:
        component['key'] = 'fiber:' + component['root']
        component['length'] = float(sum(members[k]['arc'][-1] for k in component['offsets']))
        for key, offset in component['offsets'].items():
            members[key]['lift'] += offset
            members[key]['node'] = component['key']
            members[key]['consistent'] = not component['conflicts']
    return members, components, excluded


class PatchFields:
    """Lazy local integer fields, with independent gauges on disconnected quads."""
    def __init__(self, patches, frame, z_range):
        self.patches, self.frame, self.z_range = patches, frame, z_range
        self.fields = {}

    def get(self, pid):
        if pid in self.fields:
            return self.fields[pid]
        from types import SimpleNamespace
        patch = self.patches[pid]
        grid = patch.zyxs.cpu().numpy()
        valid_vertex = np.isfinite(grid).all(2) & (grid[..., 0] >= self.z_range[0]) & (grid[..., 0] < self.z_range[1])
        valid = (patch.valid_quad_mask.cpu().numpy() & valid_vertex[:-1, :-1]
                 & valid_vertex[1:, :-1] & valid_vertex[:-1, 1:] & valid_vertex[1:, 1:])
        proxy = SimpleNamespace(valid_quad_mask=torch.from_numpy(valid), ij_to_zyx=patch.ij_to_zyx)
        graph = fiw.cop.build_patch_graph(proxy, 0.)
        xyz = graph.quad_center_xyz[:, ::-1].copy()
        theta = self.frame(xyz)
        _, labels = connected_components(graph.csr, directed=False)
        predecessors = np.full(graph.n, -9999, dtype=np.int64)
        coo = graph.csr.tocoo()
        steps = branch_step(theta[coo.row], theta[coo.col])
        for label in np.unique(labels):
            root = int(np.flatnonzero(labels == label)[0])
            order, component_parents = breadth_first_order(graph.csr, root, directed=False)
            predecessors[order] = component_parents[order]
        lift = tree_lifts(theta, predecessors)
        bad = (lift[coo.col] - lift[coo.row]) != steps
        bad_labels = set(int(x) for x in labels[coo.row[bad]])
        # Quad areas and normals are geometric support, never theta/radius votes.
        r, c = graph.node_qi, graph.node_qj
        a, b, d, e = grid[r, c], grid[r + 1, c], grid[r, c + 1], grid[r + 1, c + 1]
        n1, n2 = np.cross(b - a, d - a), np.cross(d - e, b - e)
        area = (np.linalg.norm(n1, axis=1) + np.linalg.norm(n2, axis=1)) / 2
        normal = n1 + n2
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
        field = dict(graph=graph, theta=theta, lift=lift, labels=labels, bad_labels=bad_labels,
                     xyz=xyz, area=area, normal=normal, valid=valid,
                     conflicts=int(bad.sum() // 2))
        # Routing is finished and every edge has been audited. Retain source
        # indices for projection/export, but not the eight-neighbor matrix or
        # distance transform for every patch in an 85k-patch catalog.
        graph.csr = None
        graph.dt = None
        self.fields[pid] = field
        return field

    @staticmethod
    def key(pid, label):
        return f'patch:{pid}:{int(label)}'

    def _locate_geometry(self, pid, ijs):
        field = self.get(pid)
        ijs = np.asarray(ijs, dtype=np.float32).reshape(-1, 2)
        xyz, valid = fiw._bilinear_ij_to_zyx_np(
            self.patches[pid].zyxs.cpu().numpy(), field['valid'], ijs)
        selected = np.flatnonzero(valid)
        qi, qj = np.floor(ijs[selected]).astype(int).T
        nodes = field['graph'].ids[qi, qj]
        ok = np.asarray([int(field['labels'][n]) not in field['bad_labels'] for n in nodes], dtype=bool)
        selected, nodes = selected[ok], nodes[ok]
        return field, selected, nodes, xyz[selected]

    def locate(self, pid, ijs):
        field, selected, nodes, xyz = self._locate_geometry(pid, ijs)
        theta = self.frame(xyz)
        lift = field['lift'][nodes] + branch_step(field['theta'][nodes], theta)
        return selected, nodes, xyz, theta, lift

    def locate_many(self, queries):
        """Evaluate the pointwise theta transform once across overlap targets."""
        located = [self._locate_geometry(pid, ijs) for pid, ijs in queries]
        if not located:
            return []
        theta = self.frame(np.concatenate([item[3] for item in located]))
        result, start = [], 0
        for field, selected, nodes, xyz in located:
            th = theta[start:start + len(nodes)]
            start += len(nodes)
            lift = field['lift'][nodes] + branch_step(field['theta'][nodes], th)
            result.append((selected, nodes, xyz, th, lift))
        return result


@dataclass
class Evidence:
    a: str
    b: str
    delta: int                 # offset[b] - offset[a]
    tokens_a: tuple
    tokens_b: tuple
    weights: tuple
    detail: dict


def hypotheses(node, active, edges, incident):
    """Cap repeated support on the same physical source interval/target quad."""
    groups = {}
    for eid in incident[node]:
        e = edges[eid]
        other = e.b if node == e.a else e.a
        if other not in active:
            continue
        offset = active[other] + (-e.delta if node == e.a else e.delta)
        g = groups.setdefault(offset, dict(tokens={}, edges=[], providers=set()))
        tokens = e.tokens_a if node == e.a else e.tokens_b
        for token, weight in zip(tokens, e.weights):
            g['tokens'][token] = max(g['tokens'].get(token, 0.), weight)
        g['edges'].append(eid)
        g['providers'].add(other)
    result = [dict(offset=int(offset), support=math.fsum(g['tokens'].values()),
                   edges=sorted(g['edges']), providers=sorted(g['providers']))
              for offset, g in groups.items()]
    result.sort(key=lambda g: (-g['support'], g['offset']))
    total = sum(g['support'] for g in result)
    for g in result:
        g['agreement'] = g['support'] / total if total else 0.
    return result


class IncrementalHypotheses:
    """Fold each newly active connection once; keep exact per-token maxima.

    Recompute changed sums with fsum, never by subtracting/adding rounded
    totals. Snapshots own their lists so later additions cannot alter history.
    """
    def __init__(self):
        self.groups = {}

    def add(self, node, other_offset, eid, edge):
        offset = other_offset + (-edge.delta if node == edge.a else edge.delta)
        group = self.groups.setdefault(offset, dict(tokens={}, edges=[], providers=set(), support=None))
        tokens = edge.tokens_a if node == edge.a else edge.tokens_b
        for token, weight in zip(tokens, edge.weights):
            previous = group['tokens'].get(token, 0.)
            if token not in group['tokens'] or weight > previous:
                group['tokens'][token] = max(previous, weight)
                group['support'] = None
        group['edges'].append(eid)
        group['providers'].add(edge.b if node == edge.a else edge.a)

    @staticmethod
    def support(group):
        if group['support'] is None:
            group['support'] = math.fsum(group['tokens'].values())
        return group['support']

    def snapshot(self):
        result = [dict(offset=int(offset), support=self.support(group),
                       edges=sorted(group['edges']), providers=sorted(group['providers']))
                  for offset, group in self.groups.items()]
        result.sort(key=lambda g: (-g['support'], g['offset']))
        total = sum(g['support'] for g in result)
        for group in result:
            group['agreement'] = group['support'] / total if total else 0.
        return result


def audit_accepted_topology(seed, active, edges):
    """Certify connectivity and exact path independence on the accepted graph."""
    adjacent = defaultdict(set)
    edge_count = 0
    for eid, edge in enumerate(edges):
        if edge.a not in active or edge.b not in active:
            continue
        residual = active[edge.b] - active[edge.a] - edge.delta
        if residual != 0:
            raise RuntimeError(f'accepted topology is inconsistent at edge {eid}: residual {residual}')
        adjacent[edge.a].add(edge.b)
        adjacent[edge.b].add(edge.a)
        edge_count += 1
    seen, queue = {seed}, [seed]
    while queue:
        for neighbor in adjacent[queue.pop()] - seen:
            seen.add(neighbor)
            queue.append(neighbor)
    if seen != set(active):
        raise RuntimeError('accepted topology has nodes disconnected from the seed')
    return dict(connected=True, path_independent=True, nonzero_residuals=0,
                num_nodes=len(active), num_edges=edge_count,
                cycle_rank=edge_count - len(active) + 1,
                scope='all eligible sampled connections between accepted components')


def conflict_witness(node, proposals, parents, edges):
    """An explicit nonzero closed walk through two incompatible contacts."""
    if len(proposals) < 2:
        return None
    first, second = proposals[0]['edges'][0], proposals[1]['edges'][0]

    def other(eid):
        edge = edges[eid]
        return edge.b if edge.a == node else edge.a

    def route_to_root(start):
        route = [start]
        while parents[route[-1]] is not None:
            route.append(parents[route[-1]][0])
        return route

    a, b = other(first), other(second)
    up_a, up_b = route_to_root(a), route_to_root(b)
    b_ancestors = set(up_b)
    lca = next(n for n in up_a if n in b_ancestors)
    steps = []

    def append(eid, start, end):
        e = edges[eid]
        steps.append(dict(edge=eid, from_node=start, to_node=end,
                          delta=int(e.delta if e.a == start else -e.delta)))

    append(first, node, a)
    for child in up_a[:up_a.index(lca)]:
        parent, eid = parents[child]
        append(eid, child, parent)
    for child in reversed(up_b[:up_b.index(lca)]):
        parent, eid = parents[child]
        append(eid, parent, child)
    append(second, b, node)
    residual = sum(step['delta'] for step in steps)
    if residual == 0:
        raise RuntimeError('conflict witness unexpectedly closes')
    return dict(steps=steps, winding_residual=residual)


def grow(seed, initial_edges, cfg, expand=None, progress=None, coincidence=None, *,
         initial_offsets=None, initial_expanded=(), on_accept=None, excluded_nodes=()):
    """Grow a connected graph with exact integer transport on every accepted edge.

    Prepare each candidate's complete incident evidence before committing it.
    All eligible connections to active nodes must propose one identical offset;
    even a zero-score contradiction blocks the addition. Scores only rank
    consistent candidates and require enough geometric support to connect them.

    ``expand(node)`` must return only edges incident to that node and finish its
    incident discovery before returning. Both endpoints are prepared before an
    edge becomes traversable, so adding a node cannot invalidate prior topology.
    """
    edges, incident = [], defaultdict(list)
    active = dict(initial_offsets) if initial_offsets is not None else {seed: 0}
    if seed not in active or active[seed] != 0:
        raise ValueError('Resumed offsets must contain the original zero-gauge seed')
    from numbers import Integral
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in active.values()):
        raise ValueError('Resumed offsets must be exact integers')
    parents = {seed: None}
    excluded_nodes = set(excluded_nodes)
    if excluded_nodes & active.keys():
        raise ValueError('Excluded nodes cannot be active')
    expanded, history, deferred = set(initial_expanded), [], set(excluded_nodes)
    geometric_conflicts = {}
    stages = (1., 2., 3.) if cfg.coincidence_metric == 'normal_grid' else (cfg.coincidence_tolerance,)
    if len(stages) > 1 and coincidence is None:
        raise ValueError('Staged normal-grid growth requires a coincidence guard')
    stage = 0
    if coincidence is not None and len(stages) > 1:
        coincidence.tolerance = stages[stage]
    stage_counts = []
    stage_start = 0
    dirty, frontier = set(), {}
    queue, accumulators = [], {}
    revision = 0
    bad_self_edges = defaultdict(list)

    def connect(node, other, eid):
        if node in active or node in deferred:
            return
        dirty.add(node)
        if other in active:
            if node not in accumulators:
                accumulators[node] = IncrementalHypotheses()
            accumulators[node].add(node, active[other], eid, edges[eid])

    def add(new_edges):
        for e in new_edges:
            eid = len(edges)
            edges.append(e)
            incident[e.a].append(eid)
            if e.b != e.a:
                incident[e.b].append(eid)
            elif e.delta != 0:
                bad_self_edges[e.a].append(eid)
            if e.a in active and e.b in active and active[e.b] - active[e.a] != e.delta:
                raise RuntimeError(f'evidence discovery contradicts already accepted topology at edge {eid}')
            dirty.update((e.a, e.b))
            connect(e.a, e.b, eid)
            if e.b != e.a:
                connect(e.b, e.a, eid)

    def changed(node):
        for eid in incident[node]:
            e = edges[eid]
            connect(e.b if node == e.a else e.a, node, eid)

    def prepare(node):
        if expand:
            discovered = list(expand(node))
            if any(node not in (e.a, e.b) for e in discovered):
                raise ValueError('expand(node) must return only incident evidence')
            add(discovered)
        expanded.add(node)
        dirty.add(node)

    add(initial_edges)
    # Reconstruct a real spanning tree, so contradiction witnesses can traverse
    # the resumed region without inventing zero-weight seed shortcuts.
    audit_accepted_topology(seed, active, edges)
    todo = deque([seed])
    while todo:
        node = todo.popleft()
        for eid in incident[node]:
            edge = edges[eid]
            other = edge.b if edge.a == node else edge.a
            if other in active and other not in parents:
                parents[other] = (node, eid)
                todo.append(other)
    for node in sorted(active):
        if node not in expanded:
            prepare(node)
    steps = 0
    while True:
        if cfg.max_steps is not None and steps >= cfg.max_steps:
            break
        for node in sorted(dirty.difference(active)):
            frontier.pop(node, None)
            # Active offsets never change, so contradictory candidates cannot
            # recover. The final pending report still checks all their edges.
            if node in deferred:
                continue
            accumulator = accumulators.get(node)
            if bad_self_edges[node]:
                proposals = accumulator.snapshot() if accumulator else []
                deferred.add(node)
                history.append(dict(action='defer', node=node, reason='internal_cycle',
                                    edges=bad_self_edges[node], hypotheses=proposals))
            elif accumulator and len(accumulator.groups) == 1:
                support = accumulator.support(next(iter(accumulator.groups.values())))
                if support >= 1:
                    revision += 1
                    entry = (-support, node, revision)
                    frontier[node] = entry
                    heapq.heappush(queue, entry)
            elif accumulator and len(accumulator.groups) > 1:
                proposals = accumulator.snapshot()
                deferred.add(node)
                history.append(dict(action='defer', node=node, reason='conflicting_connections',
                                    hypotheses=proposals,
                                    conflict_cycle=conflict_witness(node, proposals, parents, edges)))
            if node in deferred:
                accumulators.pop(node, None)
        dirty.clear()
        if not frontier:
            stage_counts.append(dict(tolerance=stages[stage], additions=steps-stage_start, exhausted=True))
            if stage+1 == len(stages):
                break
            stage += 1
            stage_start = steps
            coincidence.tolerance = stages[stage]
            history.append(dict(action='advance_tolerance', tolerance=stages[stage]))
            if progress:
                print(f'Normal thickness growth: exhausted {stages[stage-1]:g} voxel pass; '
                      f'retrying at {stages[stage]:g} voxels', flush=True)
            # Rebuild using ALL currently accepted neighbors. Edges discovered
            # while this candidate was deferred may introduce contradictions.
            retry = sorted(geometric_conflicts)
            for node in retry:
                deferred.discard(node)
                geometric_conflicts.pop(node)
                accumulators.pop(node, None)
                for eid in incident[node]:
                    e = edges[eid]
                    other = e.b if node == e.a else e.a
                    connect(node, other, eid)
            continue
        # Bound stale heap storage while keeping exactly the same score/name
        # ordering as the former full frontier scan.
        if len(queue) > max(1024, 2 * len(frontier)):
            queue = list(frontier.values())
            heapq.heapify(queue)
        while True:
            entry = heapq.heappop(queue)
            _, node, _ = entry
            if frontier.get(node) == entry:
                break
        frontier.pop(node)
        if node not in expanded:
            prepare(node)
            # Newly discovered overlaps may change feasibility or priority.
            # Re-evaluate the frontier before committing anything.
            continue
        proposals = accumulators.pop(node).snapshot()
        best = proposals[0]
        if coincidence is not None:
            conflict = coincidence.check(node, best['offset'])
            if conflict is not None:
                deferred.add(node)
                geometric_conflicts[node] = conflict
                history.append(dict(action='defer', node=node, reason='geometric_conflict',
                                    hypotheses=proposals, geometric_conflict=conflict,
                                    tolerance=stages[stage]))
                if progress:
                    print(f'geometric conflict: {node} against {conflict.get("other", "itself")}: '
                          f'{conflict.get("max_distance", conflict.get("detail"))}', flush=True)
                continue
        checked_edges = []
        for eid in incident[node]:
            edge = edges[eid]
            other = edge.b if edge.a == node else edge.a
            if other != node and other not in active:
                continue
            a = best['offset'] if edge.a == node else active[edge.a]
            b = best['offset'] if edge.b == node else active[edge.b]
            if b - a != edge.delta:
                raise RuntimeError(f'pre-commit consistency check failed at edge {eid}')
            checked_edges.append(eid)
        active[node] = best['offset']
        if coincidence is not None:
            coincidence.accept(node, best['offset'])
        parent_edge = next(eid for eid in checked_edges if edges[eid].a != edges[eid].b)
        parent = edges[parent_edge].b if edges[parent_edge].a == node else edges[parent_edge].a
        parents[node] = (parent, parent_edge)
        changed(node)
        history.append(dict(action='accept', node=node, offset=best['offset'],
                            hypotheses=proposals, checked_edges=checked_edges,
                            parent=parent, parent_edge=parent_edge, tolerance=stages[stage]))
        if on_accept is not None:
            on_accept(node, best, edges)
        steps += 1
        if progress:
            progress(steps, node, best)
    if len(stage_counts) <= stage:
        stage_counts.append(dict(tolerance=stages[stage], additions=steps-stage_start, exhausted=False))
    pending = []
    for node in sorted(set(incident) - set(active)):
        proposals = hypotheses(node, active, edges, incident)
        reason = ('quarantined' if node in excluded_nodes else
                  'geometric_conflict' if node in geometric_conflicts else
                  'internal_cycle' if bad_self_edges[node] else 'unreached' if not proposals
                  else 'conflicting_connections' if len(proposals) > 1
                  else 'insufficient_support' if proposals[0]['support'] < 1 else 'step_limit')
        pending.append(dict(node=node, reason=reason, hypotheses=proposals,
                            geometric_conflict=geometric_conflicts.get(node),
                            internally_conflicting_edges=bad_self_edges[node],
                            conflict_cycle=conflict_witness(node, proposals, parents, edges),
                            incident_discovery_complete=node in expanded))
    topology = audit_accepted_topology(seed, active, edges)
    return dict(seed=seed, offsets=active, history=history, pending=pending,
                first_ambiguity=next((h for h in history if h['action'] == 'defer'), None),
                conflicts=[], geometric_conflicts=geometric_conflicts,
                geometric_consistency=coincidence.report() if coincidence is not None else None,
                topology=topology, steps=steps, thickness_stages=stage_counts,
                stopped_at_limit=cfg.max_steps is not None and steps >= cfg.max_steps), edges


class GeometryEvidence:
    def __init__(self, patches, members, frame, fields, linking, cfg, inward, fit_config):
        built = _build_surface_patch_index(patches, max(linking.distance, cfg.overlap_search_distance))
        if built is None:
            raise RuntimeError('growth requires the existing vc_spiral.surface_index backend')
        self.index, self.surface_ids = built
        self.surface_index = {pid: i for i, pid in enumerate(self.surface_ids)}
        self.patches, self.members, self.frame, self.fields = patches, members, frame, fields
        self.linking, self.cfg, self.inward, self.fit_config = linking, cfg, inward, fit_config
        self.member_nodes = defaultdict(list)
        for key, member in members.items():
            if member['consistent']:
                self.member_nodes[member['node']].append(key)
        self.patch_nodes = {}
        self.queried_fibers = set()
        self.overlap_pairs = set()
        self.counts = Counter()

    def register_patch(self, pid, label):
        key = self.fields.key(pid, label)
        self.patch_nodes[key] = (pid, int(label))
        return key

    def contacts(self, member):
        """All eligible patch alternatives; no per-point winning patch."""
        cfg, linking = self.cfg, self.linking
        xyz = member['zyx']
        # Transport uses the full dense member even when contact queries stride.
        positions = np.unique(np.r_[np.arange(0, len(xyz), linking.fiber_stride), len(xyz) - 1])
        query = xyz[positions]
        offsets, surfaces, distances, ijs = self.index.locate_all_xyz_batch(
            np.ascontiguousarray(query[:, ::-1]), linking.distance)
        pi = np.repeat(np.arange(len(query)), np.diff(offsets))
        self.counts['candidate_point_patch_hits'] += len(pi)
        if not len(pi):
            return []
        tag = fiber_collection_hv_tag(member['pcl'],
            min_z_fraction=self.fit_config['pcl_vertical_fiber_min_z_fraction'],
            min_auto_certainty=self.fit_config['pcl_vertical_fiber_min_auto_certainty'])
        fraction = linking.vertical_max_behind_fraction if tag == 'V' else linking.horizontal_max_front_fraction if tag == 'H' else None
        wrong = np.zeros(len(pi), dtype=bool)
        if fraction is not None:
            options = PatchLinkOptions(side_rules={0: SIDE_FRONT if tag == 'V' else SIDE_BEHIND},
                inward_direction=self.inward, side_margin=linking.side_margin_voxels,
                max_wrong_side_fractions={0: fraction})
            wrong, defined = _side_measurements(query, pi, surfaces, ijs,
                lambda s: self.patches[self.surface_ids[s]], options.side_rule(0), options,
                interpolate=self.index.interpolate_ij)
            pi, surfaces, distances, ijs, wrong = [x[defined] for x in (pi, surfaces, distances, ijs, wrong)]
            self.counts['undefined_side_hits'] += int((~defined).sum())
        support, _, wrong_count = _window_hit_statistics(
            pi, surfaces, distances, len(query), linking.window_points // 2, wrong)
        keep = support >= linking.min_points
        if fraction is not None:
            keep &= wrong_count <= fraction * support
        self.counts['window_rejected_hits'] += int((~keep).sum())
        pi, surfaces, distances, ijs = [x[keep] for x in (pi, surfaces, distances, ijs)]
        evidence = []
        for surface in np.unique(surfaces):
            select = np.flatnonzero(surfaces == surface)
            select = select[np.argsort(pi[select], kind='stable')]
            pid = self.surface_ids[surface]
            valid, nodes, feet, theta, lift = self.fields.locate(pid, ijs[select])
            self.counts['invalid_or_inconsistent_patch_hits'] += len(select) - len(valid)
            select = select[valid]
            dense = positions[pi[select]]
            distance = np.linalg.norm(xyz[dense] - feet, axis=1)
            accurate = distance <= linking.distance
            self.counts['bilinear_distance_rejected_hits'] += int((~accurate).sum())
            select, dense, nodes, theta, lift, distance = [x[accurate] for x in
                                                         (select, dense, nodes, theta, lift, distance)]
            field = self.fields.get(pid)
            labels = field['labels'][nodes]
            deltas = member['lift'][dense] + branch_step(member['theta'][dense], theta) - lift
            # Eligibility must depend only on geometry, never on whether an
            # offset agrees with its neighbors. First find supported intervals;
            # then retain *every* offset within each eligible interval.
            split = np.flatnonzero((np.diff(pi[select]) != 1) | (np.diff(labels) != 0)) + 1
            for interval in np.split(np.arange(len(select)), split):
                if len(interval) < 2:
                    continue
                interval_length = float(member['arc'][dense[interval[-1]]] - member['arc'][dense[interval[0]]])
                if interval_length < cfg.min_contact_length:
                    self.counts['short_contact_intervals'] += 1
                    continue
                modes = np.flatnonzero(np.diff(deltas[interval]) != 0) + 1
                for run in np.split(interval, modes):
                    dense_run = dense[run]
                    lengths = np.diff(member['arc'][dense_run])
                    length = float(lengths.sum())
                    run_distance = distance[run]
                    weights = lengths / cfg.support_length / (1 + ((run_distance[:-1] + run_distance[1:]) / (2 * cfg.fit_scale)) ** 2)
                    tokens = tuple(f'{member["key"]}:{a}:{b}' for a, b in zip(dense_run[:-1], dense_run[1:]))
                    # A single sample earns no length, but its constraint must
                    # still veto a different offset (even with weight zero).
                    patch_node = self.register_patch(pid, labels[run[0]])
                    detail = dict(kind='contact', member=member['key'], source_file=member['source_file'],
                        patch_id=pid, from_point_id=int(member['ids'][dense_run[0]]),
                        to_point_id=int(member['ids'][dense_run[-1]]),
                        from_ij=ijs[select[run[0]]].tolist(), to_ij=ijs[select[run[-1]]].tolist(),
                        length=length, eligible_interval_length=interval_length,
                        rms_distance=float(np.sqrt(np.mean(run_distance ** 2))))
                    evidence.append(Evidence(member['node'], patch_node, int(deltas[run[0]]),
                                             tokens, tokens, tuple(weights.tolist()), detail))
                    self.counts['contact_intervals'] += 1
                    if length < cfg.min_contact_length:
                        self.counts['short_constraints_in_eligible_intervals'] += 1
        return evidence

    def overlaps(self, pid, label):
        """Reciprocal, tangent-compatible overlap; no geometry merging."""
        if not self.cfg.overlaps:
            return []
        cfg = self.cfg
        a = self.fields.get(pid)
        src = np.flatnonzero(a['labels'] == label)
        if not len(src):
            return []
        offsets, surfaces, distances, ijs = self.index.locate_all_xyz_batch(
            np.ascontiguousarray(a['xyz'][src, ::-1]), cfg.overlap_search_distance)
        query_ids = np.repeat(np.arange(len(src)), np.diff(offsets))
        result, targets = [], []
        for surface in np.unique(surfaces):
            other = self.surface_ids[surface]
            if other == pid:
                continue
            # Process per source component: disconnected pieces are independent.
            pair_key = (pid, label, other)
            if pair_key in self.overlap_pairs:
                continue
            self.overlap_pairs.add(pair_key)
            select = np.flatnonzero(surfaces == surface)
            # Any retained overlap area is bounded by the area of these source
            # quads, even before checking target geometry, normals or reciprocity.
            # Skip impossible overlaps without building a target potential field.
            possible_source = src[query_ids[select]]
            if float(a['area'][np.unique(possible_source)].sum()) < cfg.overlap_min_area:
                self.counts['overlap_area_bound_rejections'] += 1
                continue
            targets.append((other, select))
        located = self.fields.locate_many([(other, ijs[select]) for other, select in targets])
        nonempty = [i for i, item in enumerate(located) if len(item[0])]
        if not nonempty:
            return result
        # Reciprocal checks all query the same source surface. Batch them so
        # native point queries can share one index and parallelize large work.
        all_feet = np.concatenate([located[i][2] for i in nonempty])
        all_na = np.concatenate([src[query_ids[targets[i][1][located[i][0]]]] for i in nonempty])
        reverse_index, _ = _build_surface_patch_index(
            {pid: self.patches[pid]}, max(self.linking.distance, cfg.overlap_search_distance))
        reverse_offsets, _, _, reverse_ij = reverse_index.locate_all_xyz_batch(
            np.ascontiguousarray(all_feet[:, ::-1]), cfg.overlap_search_distance)
        reverse_rows = np.repeat(np.arange(len(all_feet)), np.diff(reverse_offsets))
        reverse_xyz, reverse_valid = fiw._bilinear_ij_to_zyx_np(
            self.patches[pid].zyxs.cpu().numpy(), a['valid'], reverse_ij)
        all_reciprocal = np.zeros(len(all_feet), dtype=bool)
        all_reciprocal[reverse_rows] = reverse_valid & (
            np.linalg.norm(reverse_xyz - a['xyz'][all_na[reverse_rows]], axis=1) <= cfg.overlap_search_distance)
        start = 0
        for (other, select), (valid, nb, feet, theta, lift) in zip(targets, located):
            if not len(valid):
                continue
            pair = tuple(sorted((pid, other)))
            na = src[query_ids[select[valid]]]
            b = self.fields.get(other)
            close = np.linalg.norm(a['xyz'][na] - feet, axis=1) <= cfg.overlap_search_distance
            aligned = np.abs(np.sum(a['normal'][na] * b['normal'][nb], axis=1)) >= cfg.overlap_normal_cos
            reciprocal = all_reciprocal[start:start + len(feet)]
            start += len(feet)
            keep = close & aligned & reciprocal
            na, nb, theta, lift, feet = [x[keep] for x in (na, nb, theta, lift, feet)]
            delta = a['lift'][na] + branch_step(a['theta'][na], theta) - lift
            for target_label in np.unique(b['labels'][nb]):
                mask = b['labels'][nb] == target_label
                # Qualify the geometric overlap as a whole. Small contradictory
                # offset groups must not disappear behind the area threshold.
                eligible_area = min(float(a['area'][np.unique(na[mask])].sum()),
                                    float(b['area'][np.unique(nb[mask])].sum()))
                if eligible_area < cfg.overlap_min_area:
                    continue
                for d in np.unique(delta[mask]):
                    selected = np.flatnonzero(mask & (delta == d))
                    # Count destination quads once too, so many-to-one matches
                    # cannot make a tiny contact look like a broad overlap.
                    area = min(float(a['area'][np.unique(na[selected])].sum()),
                               float(b['area'][np.unique(nb[selected])].sum()))
                    dist = np.linalg.norm(a['xyz'][na[selected]] - feet[selected], axis=1)
                    weights = a['area'][na[selected]] / cfg.overlap_min_area / (1 + (dist / cfg.fit_scale) ** 2)
                    scale = area / max(float(a['area'][na[selected]].sum()), 1e-12)
                    weights *= scale
                    ka, kb = self.register_patch(pid, label), self.register_patch(other, target_label)
                    result.append(Evidence(ka, kb, int(d),
                        tuple(f'quad:{pid}:{n}' for n in na[selected]),
                        tuple(f'quad:{other}:{n}' for n in nb[selected]), tuple(weights.tolist()),
                        dict(kind='overlap', patches=list(pair), area=area, eligible_overlap_area=eligible_area,
                             rms_distance=float(np.sqrt(np.mean(dist ** 2))))))
        return result

    def expand(self, node):
        result = []
        if node in self.member_nodes:
            for key in sorted(self.member_nodes[node]):
                if key not in self.queried_fibers:
                    self.queried_fibers.add(key)
                    result.extend(self.contacts(self.members[key]))
        elif node in self.patch_nodes:
            pid, label = self.patch_nodes[node]
            result.extend(self.overlaps(pid, label))
        return result


def save_fields(path, members, fields, offsets):
    arrays, catalog = {}, []
    for member in members.values():
        if member['node'] not in offsets:
            continue
        prefix = f'f{len(catalog)}'
        arrays[prefix + '_point_ids'] = member['ids']
        arrays[prefix + '_theta_fraction'] = member['theta']
        arrays[prefix + '_turns'] = member['lift'] + offsets[member['node']]
        catalog.append(dict(prefix=prefix, kind='fiber', member=member['key'],
                            source_file=member['source_file'], node=member['node']))
    for pid, field in fields.fields.items():
        for label in np.unique(field['labels']):
            node = fields.key(pid, label)
            if node not in offsets:
                continue
            select = np.flatnonzero(field['labels'] == label)
            prefix = f'p{len(catalog)}'
            arrays[prefix + '_ij'] = np.column_stack((field['graph'].node_qi[select] + .5,
                                                       field['graph'].node_qj[select] + .5))
            arrays[prefix + '_theta_fraction'] = field['theta'][select]
            arrays[prefix + '_turns'] = field['lift'][select] + offsets[node]
            catalog.append(dict(prefix=prefix, kind='patch', patch_id=pid, node=node))
    np.savez_compressed(path, **arrays)
    return catalog


def main(argv=None):
    from winding_workspace import export_workspace, workspace_settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path('winding_growth.json'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--no-overlaps', action='store_true')
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed-mode', choices=('largest_component', 'all_components'),
                       help='Override growth.seed_mode from the configuration')
    seeds.add_argument('--all-fiber-components', action='store_true',
                       help='Alias for --seed-mode all_components')
    args = parser.parse_args(argv)
    try:
        raw = json.loads(args.config.read_text())
        if not isinstance(raw, dict):
            raise ValueError('configuration must be a JSON object')
        for name in ('checkpoint', 'patches_dir', 'z_range'):
            if name not in raw:
                raise ValueError(f'missing required input setting {name!r}')
        settings = raw.get('growth', {})
        GrowthConfig.from_json(settings)
        settings = dict(settings)
        if args.max_steps is not None:
            settings['max_steps'] = args.max_steps
        if args.no_overlaps:
            settings['overlaps'] = False
        if args.seed_mode is not None:
            settings['seed_mode'] = args.seed_mode
        if args.all_fiber_components:
            settings['seed_mode'] = 'all_components'
        cfg = GrowthConfig.from_json(settings)
        workspace_settings(raw.get('workspace', {}))
        linking = AnalysisLinkingConfig.from_json(raw.get('linking', {}))
        if linking.patch_stride != 1 or linking.strips:
            raise ValueError('growth requires linking.patch_stride=1 and linking.strips=false')
        z_range = fiw.parse_z_range(raw['z_range'])
    except (OSError, ValueError, KeyError, TypeError, fiw.click.BadParameter) as exc:
        parser.error(str(exc))
    torch.set_grad_enabled(False)
    from checkpoint_io import load_checkpoint_cpu
    checkpoint = load_checkpoint_cpu(raw['checkpoint'])
    umbilicus = fiw.resolve_umbilicus_path(raw['patches_dir'], raw.get('umbilicus'))
    fibers_path = fiw.resolve_fibers_path(raw['patches_dir'], raw.get('fibers', ''))
    fit_cfg, scroll, paths, model_lo, model_hi = fiw.build_fit_inputs(
        checkpoint, raw['patches_dir'], [], *z_range, umbilicus, fibers_path)
    # Growth always uses one surviving surface component per source patch,
    # including when an older checkpoint explicitly disabled this cleanup.
    # Keep the fitter's erosion amount and per-patch metadata overrides.
    fit_cfg.update({'patch_keep_largest_component': True})
    patch_preprocessing = dict(erode_cells_default=fit_cfg['patch_erode_patches'],
        erosion_override_key='spiral_patch_erode_cells', keep_largest_component=True,
        component_connectivity='4-connected valid quads after erosion')
    if z_range[0] < model_lo or z_range[1] > model_hi:
        parser.error('z_range extends beyond the checkpoint domain')
    context = fiw.fs.FitContext(fit_cfg, scroll=scroll, paths=paths)
    transform, dr = fiw.build_transform(checkpoint, fit_cfg, context, model_lo, model_hi, device=args.device)
    frame = ThetaFrame(transform, dr)
    linking = linking.resolve_distance(fit_cfg['pcl_link_distance_tolerance'])
    print('loading original patches and fibers (no winner selection)', flush=True)
    print(f'patch preprocessing: {json.dumps(patch_preprocessing)}', flush=True)
    patches = context._load_patches_from_dir(raw['patches_dir'])
    fibers, _ = load_fiber_point_collections(fibers_path, 0, min_point_spacing=0.,
                                             base_shape_zyx=scroll.base_shape_zyx)
    print('computing fiber potentials and checking branch cycles', flush=True)
    members, components, excluded_links = prepare_fibers(fibers, frame, z_range)
    eligible = [c for c in components if not c['conflicts']]
    if not eligible:
        parser.error('no internally consistent fiber component in the requested domain')
    seed = min(eligible, key=lambda c: (-c['length'], c['key']))
    print(f'seed: {seed["key"]}, {len(seed["offsets"])} members, {seed["length"]:.1f} voxels', flush=True)
    fields = PatchFields(patches, frame, z_range)
    geometry = GeometryEvidence(patches, members, frame, fields, linking, cfg,
        model_inward_direction(transform, dr, args.device, model_lo, model_hi), fit_cfg)
    # All fiber alternatives are needed for reverse propagation from a patch
    # into a new component. Patch overlaps remain lazy, discovered at the frontier.
    edges = []
    for i, key in enumerate(sorted(members)):
        if members[key]['consistent']:
            edges.extend(geometry.contacts(members[key]))
            geometry.queried_fibers.add(key)
        if i % 50 == 0:
            print(f'contact candidates: {i + 1}/{len(members)} members, {len(edges)} intervals', flush=True)
    def progress(step, node, best):
        if step <= 10 or step % 25 == 0:
            print(f'growth {step}: {node}, offset={best["offset"]}, '
                  f'support={best["support"]:.2f}, agreement={best["agreement"]:.3f}', flush=True)
    from winding_coincidence import CoincidenceGuard
    if cfg.seed_mode == 'all_components':
        from winding_multi_growth import grow_all_components, publish_regions
        regions, edges, runs, merges = grow_all_components(
            eligible, edges, cfg, geometry.expand,
            lambda: CoincidenceGuard(fields, geometry.patch_nodes, cfg.coincidence_tolerance,
                                     cfg.coincidence_metric, cfg.coincidence_grid_size, cfg.coincidence_neighborhood_voxels),
            progress=progress)
        publish_regions(args.output, regions, edges, runs, merges, members, fields,
                        frame, raw, cfg, components, excluded_links, linking,
                        patch_preprocessing, geometry.counts, args.config)
        return
    coincidence = CoincidenceGuard(fields, geometry.patch_nodes, cfg.coincidence_tolerance,
                                     cfg.coincidence_metric, cfg.coincidence_grid_size, cfg.coincidence_neighborhood_voxels)
    result, edges = grow(seed['key'], edges, cfg, geometry.expand, progress, coincidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    field_path = args.output.with_suffix('.potentials.npz')
    evidence_report = []
    for i, e in enumerate(edges):
        residual = (result['offsets'][e.b] - result['offsets'][e.a] - e.delta
                    if e.a in result['offsets'] and e.b in result['offsets'] else None)
        evidence_report.append(dict(id=i, a=e.a, b=e.b, delta=e.delta, support=math.fsum(e.weights),
            status='unassigned_endpoint' if residual is None else 'consistent' if residual == 0 else 'conflicting',
            residual=residual, **e.detail))
    result.update(schema_version=3, acceptance_rule=('all_eligible_connections_and_local_normal_grid_thickness'
                             if cfg.coincidence_metric == 'normal_grid'
                             else 'all_eligible_connections_and_theta_height_coincidence'),
        config=str(args.config), settings=asdict(cfg),
        patch_preprocessing=patch_preprocessing,
        linking=asdict(linking), theta_frame='checkpoint', gauge='seed component root has turn count zero',
        potential_convention='unwrapped theta in turns = theta_fraction + turns; add an unknown integer for absolute winding',
        fiber_components=components, excluded_branch_links=excluded_links,
        patch_internal_conflicts={pid: f['conflicts'] for pid, f in fields.fields.items() if f['conflicts']},
        candidate_counts=dict(geometry.counts),
        evidence=evidence_report,
        fields_file=str(field_path), fields=save_fields(field_path, members, fields, result['offsets']),
        ignored_legacy_options=['patch_id', 'pcl', 'detect_loops', 'min_fix', 'edge_divergence',
                                'linking.selection', 'linking.strip_*'],
        counts=dict(accepted_patches=sum(n.startswith('patch:') for n in result['offsets']),
                    accepted_fiber_components=sum(n.startswith('fiber:') for n in result['offsets']),
                    pending=len(result['pending']), conflicting_intervals=len(result['conflicts'])))
    print('exporting fixed winding layout for Constraint Graph', flush=True)
    workspace_path = export_workspace(args.output, members, fields, result['offsets'],
                                     seed['key'], frame, raw)
    result['workspace_file'] = str(workspace_path)
    args.output.write_text(json.dumps(result, indent=2, default=fiw._to_py) + '\n')
    print(json.dumps(result['counts']), flush=True)
    print(f'wrote {args.output} and {field_path}', flush=True)
    print(f'Open in Constraint Graph: {workspace_path}', flush=True)


if __name__ == '__main__':
    main()
