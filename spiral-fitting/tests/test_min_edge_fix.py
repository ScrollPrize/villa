"""solve_min_edge_fix repairs the winding graph with every measured equation in
the model, and with a potential box that holds the optimum."""

import numpy as np

import find_inconsistent_windings as fiw


def _edge_graph(edges):
    adjacency = {}
    for k, (P, R, D) in enumerate(edges):
        common = {'pcl_id': k, 'pcl_name': f'pcl{k}', 'source_file': 'relative_windings.json',
                  'kind': 'relative', 'from_zyx': np.zeros(3, np.float32),
                  'to_zyx': np.zeros(3, np.float32), 'raw_winding_delta': D,
                  'pcl_unwrap_adjustment': 0, 'pcl_branch_delta': 0}
        adjacency.setdefault(P, []).append({
            **common, 'neighbor': R, 'from_ij': np.zeros(2), 'to_ij': np.zeros(2),
            'from_point_id': 1, 'to_point_id': 2, 'winding_delta': D})
        adjacency.setdefault(R, []).append({
            **common, 'neighbor': P, 'from_ij': np.zeros(2), 'to_ij': np.zeros(2),
            'from_point_id': 2, 'to_point_id': 1, 'winding_delta': -D})
    return adjacency


def _reached(accs):
    return {p: {'acc': acc, 'entry_ij': np.zeros(2), 'hops': 0} for p, acc in accs.items()}


def _closes(edges, fix):
    deltas = {k: D for k, (_, _, D) in enumerate(edges)}
    for e in fix['edges']:
        sign = 1 if e['from_patch'] == edges[e['rel_pcl_id']][0] else -1
        deltas[e['rel_pcl_id']] = sign * e['suggested_winding_delta']
    adjacency = {}
    for k, (P, R, _) in enumerate(edges):
        adjacency.setdefault(P, []).append((R, deltas[k]))
        adjacency.setdefault(R, []).append((P, -deltas[k]))
    potential = {}
    for start in adjacency:
        if start in potential:
            continue
        potential[start] = 0
        stack = [start]
        while stack:
            a = stack.pop()
            for b, d in adjacency[a]:
                if b not in potential:
                    potential[b] = potential[a] + d
                    stack.append(b)
                elif potential[b] != potential[a] + d:
                    return False
    return True


def test_min_edge_fix_keeps_every_equation_and_only_restricts_edits():
    edges = [('S', 'A', 0), ('A', 'B', 0), ('B', 'E', 0), ('B', 'C', 0),
             ('S', 'B', 1), ('A', 'E', 1), ('A', 'C', 0)]
    on_inconsistent_cycles = {(k, frozenset((1, 2))) for k in (0, 1, 2, 4, 5)}
    fix = fiw.solve_min_edge_fix(
        _reached({'S': 0, 'A': 0, 'B': 0, 'E': 0, 'C': 0}), _edge_graph(edges),
        lambda *_: 0, allowed_edge_keys=on_inconsistent_cycles)
    assert fix['num_edges_considered'] == 7
    assert fix['num_edges_editable'] == 5
    changed = {e['rel_pcl_id'] for e in fix['edges']}
    assert len(changed & {0, 4}) == 1 and len(changed & {2, 5}) == 1 and len(changed) == 2
    assert _closes(edges, fix)


def test_min_edge_fix_box_holds_a_wrong_tree_edge_larger_than_its_margin():
    edges = [('S', 'A', 100), ('S', 'A', 0), ('S', 'A', 0)]
    fix = fiw.solve_min_edge_fix(_reached({'S': 0, 'A': -100}), _edge_graph(edges), lambda *_: 0)
    assert fix['num_edges_changed'] == 1
    assert fix['edges'][0]['rel_pcl_id'] == 0
    assert fix['edges'][0]['suggested_winding_delta'] == 0
    assert _closes(edges, fix)
