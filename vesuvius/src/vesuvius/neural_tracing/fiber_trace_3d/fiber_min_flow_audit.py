"""Optional diagnostics for saved or freshly solved minimum-flow experiments."""
from collections import Counter
from pathlib import Path
import argparse
import csv
import json
import sqlite3

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack, eye, csr_matrix

from .fiber_min_flow import integer_targets
from .fiber_min_flow_evaluation import sign_counts


def endpoints(data, solution):
    return solution.get('event_piece_a', data['i']), solution.get('event_piece_b', data['j'])


def objective_cost(graph, winding):
    residual = graph['target']-(winding[graph['b']]-winding[graph['a']])
    return float(np.sum(graph['weight']*np.where(graph.get('one_sided', np.zeros(len(residual), bool)),
                                               np.maximum(residual, 0), abs(residual))))


def sign_constrained(d, s, optimal_cost=None):
    n = len(s['winding'])
    m = len(s['a'])
    rows = np.arange(m)
    b = coo_matrix((np.r_[-np.ones(m), np.ones(m)],
        (np.r_[rows, rows], np.r_[s['a'], s['b']])), shape=(m, n)).tocsr()
    known = d['sign'] != 0
    ea, eb = endpoints(d, s)
    a, z, sign = ea[known], eb[known], d['sign'][known]
    family_delta = d['family'][d['j']]-d['family'][d['i']]
    minimum = ((~d['parallel'][known]) & (sign*family_delta[known] < 0)).astype(int)
    transport = s['fiber_b'][known]-s['fiber_a'][known]-s['connector'][known]
    lower = minimum-sign*transport
    rr = np.arange(len(a))
    order = coo_matrix((np.r_[sign, -sign], (np.r_[rr, rr], np.r_[a, z])), shape=(len(a), n)).tocsr()
    two_sided = ~s.get('one_sided', np.zeros(m, bool))
    residual_rows = m+int(two_sided.sum())
    mat = vstack([hstack([b[two_sided], -eye(m, format='csr')[two_sided]]), hstack([-b, -eye(m)]),
                  hstack([order, csr_matrix((len(a), m))])], format='csr')
    rhs = np.r_[s['target'][two_sided], -s['target'], -lower]
    if optimal_cost is None:
        objective = np.r_[np.zeros(n), s['weight']]
        bounds = [(None, None)]*n+[(0, None)]*m
    else:
        # Minimize total ordering deficit on the original optimal L1 face.
        slack = vstack([csr_matrix((residual_rows, len(a))), -eye(len(a))], format='csr')
        mat = vstack([hstack([mat, slack]), csr_matrix(np.r_[np.zeros(n), s['weight'], np.zeros(len(a))][None])], format='csr')
        rhs = np.r_[rhs, optimal_cost]
        objective = np.r_[np.zeros(n+m), np.ones(len(a))]
        bounds = [(None, None)]*n+[(0, None)]*(m+len(a))
    result = linprog(objective, A_ub=mat, b_ub=rhs, bounds=bounds, method='highs-ds')
    if result.status == 2 and optimal_cost is None:
        # Convert k_q-k_p>=L into upper-difference arc q->p of cost -L.
        import networkx as nx
        graph = nx.DiGraph()
        graph.add_nodes_from(range(n+1))
        for node in range(n):
            graph.add_edge(n, node, weight=0, input_index=None)
        indices = np.flatnonzero(known)
        for x, y, sg, bound, index in zip(a, z, sign, lower, indices):
            p, q = (int(x), int(y)) if sg > 0 else (int(y), int(x))
            weight = -int(bound)
            if not graph.has_edge(q, p) or weight < graph[q][p]['weight']:
                graph.add_edge(q, p, weight=weight, input_index=int(index))
        cycle = nx.find_negative_cycle(graph, n, weight='weight')
        edges = [dict(from_node=int(x), to_node=int(y), from_source=int(s.get('source_id', d['source_id'])[x]), to_source=int(s.get('source_id', d['source_id'])[y]),
            input_index=graph[x][y]['input_index'], event_id=int(d['event_ids'][graph[x][y]['input_index']]),
            upper_bound=graph[x][y]['weight']) for x, y in zip(cycle[:-1], cycle[1:])]
        return dict(feasible=False, contradictory_cycle=edges,
                    cycle_bound_sum=sum(e['upper_bound'] for e in edges))
    if not result.success:
        raise RuntimeError(result.message)
    k = np.rint(result.x[:n]).astype(np.int64)
    rounding = float(np.max(abs(result.x[:n]-k), initial=0))
    if rounding > 1e-5:
        raise ValueError('Diagnostic LP returned noninteger potentials; refusing to round')
    if optimal_cost is None and np.max(order@k+lower, initial=0) > 1e-7:
        raise ValueError('Rounded sign-constrained potentials violate order')
    cost = objective_cost(s, k)
    deficit = np.maximum(order@k+lower, 0)
    if abs((cost if optimal_cost is None else deficit.sum())-result.fun) > 1e-6:
        raise ValueError('Rounded sign-constrained objective not optimal')
    if optimal_cost is not None and abs(cost-optimal_cost) > 1e-6:
        raise ValueError('Lexicographic solve changed original L1 optimum')
    return dict(feasible=True, objective=cost, potential_rounding_max=rounding,
                violations=int(np.sum(deficit > 0)), total_deficit=float(deficit.sum()),
                winding=k.tolist())


def internal_details(d, s, config):
    """Preserve event-level signs, fractional residuals and unsigned alternatives."""
    ea, eb = endpoints(d, s)
    delta = s['winding'][eb]-s['winding'][ea]+s['fiber_b']-s['fiber_a']-s['connector']
    sign = d['sign']
    fd = d['family'][d['j']]-d['family'][d['i']]
    margin = (~d['parallel']) & (sign*fd < 0)
    bad = (sign != 0) & (margin-sign*delta > 0)
    rows = []
    for i in np.flatnonzero(bad):
        a, b = d['i'][i], d['j'][i]
        rows.append(dict(input_index=int(i), event_id=int(d['event_ids'][i]),
            source_a=int(d['source_id'][a]), source_b=int(d['source_id'][b]),
            piece_a=int(ea[i]), piece_b=int(eb[i]),
            family_a='HV'[d['family'][a]], family_b='HV'[d['family'][b]],
            parallel=bool(d['parallel'][i]), measured_distance=float(d['distance'][i]), sign=int(sign[i]),
            physical_delta=int(delta[i]), required_signed_minimum=int(margin[i]),
            violation_steps=int(margin[i]-sign[i]*delta[i]), target=float(s['measured_target'][i]),
            target_residual=float(delta[i]-s['measured_target'][i]),
            fiber_a_crossings=int(s['fiber_a'][i]), fiber_b_crossings=int(s['fiber_b'][i]),
            connector_crossings=int(s['connector'][i]),
            same_component=bool(s['component'][ea[i]] == s['component'][eb[i]]),
            kind='reversal' if sign[i]*delta[i] < 0 else 'missing_next_winding'))
    freq = Counter([r['source_a'] for r in rows]+[r['source_b'] for r in rows])
    counts = sign_counts(d['parallel'], sign, delta, margin)
    for part in counts.values():
        part.update(known_events=part['events'], percent=100*part['violations']/max(1, part['events']))
    distance = d['distance']*config.get('lasagna_winding_scale', 1.)
    offset = config.get('perpendicular_offset', .5)
    plus, _ = integer_targets(distance, np.ones(len(sign)), fd, d['parallel'], offset)
    minus, _ = integer_targets(distance, -np.ones(len(sign)), fd, d['parallel'], offset)
    unsigned_error = np.minimum(abs(delta-plus), abs(delta-minus))
    excluded = ~s['valid'] & (sign == 0)
    same = s['component'][ea] == s['component'][eb]
    residual = s['target']-(s['winding'][s['b']]-s['winding'][s['a']])
    result = dict(internal=counts, reversed_order=sum(r['kind']=='reversal' for r in rows),
        missing_next_winding=sum(r['kind']=='missing_next_winding' for r in rows),
        max_violation_steps=max((r['violation_steps'] for r in rows), default=0),
        affected_fibers=len(freq), most_affected=[dict(source_id=int(k), incident_violations=int(v))
                                              for k, v in freq.most_common(10)],
        all_fiber_counts={str(k): v for k, v in sorted(freq.items())},
        violations_involving_any_seam_transport=sum(bool(r['fiber_a_crossings'] or r['fiber_b_crossings']
                                                        or r['connector_crossings']) for r in rows),
        contradictions=int(np.count_nonzero(residual)), objective=objective_cost(s, s['winding']),
        components=len(np.unique(s['component'])),
        cross_component_known_events=int(np.sum((sign != 0) & ~same)),
        gauge_note='Internal diagnostics use saved gauges; cross-component comparisons are unidentifiable.',
        excluded_unsigned=dict(count=int(excluded.sum()),
            compatible_with_one_target=int(np.sum(excluded & (unsigned_error == 0))),
            nearest_target_l1=float(unsigned_error[excluded].sum()),
            unidentifiable_events=int(np.sum(excluded & ~same))))
    return result, rows


def gt_details(events):
    bad = [e for e in events if e['sign_violation']]
    cosine = [abs(e['normal_cosine']) for e in bad if e.get('normal_cosine') is not None]
    return dict(events=len(bad), ordinary_fibers=len({e['test_source'] for e in bad}),
        gt_fibers=dict(Counter(e['gt_source'] for e in bad)),
        steps=dict(Counter(e['violation_steps'] for e in bad)),
        reversals=int(sum(np.sign(e['signed_distance'])*e['predicted_gt_minus_test'] < 0 for e in bad)),
        missing_next_winding=sum(e['predicted_gt_minus_test'] == 0 for e in bad),
        ordinary_fiber_counts=dict(Counter(e['test_source'] for e in bad)),
        normal_abs_cosine_min=min(cosine, default=None), normal_abs_cosine_max=max(cosine, default=None))


def compare_targets(baseline, trial, data, config):
    """Cross-evaluate rounded (baseline) and fractional (trial) optima."""
    keys = ['source_id', 'valid', 'fiber_a', 'fiber_b', 'connector']
    if 'parent' in baseline or 'parent' in trial:
        keys += ['parent', 'start', 'end', 'event_piece_a', 'event_piece_b']
    for key in keys:
        if not np.array_equal(baseline.get(key), trial.get(key)):
            raise ValueError('Target comparison input/topology mismatch: '+key)
    a, b = baseline['component'], trial['component']
    pairs = np.unique(np.column_stack([a, b]), axis=0)
    if len(pairs) != len(np.unique(a)) or len(pairs) != len(np.unique(b)):
        raise ValueError('Target comparison component partitions differ')
    delta = trial['winding']-baseline['winding']
    for c in np.unique(a):
        mask = a == c
        values, counts = np.unique(delta[mask], return_counts=True)
        delta[mask] -= values[np.argmax(counts)]
    costs = dict(rounded_baseline=objective_cost(baseline, baseline['winding']),
                 rounded_trial=objective_cost(baseline, trial['winding']),
                 fractional_baseline=objective_cost(trial, baseline['winding']),
                 fractional_trial=objective_cost(trial, trial['winding']))
    if costs['rounded_baseline'] > costs['rounded_trial']+1e-7 or costs['fractional_trial'] > costs['fractional_baseline']+1e-7:
        raise ValueError('Cross-objective optimality check failed')
    direct_checked = not np.any(trial['one_sided'])
    if direct_checked:
        ea, eb = endpoints(data, trial)
        valid = trial['valid']
        weights = np.where(data['parallel'], 1., config.get('perpendicular_relative_weight', 1.))
        for s in (baseline, trial):
            direct = np.sum(weights[valid]*abs((s['winding'][eb]-s['winding'][ea]-trial['corrected_target'])[valid]))
            np.testing.assert_allclose(direct, objective_cost(trial, s['winding']), rtol=1e-12, atol=1e-10)
    changed = np.flatnonzero(delta)
    return dict(objectives=costs, fractional_expansion_checked=direct_checked,
        changed_nodes_after_component_mode_gauge=len(changed),
        changed_node_ids=trial['source_id'][changed].tolist(), changed_node_indices=changed.tolist(),
        changed_pieces=[dict(source_id=int(trial['source_id'][i]), start=float(trial['start'][i]),
                            end=float(trial['end'][i])) for i in changed] if 'parent' in trial else [],
        max_node_change=int(np.max(abs(delta), initial=0)),
        rounded_arcs=len(baseline['a']), fractional_arcs=len(trial['a']))


def audit_measurements(config, data, sources):
    """Recompute the original cached measurements; missing sources are explicit."""
    from .fiber_min_flow_experiment import arrays, digest
    from .fiber_normal_convention import inward_solution
    from .fiber_uv_layout import signed_order
    actual = digest(config['tensor_cache'])
    if config.get('tensor_sha256') and actual != config['tensor_sha256']:
        raise ValueError('Tensor cache differs from expected hash')
    missing = [str(sources[k]) for k in ('database', 'stage1') if not Path(sources[k]).is_file()]
    if missing:
        return dict(source_afc_recomputation='unavailable', missing_inputs=missing, rows_verified=0,
                    tensor_sha256=actual, expected_tensor_hash_checked=bool(config.get('tensor_sha256')))
    normal = inward_solution(arrays(sources['stage1']))
    with sqlite3.connect(Path(sources['database']).resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.execute('CREATE TEMP TABLE selected(id INTEGER PRIMARY KEY)')
        db.executemany('INSERT INTO selected VALUES (?)', ((int(i),) for i in np.unique(data['event_ids'])))
        rows = db.execute('SELECT id,fiber_a,fiber_b,ax,ay,az,bx,by,bz,parallel_score,perpendicular_score,'
            'winding_distance,parallel_winding_distance,normal_a,normal_b FROM constraints '
            'WHERE id IN (SELECT id FROM selected)').fetchall()
    by_id = {r[0]: r for r in rows}
    if any(int(i) not in by_id for i in data['event_ids']):
        raise ValueError('Cached event ID missing from source AFC')
    rows = [by_id[int(i)] for i in data['event_ids']]
    if not rows:
        return dict(source_afc_recomputation='verified', rows_verified=0)
    ids = np.array([r[1:3] for r in rows], np.int64)
    expected = np.column_stack([data['source_id'][data['i']], data['source_id'][data['j']]])
    if not np.array_equal(ids, expected):
        raise ValueError('Source AFC endpoint identity differs from cache')
    lookup = {int(s): i for i, s in enumerate(normal['ids'])}
    if any(int(s) not in lookup for s in ids.flat):
        raise ValueError('Source AFC fiber missing from normal solution')
    ends = np.array([[lookup[int(s)] for s in pair] for pair in ids])
    nums = np.array([r[3:13] for r in rows], float)
    par = nums[:, 6] > nums[:, 7]
    distance = np.where(par, nums[:, 9], nums[:, 8])
    def normals(column, side):
        records = [json.loads(r[column]) for r in rows]
        return np.array([r['xyz'] if r['valid'] else [np.nan]*3 for r in records])*normal['normal_sign'][ends[:, side], None]
    sign = signed_order(nums[:, 3:6]-nums[:, :3], normals(13, 0), normals(14, 1), sources.get('sign_cosine_min', .1))
    if not np.array_equal(sign, data['sign']) or not np.array_equal(par, data['parallel']) or not np.allclose(distance, data['distance'], rtol=1e-7, atol=1e-8):
        raise ValueError('Source AFC measurements do not match cached constraints')
    return dict(source_afc_recomputation='verified', rows_verified=len(rows), signs_identical=True,
        family_classification_identical=True, distance_max_roundoff=float(np.max(abs(distance-data['distance']))),
        sign_cosine_min=sources.get('sign_cosine_min', .1), tensor_sha256=actual,
        input_sha256={k: digest(sources[k]) for k in ('database', 'stage1')})

def export_rows(path, rows):
    with Path(path).open('w', newline='') as stream:
        columns = list(dict.fromkeys(k for row in rows for k in row)) or ['event_id']
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def audit_case(case, data, solution, folder, options, source_path=None):
    """Write opt-in diagnostics without changing the saved solution."""
    from .fiber_min_flow_experiment import arrays, read, write, digest
    from .fiber_min_flow_benchmark import Paths
    from .fiber_min_flow_evaluation import small_reference, large_reference
    from .fiber_min_flow_split import Pieces
    folder = Path(folder)
    source_path = Path(source_path) if source_path is not None else folder.parent/'solution.npz'
    folder.mkdir(parents=True, exist_ok=True)
    report = {}
    evaluation = case.get('evaluation')
    if evaluation:
        reference = read(evaluation['reference'])
        if evaluation['mode'] == 'cross_events':
            paths = Paths(case['dataset'], data)
            gc = evaluation['dataset']
            gd = arrays(gc['tensor_cache'])
            gp = Paths(gc, gd)
            events = read(evaluation['events'])
            pieces = None
            if 'parent' in solution:
                split = case['split']
                pieces = Pieces(data, split.get('minimum_length_vx', 1000),
                    split.get('minimum_perpendicular', 4), split.get('protected_sources', []))
            def score(s, details):
                return large_reference(data, s, paths, reference, events, gd, gp, details, pieces)
        else:
            def score(s, details):
                return small_reference(data, s, reference, s)
    else:
        def score(s, details):
            return None
    details = []
    report['reference'] = score(solution, details)
    report['solution_sha256'] = digest(source_path) if source_path.is_file() else None
    metrics_path = source_path.with_name('metrics.json')
    if metrics_path.is_file():
        saved = read(metrics_path)
        if 'certificate' in saved:
            np.testing.assert_allclose(objective_cost(solution, solution['winding']),
                                       saved['certificate']['primal'], rtol=1e-12, atol=1e-8)
            report['certificate'] = saved['certificate']
        if evaluation and 'reference' in saved:
            # Historical reports have different descriptive metadata; compare scores.
            for key in ('all', 'parallel', 'perpendicular', 'component_gauges', 'signs'):
                if key in saved['reference'] and saved['reference'][key] != report['reference'].get(key):
                    raise ValueError('Recomputed reference differs from saved metrics: '+key)
            report['reference_recomputed_identically'] = True
    if options.get('details', False):
        report['internal'], rows = internal_details(data, solution, case['dataset'])
        export_rows(folder/'internal_violations.csv', rows)
        write(folder/'internal_violations.json', rows)
        if 'parent' in solution:
            links = np.flatnonzero(np.diff(solution['parent']) == 0)
            jumps = solution['winding'][links+1]-solution['winding'][links]
            boundaries = [dict(source_id=int(solution['source_id'][i]), left_piece=int(i),
                right_piece=int(i+1), arc=float(solution['end'][i]), raw_jump=int(jump),
                identifiable=bool(solution['component'][i] == solution['component'][i+1]))
                for i, jump in zip(links, jumps)]
            export_rows(folder/'boundaries.csv', boundaries)
            with sqlite3.connect(Path(case['dataset']['afv']).as_uri()+'?mode=ro', uri=True) as db:
                row = db.execute("SELECT value FROM metadata WHERE key='gt_source_id_map'").fetchone()
            gt_sources = list(map(int, json.loads(row[0]).values())) if row else []
            report['split_details'] = dict(boundaries=len(links),
                raw_nonzero_boundary_jumps=int(np.count_nonzero(jumps)),
                raw_boundary_max_jump=int(np.max(abs(jumps), initial=0)),
                note='Raw jumps include unidentifiable component offsets; use identifiable flags.',
                gt_original_fibers=len(gt_sources),
                gt_pieces=int(np.isin(solution['source_id'], gt_sources).sum()))
        if evaluation and evaluation['mode'] == 'cross_events':
            report['gt_violation_detail'] = gt_details(details)
            export_rows(folder/'gt_events.csv', details)
            bad = [e for e in details if e['sign_violation']]
            export_rows(folder/'gt_violations.csv', bad)
            write(folder/'gt_violations.json', bad)
    if options.get('measurements'):
        report['measurements'] = audit_measurements(case['dataset'], data, options['measurements'])
    if options.get('sign_analysis', False):
        cost = objective_cost(solution, solution['winding'])
        for name, optimum in [('all_known_signs_enforced', None), ('least_ordering_deficit_at_same_l1_optimum', cost)]:
            alternative = sign_constrained(data, solution, optimum)
            if alternative['feasible']:
                winding = np.asarray(alternative.pop('winding'), np.int64)
                variant = dict(solution, winding=winding)
                alternative['extra_l1_cost'] = alternative['objective']-cost
                alternative_details = []
                alternative['reference'] = score(variant, alternative_details)
                # Only labels and topology: original flows certify the original solve, not these variants.
                fields = {k: solution[k] for k in ('source_id', 'parent', 'start', 'end', 'component') if k in solution}
                np.savez_compressed(folder/(name+'.npz'), winding=winding, **fields)
                if options.get('details', False):
                    summary, rows = internal_details(data, variant, case['dataset'])
                    alternative['internal'] = summary
                    export_rows(folder/(name+'_internal_violations.csv'), rows)
                    export_rows(folder/(name+'_gt_events.csv'), alternative_details)
                    alternative['gt_violation_detail'] = gt_details(alternative_details)
            report[name] = alternative
    if options.get('rounded_baseline'):
        path = Path(options['rounded_baseline'])
        baseline = arrays(path)
        # Verify the baseline really is the rounded graph for this same input.
        validate_solution(dict(case, target_mode='rounded'), data, baseline)
        comparison = compare_targets(baseline, solution, data, case['dataset'])
        comparison['baseline_solution_sha256'] = digest(path)
        comparison['rounded_reference'] = score(baseline, [])
        comparison['fractional_reference'] = report['reference']
        comparison['rounded_internal_signs'] = internal_details(data, baseline, case['dataset'])[0]['internal']
        comparison['fractional_internal_signs'] = internal_details(data, solution, case['dataset'])[0]['internal']
        for prefix, metrics_path in [('rounded', path.with_name('metrics.json')),
                                      ('fractional', source_path.with_name('metrics.json'))]:
            if metrics_path.is_file():
                metrics = read(metrics_path)
                comparison[prefix+'_certificate'] = metrics.get('certificate')
                comparison[prefix+'_flow_seconds'] = metrics.get('certificate', {}).get('seconds')
        report['target_comparison'] = comparison
    write(folder/'summary.json', report)
    return report


def validate_solution(case, data, solution):
    """Ensure a saved result belongs to the explicitly configured data and graph."""
    from .fiber_min_flow_benchmark import Paths, graph
    from .fiber_min_flow_split import Pieces
    from scipy.sparse.csgraph import connected_components
    n = len(data['source_id'])
    sign_only = case.get('sign_only', False)
    g, transport = graph(case['dataset'], data, Paths(case['dataset'], data), sign_only,
                         case.get('target_mode', 'fractional'))
    expected = dict(transport, source_id=data['source_id'])
    if case.get('split') is not None:
        split = case['split']
        pieces = Pieces(data, split.get('minimum_length_vx', 1000),
                        split.get('minimum_perpendicular', 4), split.get('protected_sources', []))
        g = pieces.remap(g, transport)
        if sign_only:
            reverse = data['sign'][transport['flow_event_index']] < 0
            g['a'][reverse], g['b'][reverse] = g['b'][reverse].copy(), g['a'][reverse].copy()
        expected.update(source_id=pieces.source_id, parent=pieces.parent, start=pieces.start,
                        end=pieces.end, event_piece_a=pieces.endpoints('a'), event_piece_b=pieces.endpoints('b'))
        n = len(pieces.parent)
    elif 'parent' in solution:
        raise ValueError('Saved split solution requires split configuration')
    for key, value in dict(expected, **g).items():
        # Initial rounded snapshots predate the expanded-arc index metadata.
        # Their endpoint/target/weight arrays and all transport are still checked.
        if key == 'flow_event_index' and key not in solution and case.get('target_mode') == 'rounded':
            continue
        if key not in solution or not np.array_equal(value, solution[key], equal_nan=True):
            raise ValueError('Saved solution/config mismatch: '+key)
    if len(solution['winding']) != n or not np.all(np.isfinite(solution['winding'])) or not np.array_equal(solution['winding'], np.rint(solution['winding'])):
        raise ValueError('Saved winding labels must be finite integers with matching node count')
    adjacency = coo_matrix((np.ones(len(g['a'])), (g['a'], g['b'])), shape=(n, n))
    _, component = connected_components(adjacency, directed=False)
    pairs = np.unique(np.column_stack([component, solution['component']]), axis=0)
    if len(pairs) != len(np.unique(component)) or len(pairs) != len(np.unique(solution['component'])):
        raise ValueError('Saved component partition differs from graph')


def main():
    from .fiber_min_flow_experiment import arrays, load_config, digest, write, read
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='External experiment config with per-case audit options')
    parser.add_argument('--source', type=Path, required=True, help='Existing results directory (never modified)')
    parser.add_argument('--output', type=Path, required=True, help='New audit output directory')
    args = parser.parse_args()
    settings, dependencies = load_config(args.config)
    output = args.output.resolve()
    inputs = set(dependencies)
    source_manifest = args.source/'manifest.json'
    if source_manifest.is_file():
        inputs.add(str(source_manifest.resolve()))
    for case in settings['cases']:
        inputs.add(str((args.source/case['name']/'solution.npz').resolve()))
        metrics = args.source/case['name']/'metrics.json'
        if metrics.is_file():
            inputs.add(str(metrics.resolve()))
    if any(output == Path(p) or output in Path(p).parents for p in inputs):
        raise ValueError('Audit output must not contain inputs')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Audit output must be empty')
    before = {p: digest(p) for p in sorted(inputs)}
    if source_manifest.is_file():
        original_hashes = read(source_manifest).get('input_hashes', {})
        for p, expected in original_hashes.items():
            if p in before and before[p] != expected:
                raise ValueError('Input differs from saved manifest: '+p)
    output.mkdir(parents=True, exist_ok=True)
    write(output/'manifest.json', dict(settings=settings, source=str(args.source.resolve()), input_hashes=before))
    for case in settings['cases']:
        config = case['dataset']
        datasets = [config]
        if case.get('evaluation', {}).get('mode') == 'cross_events':
            datasets.append(case['evaluation']['dataset'])
        for dataset in datasets:
            if dataset.get('tensor_sha256') and before[dataset['tensor_cache']] != dataset['tensor_sha256']:
                raise ValueError('Tensor hash mismatch')
        data = arrays(config['tensor_cache'])
        path = args.source/case['name']/'solution.npz'
        solution = arrays(path)
        validate_solution(case, data, solution)
        report = audit_case(case, data, solution, output/case['name'], case.get('audit', {}), path)
        report['solution_sha256'] = digest(path)
        write(output/case['name']/'summary.json', report)
        print('Audited '+case['name'], flush=True)
    if any(digest(p) != h for p, h in before.items()):
        raise ValueError('Audit inputs changed during execution')


if __name__ == '__main__':
    main()
