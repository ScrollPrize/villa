"""Audit persisted winding solutions without modifying inputs or solutions."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack, eye, csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vesuvius/src'))
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import integer_targets
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_benchmark import (
    arrays, digest, large_reference, Paths, read, write)


def sign_constrained(d, s, optimal_cost=None):
    n = len(s['winding'])
    m = len(s['a'])
    rows = np.arange(m)
    b = coo_matrix((np.r_[-np.ones(m), np.ones(m)],
        (np.r_[rows, rows], np.r_[s['a'], s['b']])), shape=(m, n)).tocsr()
    known = d['sign'] != 0
    a, z, sign = d['i'][known], d['j'][known], d['sign'][known]
    minimum = ((~d['parallel'][known]) & (sign*(d['family'][z]-d['family'][a]) < 0)).astype(int)
    transport = s['fiber_b'][known]-s['fiber_a'][known]-s['connector'][known]
    lower = minimum-sign*transport
    rr = np.arange(len(a))
    order = coo_matrix((np.r_[sign, -sign], (np.r_[rr, rr], np.r_[a, z])), shape=(len(a), n)).tocsr()
    mat = vstack([hstack([b, -eye(m)]), hstack([-b, -eye(m)]),
                  hstack([order, csr_matrix((len(a), m))])], format='csr')
    rhs = np.r_[s['target'], -s['target'], -lower]
    if optimal_cost is None:
        objective = np.r_[np.zeros(n), s['weight']]
        bounds = [(None, None)]*n+[(0, None)]*m
    else:
        # Minimize total ordering deficit on the original optimal L1 face.
        slack = vstack([csr_matrix((2*m, len(a))), -eye(len(a))], format='csr')
        mat = vstack([hstack([mat, slack]), csr_matrix(np.r_[np.zeros(n), s['weight'], np.zeros(len(a))][None])], format='csr')
        rhs = np.r_[rhs, optimal_cost]
        objective = np.r_[np.zeros(n+m), np.ones(len(a))]
        bounds = [(None, None)]*n+[(0, None)]*(m+len(a))
    result = linprog(objective, A_ub=mat, b_ub=rhs, bounds=bounds, method='highs-ds')
    if result.status == 2:
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
        edges = [dict(from_source=int(d['source_id'][x]), to_source=int(d['source_id'][y]),
            input_index=graph[x][y]['input_index'], event_id=int(d['event_ids'][graph[x][y]['input_index']]),
            upper_bound=graph[x][y]['weight']) for x, y in zip(cycle[:-1], cycle[1:])]
        return dict(feasible=False, contradictory_cycle=edges,
                    cycle_bound_sum=sum(e['upper_bound'] for e in edges))
    if not result.success:
        raise RuntimeError(result.message)
    k = np.rint(result.x[:n]).astype(np.int64)
    rounding = float(np.max(abs(result.x[:n]-k)))
    if optimal_cost is None and np.max(order@k+lower, initial=0) > 1e-7:
        raise ValueError('Rounded sign-constrained potentials violate order')
    cost = float(s['weight']@abs(b@k-s['target']))
    deficit = np.maximum(order@k+lower, 0)
    if abs((cost if optimal_cost is None else deficit.sum())-result.fun) > 1e-6:
        raise ValueError('Rounded sign-constrained objective not optimal')
    if optimal_cost is not None and abs(cost-optimal_cost) > 1e-6:
        raise ValueError('Lexicographic solve changed original L1 optimum')
    return dict(feasible=True, objective=cost, potential_rounding_max=rounding,
                violations=int(np.sum(deficit > 0)), total_deficit=float(deficit.sum()),
                winding=k.tolist())


def run(source, output):
    manifest = read(source/'manifest.json')
    summary = {}
    original_hashes = {}
    for case in manifest['cases']:
        name = case['name']
        cfg = case['config']
        if digest(cfg['tensor_cache']) != case['tensor_sha256']:
            raise ValueError('Changed input cache')
        saved = source/name/'solution.npz'
        original_hashes[str(saved)] = digest(saved)
        d, s = arrays(cfg['tensor_cache']), arrays(saved)
        old = read(source/name/'metrics.json')
        sign = d['sign']
        known = sign != 0
        delta = s['winding'][d['j']]-s['winding'][d['i']]+s['fiber_b']-s['fiber_a']-s['connector']
        fd = d['family'][d['j']]-d['family'][d['i']]
        margin = ((~d['parallel']) & (sign*fd < 0)).astype(int)
        violation = margin-sign*delta
        bad = known & (violation > 0)
        reverse_margin = ((~d['parallel']) & ((-sign)*(-fd) < 0)).astype(int)
        assert np.array_equal(known & (reverse_margin-(-sign)*(-delta) > 0), bad)
        rows = []
        for index in np.flatnonzero(bad):
            a, b = d['i'][index], d['j'][index]
            rows.append(dict(input_index=int(index), event_id=int(d['event_ids'][index]),
                source_a=int(d['source_id'][a]), source_b=int(d['source_id'][b]),
                family_a='HV'[d['family'][a]], family_b='HV'[d['family'][b]],
                measured_distance=float(d['distance'][index]), sign=int(sign[index]),
                physical_delta=int(delta[index]), required_signed_minimum=int(margin[index]),
                violation_steps=int(violation[index]), target=int(s['measured_target'][index]),
                target_residual=int(delta[index]-s['measured_target'][index]),
                fiber_a_crossings=int(s['fiber_a'][index]), fiber_b_crossings=int(s['fiber_b'][index]),
                connector_crossings=int(s['connector'][index]),
                kind='reversal' if sign[index]*delta[index] < 0 else 'missing_next_winding'))
        if rows:
            output.mkdir(parents=True, exist_ok=True)
            with (output/f'{name}_violations.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        counts = {kind: dict(violations=int(np.sum(bad&mask)), known_events=int(np.sum(known&mask)),
            percent=float(100*np.sum(bad&mask)/max(1, np.sum(known&mask))))
            for kind, mask in [('parallel', d['parallel']), ('perpendicular', ~d['parallel'])]}
        for kind in counts:
            assert counts[kind]['violations'] == old['signs'][kind]['violations']
            assert counts[kind]['known_events'] == old['signs'][kind]['events']
        frequencies = Counter([r['source_a'] for r in rows]+[r['source_b'] for r in rows])
        excluded = ~s['valid']
        plus, _ = integer_targets(d['distance'], np.ones(len(sign)), fd, d['parallel'])
        minus, _ = integer_targets(d['distance'], -np.ones(len(sign)), fd, d['parallel'])
        unsigned_error = np.minimum(abs(delta-plus), abs(delta-minus))
        used = s['valid']
        residual = s['target']-(s['winding'][s['b']]-s['winding'][s['a']])
        objective = float(np.sum(s['weight']*np.where(s['one_sided'], np.maximum(residual, 0), abs(residual))))
        assert objective == old['certificate']['primal']
        result = dict(internal=counts, reversed_order=sum(r['kind']=='reversal' for r in rows),
            missing_next_winding=sum(r['kind']=='missing_next_winding' for r in rows),
            max_violation_steps=max((r['violation_steps'] for r in rows), default=0),
            affected_fibers=len(frequencies), most_affected=freq(frequencies),
            violations_involving_any_seam_transport=sum(bool(r['fiber_a_crossings'] or r['fiber_b_crossings'] or r['connector_crossings']) for r in rows),
            contradictions=int(np.sum(residual != 0)), objective=objective,
            reference=old['reference'], components=old['certificate']['components'],
            excluded_unsigned=dict(count=int(excluded.sum()),
                compatible_with_one_target=int(np.sum(excluded&(unsigned_error == 0))),
                nearest_target_l1=float(unsigned_error[excluded].sum())),
            solution_sha256=original_hashes[str(saved)])
        if name in ('free', 'joint'):
            paths = Paths(cfg, d)
            reference = read('/home/hendrik/villa2/global-fiber-sgd/gt_winding_reference.json')
            def score(winding):
                return large_reference(Path('/home/hendrik/villa2/global-fiber-sgd'), d,
                    dict(s, winding=winding), paths, reference)
            check = score(s['winding'])
            assert check == old['reference']
            result['reference_recomputed_identically'] = True
            constrained = sign_constrained(d, s)
            if constrained['feasible']:
                winding = np.array(constrained.pop('winding'))
                np.savez_compressed(output/f'{name}_sign_constrained.npz', winding=winding, source_id=d['source_id'])
                constrained['extra_l1_cost'] = constrained['objective']-objective
                constrained['reference'] = score(winding)
            result['all_known_signs_enforced'] = constrained
            tie = sign_constrained(d, s, optimal_cost=objective)
            winding = np.array(tie.pop('winding'))
            np.savez_compressed(output/f'{name}_least_deficit_optimum.npz', winding=winding, source_id=d['source_id'])
            tie['reference'] = score(winding)
            result['least_ordering_deficit_at_same_l1_optimum'] = tie
        summary[name] = result
        write(output/'summary.json', summary)
        print(json.dumps(dict(case=name, counts=counts, constrained=result.get('all_known_signs_enforced'))), flush=True)
    for path, before in original_hashes.items():
        assert digest(path) == before
    return summary


def freq(counter):
    return [dict(source_id=int(k), incident_violations=int(v)) for k, v in counter.most_common(10)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'global-fiber-min-flow/data/seven_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'global-fiber-min-flow/data/sign_audit_v1')
    args = parser.parse_args()
    run(args.source, args.output)
