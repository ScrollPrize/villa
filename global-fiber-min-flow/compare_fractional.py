"""Compare saved rounded and fractional runs without changing either solution."""
from pathlib import Path
import numpy as np
from audit_gt_signs import arrays, read, write, digest

ROOT = Path(__file__).resolve().parent


def cost(graph, winding):
    residual = graph['target']-(winding[graph['b']]-winding[graph['a']])
    return float(np.sum(graph['weight']*np.where(graph['one_sided'], np.maximum(residual, 0), abs(residual))))


def run():
    baseline, trial = ROOT/'data/seven_v1', ROOT/'data/seven_fractional_v1'
    old_manifest, new_manifest = read(baseline/'manifest.json'), read(trial/'manifest.json')
    assert old_manifest['input_hashes'] == new_manifest['input_hashes']
    summary = {}
    for case in new_manifest['cases']:
        name = case['name']
        old, new = [arrays(p/name/'solution.npz') for p in (baseline, trial)]
        for field in ('source_id', 'family', 'component', 'valid', 'fiber_a', 'fiber_b', 'connector'):
            np.testing.assert_array_equal(old[field], new[field])
        delta = new['winding']-old['winding']
        for c in np.unique(old['component']):
            mask = old['component'] == c
            values, counts = np.unique(delta[mask], return_counts=True)
            delta[mask] -= values[np.argmax(counts)]
        om, nm = [read(p/name/'metrics.json') for p in (baseline, trial)]
        objectives = dict(rounded_baseline=cost(old, old['winding']), rounded_trial=cost(old, new['winding']),
                          fractional_baseline=cost(new, old['winding']), fractional_trial=cost(new, new['winding']))
        assert objectives['rounded_baseline'] <= objectives['rounded_trial']+1e-7
        assert objectives['fractional_trial'] <= objectives['fractional_baseline']+1e-7
        d = arrays(case['config']['tensor_cache'])
        valid = new['valid']
        weights = np.where(d['parallel'], 1., case['config'].get('perpendicular_relative_weight', 1.))
        if not case['sign_only']:
            for s in (old, new):
                direct = np.sum(weights[valid]*abs((s['winding'][d['j']]-s['winding'][d['i']]-new['corrected_target'])[valid]))
                np.testing.assert_allclose(direct, cost(new, s['winding']), rtol=1e-12)
        def counts(metrics):
            return {kind: dict(exact=round(part['correct_percent']*part['events']/100), events=part['events'], l1=part['l1'])
                    for kind, part in metrics['reference'].items() if kind in ('all', 'parallel', 'perpendicular')}
        summary[name] = dict(objectives=objectives, changed_nodes_after_component_mode_gauge=int(np.sum(delta != 0)),
            changed_node_ids=old['source_id'][delta != 0].tolist(),
            max_node_change=int(np.max(abs(delta), initial=0)),
            rounded_reference=counts(om), fractional_reference=counts(nm),
            rounded_gt_signs=om['reference'].get('signs'), fractional_gt_signs=nm['reference'].get('signs'),
            rounded_internal_signs=om['signs'], fractional_internal_signs=nm['signs'],
            rounded_flow_seconds=om['certificate']['seconds'], fractional_flow_seconds=nm['certificate']['seconds'],
            rounded_arcs=len(old['a']), fractional_arcs=len(new['a']),
            fractional_certificate=nm['certificate'],
            baseline_solution_sha256=digest(baseline/name/'solution.npz'), trial_solution_sha256=digest(trial/name/'solution.npz'))
    write(trial/'comparison.json', summary)
    for name, s in summary.items():
        print(name, s['objectives'], 'changed', s['changed_nodes_after_component_mode_gauge'],
              'GT exact', s['rounded_reference']['all'], '->', s['fractional_reference']['all'],
              'GT signs', s['rounded_gt_signs'], '->', s['fractional_gt_signs'])
    return summary


if __name__ == '__main__':
    run()
