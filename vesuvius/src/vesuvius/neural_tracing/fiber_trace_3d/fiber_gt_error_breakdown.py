"""Saved GT-event diagnostics; no optimization or per-subgroup gauge fitting."""
import csv
import io
from pathlib import Path

import numpy as np
import torch

from lasagna.inference_provenance import atomic_write, sha256_file
from . import fiber_implicit_winding as density


def event_features(events, solution, config, order):
    source = solution['source_id']
    if len(np.unique(source)) != len(source):
        raise ValueError('Length report requires whole unsplit source fibers')
    indices = np.asarray([e['test_piece'] for e in events], dtype=int)
    if np.any(indices < 0) or np.any(indices >= len(source)):
        raise ValueError('Invalid test_piece')
    if not np.array_equal(source[indices], [e['test_source'] for e in events]):
        raise ValueError('GT event source identity mismatch')
    length = solution['end'] - solution['start']
    if not np.isfinite(length).all() or np.any(length <= 0):
        raise ValueError('Invalid fiber lengths')
    state = torch.as_tensor(solution['state'])
    with torch.no_grad():
        sigma, bound = density.saved_sigma(state, solution, config)
        _, lp = density.hypotheses(state, sigma, bound)
        enabled_p = state[:, 3].sigmoid().numpy()
        map_p = lp.exp().amax(1).numpy()
    residual = np.asarray([e['residual'] for e in events], dtype=float)
    if not np.isfinite(residual).all() or not np.isfinite(order).all():
        raise ValueError('Nonfinite GT data')
    return dict(source=source[indices], piece=indices,
        gt_source=np.asarray([e['gt_source'] for e in events]),
        parallel=np.asarray([e['parallel'] for e in events], dtype=bool),
        enabled=np.asarray([e['enabled'] for e in events], dtype=bool),
        signed=np.asarray([e['normal_cosine'] != 0 for e in events]),
        error=np.abs(residual), residual=residual,
        correct=np.floor(residual + .5) == 0,
        sign_bad=np.asarray(order) < 0, order=np.asarray(order),
        length=length[indices], family=solution['family'][indices],
        enabled_probability=enabled_p[indices], map_probability=map_p[indices],
        joint_probability=(enabled_p * map_p)[indices])


def summarize(f, mask):
    n = int(mask.sum())
    signed = mask & f['signed']
    bad = signed & f['sign_bad']
    sources = np.unique(f['source'][mask])
    e = f['error'][mask]
    return dict(events=n, fibers=len(sources), signed_events=int(signed.sum()),
        violations=int(bad.sum()),
        sign_percent=float(100 * bad.sum() / signed.sum()) if signed.any() else None,
        violating_fibers=len(np.unique(f['source'][bad])),
        correct_percent=float(100 * f['correct'][mask].mean()) if n else None,
        l1=float(e.mean()) if n else None,
        median=float(np.median(e)) if n else None,
        p95=float(np.quantile(e, .95)) if n else None,
        max=float(e.max()) if n else None,
        fiber_equal_l1=float(np.mean([f['error'][mask & (f['source'] == s)].mean()
                                     for s in sources])) if n else None)


def interval_masks(values, edges):
    return [(f'[{lo:g},{hi:g})' if hi is not None else f'[{lo:g},inf)',
             (values >= lo) & ((values < hi) if hi is not None else True))
            for lo, hi in zip(edges, edges[1:] + [None])]


def tables(f):
    prob = interval_masks(f['joint_probability'], [0, .25, .5, .7, .9])
    length = interval_masks(f['length'], [0, 500, 1000, 2000, 4000])
    enabled = interval_masks(f['enabled_probability'], [0, .5, .9, .99, .999])
    conditional = interval_masks(f['map_probability'], [0, .25, .5, .7, .9])
    groups = [('overall', 'all', np.ones(len(f['source']), bool))]
    groups += [('joint_probability', label, mask) for label, mask in prob]
    groups += [('map_probability', label, mask) for label, mask in conditional]
    groups += [('enabled_probability', label, mask) for label, mask in enabled]
    groups += [('length_base_vx', label, mask) for label, mask in length]
    groups += [('probability_x_length', p + ' / ' + l, pm & lm)
               for p, pm in prob for l, lm in length]
    groups += [('confidence_cutoff', f'>={p:g}', f['joint_probability'] >= p)
               for p in [0, .25, .5, .7, .9]]
    groups += [('gt_source', str(int(s)), f['gt_source'] == s)
               for s in np.unique(f['gt_source'])]
    rows = []
    for kind, type_mask in [('all', np.ones(len(f['source']), bool)),
                            ('parallel', f['parallel']), ('perpendicular', ~f['parallel'])]:
        for grouping, label, mask in groups:
            rows.append(dict(kind=kind, grouping=grouping, bin=label,
                             **summarize(f, mask & type_mask & f['enabled'])))
    return rows


def csv_text(records):
    if not records:
        return ''
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue()


def run(root, events, config, order, saved, output):
    root, output = Path(root), Path(output)
    with np.load(root / 'solve/solution.npz') as z:
        solution = {k: z[k] for k in z.files}
    f = event_features(events, solution, config, order)
    rows = tables(f)
    for kind in ['parallel', 'perpendicular']:
        row = next(r for r in rows if r['kind'] == kind and r['grouping'] == 'overall')
        expected = saved['score']['gt_' + kind]
        if row['events'] != expected['events'] or not np.isclose(row['l1'], expected['l1'], atol=1e-10):
            raise ValueError('Breakdown does not reproduce saved GT score')
        if not np.isclose(row['correct_percent'], expected['correct_percent']):
            raise ValueError('Breakdown correctness differs from saved score')
    fibers = []
    for s in np.unique(f['source']):
        mask = f['source'] == s
        i = np.flatnonzero(mask)[0]
        for kind, select in [('all', mask), ('parallel', mask & f['parallel']),
                              ('perpendicular', mask & ~f['parallel'])]:
            fibers.append(dict(source=int(s), kind=kind, family=int(f['family'][i]),
                length_base_vx=float(f['length'][i]),
                enabled_probability=float(f['enabled_probability'][i]),
                map_probability=float(f['map_probability'][i]),
                joint_probability=float(f['joint_probability'][i]),
                **summarize(f, select & f['enabled'])))
    fibers.sort(key=lambda r: (-r['violations'], -(r['l1'] or 0), r['source'], r['kind']))
    records = [dict(event_id=e['event_id'], source=int(f['source'][i]),
                    gt_source=int(f['gt_source'][i]), parallel=bool(f['parallel'][i]),
                    enabled=bool(f['enabled'][i]), length_base_vx=float(f['length'][i]),
                    enabled_probability=float(f['enabled_probability'][i]),
                    map_probability=float(f['map_probability'][i]),
                    joint_probability=float(f['joint_probability'][i]),
                    residual=float(f['residual'][i]), correct=bool(f['correct'][i]),
                    signed_order=float(f['order'][i]), sign_violation=bool(f['sign_bad'][i]))
               for i, e in enumerate(events)]
    summary = dict(rows=rows, fibers=fibers, global_offset=saved['score']['gt_global_offset'],
        scope='Enabled GT cross-events; test-fiber probability/whole length; GT side fixed',
        definition='Joint=P(enabled)*P(MAP winding); correct residual in [-.5,.5); strict negative MAP sign; one saved global offset',
        enabled_probability_quantiles=np.quantile(f['enabled_probability'], [0,.05,.5,.95,1]).tolist(),
        hashes={str(p): sha256_file(p) for p in [root/'gt_events.json', root/'results.json',
                root/'solve/solution.npz', root/'solve/config.json', Path(__file__)]})
    output.mkdir(parents=True, exist_ok=True)
    atomic_write(output/'breakdown.json', summary, durable=False)
    (output/'events.csv').write_text(csv_text(records), encoding='utf-8')
    (output/'fibers.csv').write_text(csv_text(fibers), encoding='utf-8')
    lines = ['# GT constraint errors by confidence and fiber length', '', summary['scope'], '',
             summary['definition'], '', 'Errors in windings. Event-weighted; fibers may supply multiple events.', '']
    for grouping in ['overall', 'joint_probability', 'length_base_vx', 'confidence_cutoff', 'gt_source']:
        lines += ['## ' + grouping, '', '| Type | Bin | Events | Fibers | Signs | Sign % | Correct % | L1 | Median | P95 | Max |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in rows:
            if r['grouping'] != grouping or r['kind'] == 'all':
                continue
            values = [r[k] for k in ['kind','bin','events','fibers','violations','sign_percent','correct_percent','l1','median','p95','max']]
            lines.append('| ' + ' | '.join(f'{x:.4f}' if isinstance(x,float) else str(x) if x is not None else '—' for x in values) + ' |')
        lines.append('')
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return dict(output=str(output), rows=[r for r in rows if r['grouping']=='overall'])
