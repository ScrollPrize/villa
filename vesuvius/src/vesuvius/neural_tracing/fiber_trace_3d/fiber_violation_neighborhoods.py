"""Read-only sign-violation proximity versus between-solve winding changes."""
import argparse
from collections import deque
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_trace_3d.fiber_density_sign_report import ordering
from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as density


def distances(adjacency, seeds):
    result = {i: 0 for i in seeds}
    queue = deque(seeds)
    while queue:
        i = queue.popleft()
        for j in adjacency[i]:
            if j not in result:
                result[j] = result[i] + 1
                queue.append(j)
    return result


def run(review):
    torch.set_num_threads(4)
    metadata = json.loads((review / 'comparison.json').read_text())
    with (review / 'all_confident_fibers.csv').open() as f:
        records = list(csv.DictReader(f))
    changed = np.array([r['category'] != 'stable' for r in records])
    baseline = float(changed.mean())
    output = dict(baseline_changed=int(changed.sum()), baseline_total=len(records),
                  baseline_percent=100 * baseline, solves={})
    for name in ('free', 'joint'):
        root = Path(metadata[name])
        config = json.loads((root / 'config.json').read_text())
        with np.load(root / 'solution.npz') as f:
            saved = dict(f)
        with np.load(config['tensor_cache']) as f:
            data = dict(f)
        state = torch.as_tensor(saved['state'])
        with torch.no_grad():
            sigma, bound = density.saved_sigma(state, saved, config)
            hypotheses, logp = density.hypotheses(state, sigma, bound)
            mapped = hypotheses[torch.arange(len(state)), logp.argmax(1)]
            confident = (logp.exp().amax(1) * state[:, 3].sigmoid()).numpy() >= .5
            margin = ordering(mapped, torch.as_tensor(saved['widths']), data,
                              config, saved=saved).numpy()
        # Score the same matched cohort; permit all confident fibers (including
        # GT-only ones) as graph nodes so violations attached to GT are retained.
        index = {int(r[name + '_piece']): k for k, r in enumerate(records)}
        for i in np.flatnonzero(confident):
            if int(i) not in index:
                index[int(i)] = len(index)
        edges = [(index[int(a)], index[int(b)], e)
                 for e, (a, b) in enumerate(zip(data['i'], data['j']))
                 if int(a) in index and int(b) in index]
        adjacency = [set() for _ in index]
        for a, b, _ in edges:
            if a != b:
                adjacency[a].add(b)
                adjacency[b].add(a)
        report = {}
        for kind in ('perpendicular', 'parallel', 'all'):
            bad = [(a, b, e) for a, b, e in edges
                   if data['sign'][e] != 0 and margin[e] < 0
                   and (kind == 'all' or bool(data['parallel'][e]) == (kind == 'parallel'))]
            seeds = {i for a, b, _ in bad for i in (a, b)}
            distance = distances(adjacency, seeds)
            rows = []
            for label, selected in [
                ('direct', [i for i, d in distance.items() if d == 0]),
                ('second_only', [i for i, d in distance.items() if d == 1]),
                ('third_only', [i for i, d in distance.items() if d == 2]),
                ('within_second', [i for i, d in distance.items() if d <= 1]),
                ('within_third', [i for i, d in distance.items() if d <= 2]),
                ('outside_third', [i for i in range(len(records)) if distance.get(i, 999) > 2]),
                ('not_direct', [i for i in range(len(records)) if i not in seeds]),
            ]:
                selected = [i for i in selected if i < len(records)]
                n = len(selected)
                count = int(changed[selected].sum()) if n else 0
                rows.append(dict(cohort=label, changed=count, total=n,
                                 percent=100 * count / n if n else None,
                                 lift=count / n / baseline if n and baseline else None))
            report[kind] = dict(violation_edges=len(bad), rows=rows)
        output['solves'][name] = report
    output['definition'] = ('Same 50%+ joint-confidence-in-both matched fiber cohort; '
        'undirected adjacency through all internal constraints, parallel and perpendicular. '
        'Direct=endpoint of a violated edge; second=one further edge; third=two further edges. '
        'Exact shells exclude closer fibers. All individually confident nodes, including GT, '
        'participate in paths; rates count only matched confident-in-both fibers. '
        'Same realized winding-change labels and global gauge as comparison.json.')
    (review / 'violation_neighborhoods.json').write_text(json.dumps(output, indent=2) + '\n')
    lines = ['# Winding changes versus internal sign violations', '', output['definition'], '',
             f'Overall: {int(changed.sum())}/{len(records)} ({100 * baseline:.2f}%).', '',
             '|Solve|Violation type|Neighborhood|Changed / fibers|Changed %|Lift|',
             '|---|---|---|---:|---:|---:|']
    for name, kinds in output['solves'].items():
        for kind, result in kinds.items():
            for r in result['rows']:
                pct = f"{r['percent']:.2f}" if r['percent'] is not None else '—'
                lift = f"{r['lift']:.2f}" if r['lift'] is not None else '—'
                lines.append(f"|{name}|{kind}|{r['cohort']}|{r['changed']}/{r['total']}|{pct}|{lift}|")
    (review / 'violation_neighborhoods.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(output, indent=2))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('review', type=Path)
    args = parser.parse_args()
    assert distances([{1}, {0, 2}, {1}, set()], {0}) == {0: 0, 1: 1, 2: 2}
    run(args.review)
