"""Analyze saved BP removal labels against audited native forward-step costs.

No inference or BP is run. Input is a JSON config listing datasets with
label, prefix (saved BP), and steps (verified diagnostic CSV). Output is JSON.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

import numpy as np


def summarize(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    return dict(n=len(a), mean=float(a.mean()), p05=float(np.quantile(a, .05)),
                median=float(np.median(a)), p95=float(np.quantile(a, .95)),
                maximum=float(a.max()))


def piece_costs(segments, begin, end):
    """Segments columns: begin,end,retained cost,full original step cost,scale."""
    if not end > begin:
        raise ValueError('empty piece')
    lo = np.searchsorted(segments[:, 1], begin, side='right')
    hi = np.searchsorted(segments[:, 0], end, side='left')
    s = segments[lo:hi]
    overlap = np.minimum(s[:, 1], end) - np.maximum(s[:, 0], begin)
    if not np.isclose(overlap.sum(), end - begin, rtol=1e-10, atol=1e-7):
        raise ValueError('piece is not fully covered by diagnostic segments')
    density = s[:, 2] / (s[:, 1] - s[:, 0])
    total = float(np.dot(density, overlap))
    return dict(total_cost=total, mean_cost_per_base_voxel=total/(end-begin),
                mean_cost_per_prediction_voxel=total/(end-begin)*float(s[0, 4]),
                max_original_step_cost=float(s[:, 3].max()),
                max_cost_per_base_voxel=float(density.max()),
                max_cost_per_prediction_voxel=float((density*s[:, 4]).max()))


def removal_labels(prefix):
    labels = {}
    paths = sorted(prefix.parent.glob(prefix.name + '_oracle_fresh_*.obj'))
    if not paths:
        raise ValueError('missing saved pruning artifacts')
    for path in paths:
        reason = re.sub(r'^.*_oracle_fresh_(?:w_\d+_)?', '', path.stem)
        with path.open() as stream:
            for line in stream:
                m = re.match(r'o (conditioned_broken|final)_piece_(\d+)\s*$', line)
                if m:
                    i = int(m[2])
                    if i in labels:
                        raise ValueError(f'duplicate piece {i}')
                    labels[i] = ('removed' if m[1] == 'conditioned_broken' else 'retained', reason)
    return labels


def load_steps(path):
    traces = defaultdict(list)
    totals = {}
    with Path(path).open() as stream:
        for x in csv.DictReader(stream):
            t = int(x['trace'])
            if int(x['segment']) != len(traces[t]):
                raise ValueError('nonsequential segment indices')
            traces[t].append([float(x[k]) for k in
                             ['begin_arc_base','end_arc_base','cost','original_step_cost','prediction_scale']])
            totals[t] = float(x['total_trace_cost'])
    traces = {t:np.asarray(s) for t,s in traces.items()}
    for t,s in traces.items():
        if not np.isclose(s[:,2].sum(), totals[t], rtol=1e-9, atol=1e-8):
            raise ValueError(f'trace cost sum mismatch {t}')
        if s[0,0] != 0 or not np.array_equal(s[1:,0], s[:-1,1]):
            raise ValueError('noncontiguous segments')
    return traces


def analyze(entry):
    prefix = Path(entry['prefix'])
    with Path(str(prefix)+'_consistency.csv').open() as stream:
        pieces = list(csv.DictReader(stream))
    classes = {}
    lengths = [float(x['end_arc_base']) - float(x['begin_arc_base']) for x in pieces]
    for field in ['winding_prepass_class', 'winding_final_class']:
        counts = Counter(x[field] for x in pieces)
        length_by_class = defaultdict(float)
        for x, length in zip(pieces, lengths):
            length_by_class[x[field]] += length
        classes[field] = {k:dict(count=v, percent=100*v/len(pieces),
                            length_percent=100*length_by_class[k]/sum(lengths))
                          for k,v in counts.items()}
    labels = removal_labels(prefix)
    if set(labels) != {int(x['piece']) for x in pieces}:
        raise ValueError('removal labels do not partition initial pieces')
    traces = load_steps(entry['steps'])
    rows = []
    for p,length in zip(pieces,lengths):
        state, reason = labels[int(p['piece'])]
        costs = piece_costs(traces[int(p['original_trace'])],
                            float(p['begin_arc_base']),float(p['end_arc_base']))
        rows.append(dict(piece=int(p['piece']), trace=int(p['original_trace']),
                         group=state, reason=reason,length_base=length, **costs))
    groups = {}
    for group in ['retained','removed']:
        subset=[r for r in rows if r['group']==group]
        groups[group]=dict(count=len(subset), percent=100*len(subset)/len(rows),
            length_percent=100*sum(r['length_base'] for r in subset)/sum(lengths),
            reasons=dict(Counter(r['reason'] for r in subset)),
            metrics={k:summarize([r[k] for r in subset]) for k in
                     ['length_base','total_cost','mean_cost_per_base_voxel',
                      'mean_cost_per_prediction_voxel','max_original_step_cost',
                      'max_cost_per_base_voxel','max_cost_per_prediction_voxel']})
    return dict(label=entry['label'], prefix=str(prefix), steps=entry['steps'],
                pieces=len(pieces),classes=classes,groups=groups,piece_costs=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=[analyze(x) for x in json.loads(args.config.read_text())]
    detail=args.output.with_suffix('.pieces.csv')
    with detail.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['dataset',*result[0]['piece_costs'][0]])
        writer.writeheader()
        for dataset in result:
            for row in dataset.pop('piece_costs'):
                writer.writerow(dict(dataset=dataset['label'],**row))
            dataset['piece_costs_csv']=str(detail)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for x in result:
        print(x['label'],json.dumps({'classes':x['classes'],'groups':x['groups']}))


if __name__ == '__main__':
    main()
