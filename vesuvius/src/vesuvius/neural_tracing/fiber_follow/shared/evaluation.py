"""The single evaluation protocol: calibrate an operating point, run fixed seeds, compare pairs.

Every rollout uses the shared threaded tracer and the resolved operating policy.
Calibration reads only calibration seeds and requires 95% scored precision before
choosing an operating point for coverage; if none qualifies, it says so and writes no
selection. Runs report the strict first-departure metrics together with decision and
geometric outcomes, by source and split. Comparisons pair identical seeds and resample
fibers. The final group never enters operating-point selection.

  python -m vesuvius.neural_tracing.fiber_follow.regression.evaluate calibrate --checkpoint CK --out DIR
  python -m vesuvius.neural_tracing.fiber_follow.regression.evaluate run --checkpoint CK --out FILE.json \\
      [--confidence .5 | --policy DIR/selection.json] [--splits final] [--max-len 2000]
  python -m vesuvius.neural_tracing.fiber_follow.regression.evaluate compare BASE.json NEW.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from vesuvius.neural_tracing.fiber_follow.shared.evaluate import (
    EvaluationAudit, OUTCOME_COUNTS, OUTCOME_LENGTHS, evaluate, monitor_coverage, summarize_outcomes,
)
from vesuvius.neural_tracing.fiber_follow.shared.experiment import jsonable
from vesuvius.neural_tracing.fiber_follow.shared.policy import checkpoint_policy
from vesuvius.neural_tracing.fiber_follow.shared.trace import TraceParams

SPLITS = ('monitor', 'calibration', 'final')
THRESHOLDS = (.5, .6, .7, .8, .85, .9, .95)
PAIRED_METRICS = ('correct', 'offtrack', 'diverged', 'premature_stop', 'distance_events', 'confirmed_switch',
                  'rejected_unsafe', 'accepted_unsafe', 'geometric_agreement')


def trace_rows(tracer, source, split, *, max_len, tolerance, batch):
    """Rows for one source/split with outcomes; monitor normalization caps availability."""
    audit = EvaluationAudit(tracer, tolerance, source.get('detector'))
    paths = {}
    rows, _ = evaluate(tracer, source['fibers'], source['manifest'][split], batch=batch, history_audit=audit,
                       on_trace=lambda seed, path, reason: paths.__setitem__((seed['fiber'], seed['t'], seed['sign']), path))
    result = []
    for row in rows:
        row = monitor_coverage(row, max_len)
        row.update(source=source['name'], split=split,
                   path=np.asarray(paths[(row['fiber'], row['t0'], row['sign'])]).round(3).tolist())
        result.append(row)
    return result


def run_policy(checkpoint, sources, splits, policy, *, checkpoint_loader, tracer_class, device, max_len, batch,
               forward_chunk=0):
    model, crop, n_history, _, ck = checkpoint_loader(checkpoint, device)
    tolerance = float(ck.get('tolerance', 1.5))
    rows = []
    for source in sources:
        tracer = tracer_class(model, source['volume'], crop, n_history,
                              TraceParams.from_policy(policy, max_len=max_len, forward_chunk=forward_chunk), device=device)
        try:
            for split in splits:
                rows.extend(trace_rows(tracer, source, split, max_len=max_len, tolerance=tolerance, batch=batch))
        finally:
            tracer.close()
    return rows, ck


def grouped_summary(rows):
    groups = {'all': rows}
    for row in rows:
        groups.setdefault(row['source'], []).append(row)
        groups.setdefault(f"{row['source']}/{row['split']}", []).append(row)
    return {name: summarize_outcomes(members) for name, members in groups.items() if members}


def manifest_digest(sources, splits):
    payload = {s['name']: {split: s['manifest'][split] for split in splits} for s in sources}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=jsonable).encode()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1, default=jsonable))


def calibrate(args, sources, *, checkpoint_loader, tracer_class):
    """Threshold sweep on calibration seeds only; select coverage at >= 95% scored precision."""
    out = Path(args.out)
    if (out/'selection.json').exists():
        raise FileExistsError('Selection already locked; use a new output directory')
    model, _, _, _, ck = checkpoint_loader(args.checkpoint, 'cpu')
    reports = []
    for threshold in args.thresholds:
        policy = checkpoint_policy(ck, model.cfg, confidence=threshold, n_commit=args.n_commit)
        rows, _ = run_policy(args.checkpoint, sources, ['calibration'], policy, checkpoint_loader=checkpoint_loader,
                             tracer_class=tracer_class, device=args.device, max_len=args.max_len, batch=args.batch,
                             forward_chunk=args.forward_chunk)
        summary = grouped_summary(rows)
        report = dict(threshold=threshold, operating_policy=policy.to_dict(), summary=summary)
        write(out/f'calibration_c{threshold:.3f}.json', dict(report, rows=rows))
        reports.append(report)
        overall = summary['all']
        print(json.dumps(dict(threshold=threshold, length_precision=overall['length_precision'],
                              coverage=overall['length_weighted_coverage'], wrong=overall['wrong_length'])), flush=True)
    eligible = [r for r in reports if r['summary']['all']['length_precision'] >= .95 and r['summary']['all']['scored_length'] > 0]
    outcome = dict(checkpoint=str(Path(args.checkpoint).resolve()), split='calibration', max_len=args.max_len,
                   manifest_sha256=manifest_digest(sources, ['calibration']), thresholds=list(args.thresholds),
                   precision_requirement=.95,
                   sweep=[dict(threshold=r['threshold'], **{k: r['summary']['all'][k] for k in
                          ('length_precision', 'length_weighted_coverage', 'wrong_length', 'correct_length')})
                          for r in reports])
    if not eligible:
        outcome['selected'] = None
        outcome['message'] = 'No calibration operating point reaches 95% scored precision; no policy selected'
        write(out/'calibration_outcome.json', outcome)
        print(outcome['message'])
        return None
    selected = max(eligible, key=lambda r: r['summary']['all']['length_weighted_coverage'])
    outcome.update(selected=selected['threshold'], operating_policy=selected['operating_policy'],
                   checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    write(out/'calibration_outcome.json', outcome)
    write(out/'selection.json', outcome)
    return outcome


def run(args, sources, *, checkpoint_loader, tracer_class):
    if 'final' in args.splits and args.policy is None and args.confidence is None:
        raise ValueError('Final evaluation needs an explicit --confidence or a calibration --policy')
    model, _, _, _, ck = checkpoint_loader(args.checkpoint, 'cpu')
    if args.policy is not None:
        selection = json.loads(Path(args.policy).read_text())
        if selection.get('split') != 'calibration' or selection.get('selected') is None:
            raise ValueError('The policy file is not a calibration selection')
        policy = checkpoint_policy(ck, model.cfg, confidence=selection['operating_policy']['confidence'],
                                   n_commit=selection['operating_policy']['n_commit'])
    else:
        policy = checkpoint_policy(ck, model.cfg, confidence=args.confidence, n_commit=args.n_commit)
    started = time.monotonic()
    rows, ck = run_policy(args.checkpoint, sources, args.splits, policy, checkpoint_loader=checkpoint_loader,
                          tracer_class=tracer_class, device=args.device, max_len=args.max_len, batch=args.batch,
                          forward_chunk=args.forward_chunk)
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), step=ck.get('step'), operating_policy=policy.to_dict(),
                  policy_source=args.policy or 'explicit', max_len=args.max_len, splits=list(args.splits),
                  sources=[s['name'] for s in sources], manifest_sha256=manifest_digest(sources, args.splits),
                  seconds=time.monotonic()-started, summary=grouped_summary(rows), rows=rows)
    write(args.out, result)
    overall = result['summary']['all']
    print(json.dumps({k: overall[k] for k in ('n', 'length_precision', 'length_weighted_coverage', 'correct_length',
                                               'wrong_length', 'divergence_count', 'premature_stop', 'distance_events')}))
    return result


def paired_report(base, new, repeats=4000, seed=0):
    """Same-seed paired differences (new minus base) with fiber-resampled 95% intervals."""
    key = lambda r: (r['source'], r['split'], r['fiber'], r['t0'], r['sign'])
    a, b = {key(r): r for r in base['rows']}, {key(r): r for r in new['rows']}
    if a.keys() != b.keys():
        raise ValueError('Paired comparison requires identical seeds')
    if base['manifest_sha256'] != new['manifest_sha256']:
        raise ValueError('Paired comparison requires the same frozen seed manifests')
    groups = {'all': sorted(a)}
    for k in sorted(a):
        groups.setdefault(k[0], []).append(k)
        groups.setdefault(f'{k[0]}/{k[1]}', []).append(k)
    value = lambda row, metric: float(row.get(metric, 0.))
    rng = np.random.default_rng(seed)
    report = {}
    for name, keys in groups.items():
        fibers = sorted({k[:3] for k in keys})
        members = {f: [k for k in keys if k[:3] == f] for f in fibers}
        deltas = np.array([[value(b[k], m)-value(a[k], m) for m in PAIRED_METRICS] for k in keys])
        index = {k: i for i, k in enumerate(keys)}
        draws = []
        for _ in range(repeats):
            chosen = [index[k] for f in rng.choice(len(fibers), len(fibers)) for k in members[fibers[f]]]
            draws.append(deltas[chosen].sum(0))
        lo, hi = np.quantile(np.asarray(draws), [.025, .975], axis=0)
        report[name] = {m: dict(base=float(sum(value(a[k], m) for k in keys)), new=float(sum(value(b[k], m) for k in keys)),
                                delta=float(deltas[:, i].sum()), ci95=[float(lo[i]), float(hi[i])])
                        for i, m in enumerate(PAIRED_METRICS)}
        report[name]['traces'] = len(keys)
    return report


def compare(args):
    base, new = (json.loads(Path(p).read_text()) for p in (args.base, args.new))
    report = dict(base=dict(checkpoint=base['checkpoint'], operating_policy=base['operating_policy']),
                  new=dict(checkpoint=new['checkpoint'], operating_policy=new['operating_policy']),
                  paired=paired_report(base, new))
    if args.out:
        write(args.out, report)
    for name, metrics in report['paired'].items():
        print(f"== {name} ({metrics['traces']} traces)")
        for metric in PAIRED_METRICS:
            row = metrics[metric]
            print(f"  {metric:22s} {row['base']:10.1f} -> {row['new']:10.1f}  delta {row['delta']:+9.1f}"
                  f"  [{row['ci95'][0]:+.1f}, {row['ci95'][1]:+.1f}]")
    return report


def parser(description=__doc__):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = ap.add_subparsers(dest='command', required=True)
    for name in ('calibrate', 'run'):
        p = commands.add_parser(name)
        p.add_argument('--checkpoint', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--device', default='cuda')
        p.add_argument('--batch', type=int, default=24, help='Concurrent traces per tracer call')
        p.add_argument('--forward-chunk', type=int, default=0, help='Rows per model forward call; 0 = whole batch')
        p.add_argument('--max-len', type=float, default=2000.)
        p.add_argument('--n-commit', type=int, help='Default: the checkpoint operating policy')
        p.add_argument('--sources', nargs='+', help='Dataset sources (default: all)')
        if name == 'calibrate':
            p.add_argument('--thresholds', type=float, nargs='+', default=THRESHOLDS)
        else:
            p.add_argument('--splits', nargs='+', choices=SPLITS, default=['final'])
            p.add_argument('--confidence', type=float)
            p.add_argument('--policy', help='calibration selection.json')
    p = commands.add_parser('compare')
    p.add_argument('base')
    p.add_argument('new')
    p.add_argument('--out')
    return ap


def main(argv=None, *, checkpoint_loader, tracer_class, source_loader):
    args = parser().parse_args(argv)
    if args.command == 'compare':
        return compare(args)
    sources = source_loader(args)
    if args.command == 'calibrate':
        return calibrate(args, sources, checkpoint_loader=checkpoint_loader, tracer_class=tracer_class)
    return run(args, sources, checkpoint_loader=checkpoint_loader, tracer_class=tracer_class)
