"""Identity evaluation for direct followers: all seeds and a frozen ambiguous subset.

  # once, before comparing checkpoints (refuses to overwrite)
  python -m vesuvius.neural_tracing.fiber_follow.regression.identity_eval subset --out EVAL/ambiguous.json
  # any direct checkpoint, including the baseline
  python -m vesuvius.neural_tracing.fiber_follow.regression.identity_eval rollouts \\
      --checkpoint output/direct_refined_run1/last.pt --subset EVAL/ambiguous.json --out EVAL/baseline
  python -m vesuvius.neural_tracing.fiber_follow.regression.identity_eval rollouts \\
      --checkpoint RUN/last.pt --subset EVAL/ambiguous.json --out EVAL/identity --baseline-rows EVAL/baseline/rows_c0.500.json

Rollout scoring is the shared monitor scoring (``score_trace`` with a capped
denominator). Identity metrics are versioned separately (``identity_version``).
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from vesuvius.neural_tracing.fiber_follow.shared.data import ZBand, fiber_manifest, load_fibers, split_fibers
from vesuvius.neural_tracing.fiber_follow.shared.evaluate import evaluate, trace_events
from vesuvius.neural_tracing.fiber_follow.shared.experiment import jsonable, paired_bootstrap, read_manifest, rollout_summary
from vesuvius.neural_tracing.fiber_follow.shared.geometry import interp_at
from vesuvius.neural_tracing.fiber_follow.shared.trace import TraceParams
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.regression.data import DirectTracer, traversal
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig
from vesuvius.neural_tracing.fiber_follow.regression.train import load_checkpoint, validate_volume_source

IDENTITY_VERSION = 1
FIBERS = '/mnt/raid_nvme/spiral_dataset_working/fibers'
MANIFEST = str(Path(__file__).parents[1]/'output'/'single_path_v11_preparation'/'seeds.json')


def load_split(fibers_dir, manifest, grid_scale=8.):
    fibers = load_fibers(fibers_dir, grid_scale=grid_scale)
    _, val = split_fibers(fibers, ZBand(45000/grid_scale, 48500/grid_scale))
    if fiber_manifest(val) != manifest['fibers']:
        raise ValueError('Validation geometry differs from frozen manifest')
    return fibers, val


def other_fibers(fibers):
    points = np.concatenate([f.points for f in fibers])
    names = np.concatenate([np.full(len(f.points), i) for i, f in enumerate(fibers)])
    return cKDTree(points), names, {f.name: i for i, f in enumerate(fibers)}


def make_subset(args):
    """Seeds whose annotated continuation passes near a different annotated fiber."""
    if args.out.exists():
        raise FileExistsError(f'{args.out} is frozen; use a new path')
    manifest = read_manifest(args.manifest)
    fibers, val = load_split(args.fibers, manifest)
    tree, ids, index = other_fibers(fibers)
    definition = dict(identity_version=IDENTITY_VERSION, max_len=args.max_len, radius=args.radius, spacing=4.,
                      manifest_sha256=manifest['sha256'], fibers=fiber_manifest(fibers))
    subset = {}
    for split in ('calibration', 'final'):
        rows = []
        for n, seed in enumerate(manifest[split]):
            f = val[seed['fiber']]
            own = index[f.name]
            arcs = seed['t']+seed['sign']*np.arange(0, args.max_len+1e-9, 4.)
            arcs = arcs[(arcs >= 0) & (arcs <= f.length)]
            contact = None
            for arc, point in zip(arcs, interp_at(f.points, f.s, arcs)):
                near = [i for i in tree.query_ball_point(point, args.radius)
                        if ids[i] != own and fibers[ids[i]].source_hash != f.source_hash]
                if near:
                    contact = dict(arc=float(abs(arc-seed['t'])), neighbor=fibers[ids[near[0]]].name)
                    break
            if contact:
                rows.append(dict(index=n, fiber=seed['fiber'], t=seed['t'], sign=seed['sign'], **contact))
        subset[split] = rows
    payload = dict(definition=definition, **subset)
    payload['sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: len(subset[k]) for k in subset} | dict(of={k: len(manifest[k]) for k in subset})))


def read_subset(path, manifest):
    saved = json.loads(Path(path).read_text())
    digest = hashlib.sha256(json.dumps({k: v for k, v in saved.items() if k != 'sha256'}, sort_keys=True).encode()).hexdigest()
    if digest != saved['sha256'] or saved['definition']['manifest_sha256'] != manifest['sha256']:
        raise ValueError('Ambiguous subset was modified or made for another manifest')
    return saved


def auc(positive, negative):
    """Mann-Whitney AUC with tied ranks."""
    positive, negative = np.asarray(positive), np.asarray(negative)
    if not len(positive) or not len(negative):
        return None
    values = np.r_[positive, negative]
    order = values.argsort(kind='stable')
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values)+1)
    for v in np.unique(values):
        tie = values == v
        ranks[tie] = ranks[tie].mean()
    return float((ranks[:len(positive)].sum()-len(positive)*(len(positive)+1)/2)/(len(positive)*len(negative)))


def identity_rows(rows, paths, decisions, seeds, val, fibers, tree, ids, index, tol=3., window=32, support=8):
    """Switches onto another annotated fiber, false stops, and per-decision departure labels."""
    for row, path, logged, seed in zip(rows, paths, decisions, seeds):
        f = val[seed['fiber']]
        lengths, _, _, end, crossing = trace_events(path, f, seed['t'], seed['sign'])
        departed = end < len(path) and crossing is None
        departure = float(lengths[max(end-1, 0)]) if departed else np.inf
        switch = None
        if departed:
            after = np.asarray(path[end:end+window])
            owners = {}
            for point in after:
                for i in tree.query_ball_point(point, tol):
                    if ids[i] != index[f.name] and fibers[ids[i]].source_hash != f.source_hash:
                        owners[ids[i]] = owners.get(ids[i], 0)+1
            best = max(owners.items(), key=lambda kv: kv[1], default=None)
            if best and best[1] >= support:
                switch = fibers[best[0]].name
        available = f.length-seed['t'] if seed['sign'] > 0 else seed['t']
        row.update(identity_version=IDENTITY_VERSION, departure_length=departure if departed else None,
                   identity_switch=switch is not None, switched_to=switch,
                   false_stop=bool(row['reason'] == 'confidence' and not departed and not row['reached_end']),
                   stop_position=float(lengths[-1]) if row['reason'] == 'confidence' else None,
                   on_track_confidence=[d['confidence'] for d in logged if d['travelled'] < min(departure, available)],
                   departed_confidence=[d['confidence'] for d in logged if departure <= d['travelled'] < available])
    return rows


def identity_summary(rows):
    wrong = np.array([r['offtrack'] for r in rows])
    on = [c for r in rows for c in r['on_track_confidence']]
    off = [c for r in rows for c in r['departed_confidence']]
    stops = [r['stop_position'] for r in rows if r['false_stop']]
    return dict(identity_version=IDENTITY_VERSION, traces=len(rows),
                identity_switches=int(sum(r['identity_switch'] for r in rows)),
                identity_switch_rate=float(np.mean([r['identity_switch'] for r in rows])) if rows else None,
                departures=int(sum(r['departure_length'] is not None for r in rows)),
                incorrect_length=float(wrong.sum()),
                incorrect_length_quantiles=dict(zip(('p50', 'p90', 'p95', 'max'), np.quantile(wrong, [.5, .9, .95, 1]).tolist())) if rows else None,
                decision_auc_on_track_vs_departed=auc(on, off), on_track_decisions=len(on), departed_decisions=len(off),
                departed_decisions_confident=float(np.mean(np.asarray(off) >= .5)) if off else None,
                false_stops=len(stops), false_stop_positions=stops)


def run_rollouts(args):
    manifest = read_manifest(args.manifest)
    subset = read_subset(args.subset, manifest)
    model, crop, nh, spec, ck = load_checkpoint(args.checkpoint, args.device)
    validate_volume_source(spec, manifest)
    fibers, val = load_split(args.fibers, manifest, spec.grid_scale)
    tree, ids, index = other_fibers(fibers)
    seeds = manifest[args.split]
    ambiguous = {r['index'] for r in subset[args.split]}
    args.out.mkdir(parents=True, exist_ok=True)
    tracer = DirectTracer(model, FiberVolume(spec), crop, nh, TraceParams(max_len=args.max_len, confidence=.5,
                          n_commit=ck.get('n_commit', 4)), device=args.device)
    curve = []
    try:
        for threshold in args.thresholds:
            tracer.p.confidence = threshold
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            rows, paths, logged = logged_rollouts(tracer, val, seeds, args.max_len)
            seconds = time.perf_counter()-started
            rows = identity_rows(rows, paths, logged, seeds, val, fibers, tree, ids, index)
            for n, row in enumerate(rows):
                row.update(seed_index=n, ambiguous=n in ambiguous)
            report = dict(checkpoint=str(Path(args.checkpoint).resolve()), architecture=ck['architecture'], step=ck.get('step'),
                          checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                          split=args.split, threshold=threshold, max_len=args.max_len, subset_sha256=subset['sha256'],
                          seconds=seconds, decisions=sum(map(len, logged)), seconds_per_decision=seconds/max(1, sum(map(len, logged))),
                          peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None)
            for name, members in (('all', rows), ('ambiguous', [r for r in rows if r['ambiguous']])):
                report[name] = dict(rollout=rollout_summary(members) if members else None, identity=identity_summary(members))
            if args.baseline_rows:
                base = json.loads(Path(str(args.baseline_rows).format(threshold=threshold)).read_text())
                for name, keep in (('all', lambda r: True), ('ambiguous', lambda r: r['ambiguous'])):
                    report[name]['paired_fiber_bootstrap_delta_95ci'] = paired_bootstrap(
                        [r for r in base if keep(r)], [r for r in rows if keep(r)])
            tag = f'c{threshold:.3f}'
            (args.out/f'report_{tag}.json').write_text(json.dumps(report, indent=2, default=jsonable))
            (args.out/f'rows_{tag}.json').write_text(json.dumps(rows, default=jsonable))
            curve.append(dict(threshold=threshold, **{name: dict(coverage=report[name]['rollout']['length_weighted_coverage'],
                                                                 precision=report[name]['rollout']['length_precision'])
                                                      for name in ('all', 'ambiguous') if report[name]['rollout']}))
            print(json.dumps(dict(threshold=threshold, **{n: {k: report[n]['identity'][k] for k in
                  ('identity_switches', 'incorrect_length', 'decision_auc_on_track_vs_departed', 'false_stops')}
                  for n in ('all', 'ambiguous')})), flush=True)
    finally:
        tracer.close()
    (args.out/'coverage_risk.json').write_text(json.dumps(curve, indent=2))


class _Decisions:
    """``evaluate``'s per-decision hook: first-point confidence and distance travelled."""
    def __init__(self):
        self.rows = []

    def start_batch(self, fibers, chunk):
        pass

    def __call__(self, i, state):
        self.rows.append(dict(travelled=float(state['travelled']), confidence=float(state['confidence'][0])))


def logged_rollouts(tracer, val, seeds, max_len):
    """One seed at a time, so every decision is attributed to its trace."""
    rows, paths, logged = [], [], []
    for seed in seeds:
        decisions = _Decisions()
        result, _ = evaluate(tracer, val, [seed], batch=1, coverage_max_len=max_len, history_audit=decisions,
                             on_trace=lambda s, p, r: paths.append(p))
        rows.extend(result)
        logged.append(decisions.rows)
    return rows, paths, logged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('subset', 'rollouts'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--manifest', default=MANIFEST)
    ap.add_argument('--fibers', default=FIBERS)
    ap.add_argument('--subset', type=Path)
    ap.add_argument('--checkpoint')
    ap.add_argument('--split', default='calibration', choices=('calibration', 'final'))
    ap.add_argument('--max-len', type=float, default=400.)
    ap.add_argument('--radius', type=float, default=6., help='Subset: proximity to another annotated fiber')
    ap.add_argument('--thresholds', type=float, nargs='+', default=(.5,))
    ap.add_argument('--baseline-rows', help='Paired comparison; may contain {threshold}')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    if args.mode == 'subset':
        return make_subset(args)
    if args.subset is None or args.checkpoint is None:
        ap.error('rollouts need --subset and --checkpoint')
    return run_rollouts(args)


if __name__ == '__main__':
    main()
