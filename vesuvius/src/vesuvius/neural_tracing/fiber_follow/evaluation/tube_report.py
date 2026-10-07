"""Per-source tube metrics (tube_scoring) for long_trace_audit folders: tables, paired differences, operating curves.

Usage (from fiber_follow/):
  python -m vesuvius.neural_tracing.fiber_follow.evaluation.tube_report LABEL=FOLDER[,FOLDER2] ... \\
      --config DATASET_CONFIG.json [--replay 0.6 0.7 0.8] [--out report.json]

Shards of one run are merged (comma-separated folders). Rows without tube scores (older runs), or all rows with
--rescore, are scored from their saved traces; --config also supplies the neighbour fibers for switch labels and
each source's voxel size (rates per 10 mm; per 1000 trace voxels without it). The first run is the reference for
paired differences. --replay rescores each run at higher confidence thresholds from its recorded gate confidences
(tube_scoring.replay_stop; approximate for threshold-dependent proposal selection).
"""
from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import tube_scoring
from .neighbor_fibers import NeighborFibers
from ..data.datasets import read_dataset_config
from ..shared.experiment import jsonable

SOURCES = ('paris4', '0175A_5mm_v1', '1447_5mm_v1', 'paris4_afv_central')
HEADLINE = (('seeds_mistake', 'seeds: mistake, kept on', 3), ('seeds_stopped_early', 'seeds: stopped early', 3),
            ('seeds_reached_end', 'seeds: reached end', 3), ('seeds_other', 'seeds: other', 3),
            ('distance_per_mistake', 'mm traced per mistake', 0), ('continued_after_mistake', 'mm traced after mistake', 1), ('annotation_per_seed', 'annotation mm per seed', 1),
            ('losses_rate', 'identity losses', 3), ('switches_rate', '  of which switches', 3),
            ('premature_stops_rate', 'premature stops', 3), ('excursions_rate', 'excursions', 3),
            ('precision', 'precision', 3), ('fiber_coverage', 'fiber coverage', 3),
            ('normal_p90', 'normal offset p90', 1), ('width_p90', 'width offset p90', 1),
            ('reached_end', 'reached end', 3), ('unscored_share', 'unscored share', 3),
            ('losses', 'losses (count)', 0), ('premature_stops', 'premature (count)', 0))
_WORK = {}


def load_run(folders):
    rows, protocol = [], None
    for folder in folders.split(','):
        folder = Path(folder)
        protocol = protocol or json.loads((folder/'protocol.json').read_text())
        for rows_file in sorted(folder.glob('*/*/rows_*.json')):
            for row in json.loads(rows_file.read_text()):
                row['npz'] = str(rows_file.parent/f"trace_{row['seed_index']:03d}.npz")
                rows.append(row)
    return rows, protocol


def trace_inputs(row):
    z = np.load(row['npz'])
    known = bool(row.get('endpoint_known', False))
    fiber = SimpleNamespace(points=z['annotation'], s=z['annotation_s'], length=float(z['annotation_s'][-1]),
                            endpoint_stop=(known, False) if row['sign'] < 0 else (False, known), name=row['fiber_name'])
    return z, fiber


def init_worker(neighbor_configs):
    """Per process: the dataset-config entries sharing each source's volume; catalogs open on first use."""
    _WORK.clear()
    _WORK.update(configs=neighbor_configs or {}, neighbors={})


def neighbors_for(source):
    if source not in _WORK.get('configs', {}):
        return None
    if source not in _WORK['neighbors']:
        _WORK['neighbors'][source] = NeighborFibers.from_configs(_WORK['configs'][source])
    return _WORK['neighbors'][source]


def score_row(job):
    row, stop, label = job
    z, fiber = trace_inputs(row)
    foreign = None
    neighbors = neighbors_for(row['source']) if label else None
    if neighbors is not None:
        foreign = neighbors.foreign(neighbors.code(row['fiber_name']), fiber.points)
    return tube_scoring.score_tube(z['path'], fiber, row['t0'], row['sign'], z['travelled'], z['frame'], row['reason'],
                                   foreign=foreign, stop_at=stop)


def score_rows(rows, stops, neighbor_configs, workers):
    """Tube scores of ``rows`` (``stops``: replayed stop lengths or None); switch labels with ``neighbor_configs``."""
    jobs = [(r, s, bool(neighbor_configs)) for r, s in zip(rows, stops)]
    if workers > 1:
        # Workers do not inherit module state under spawn/forkserver: hand them the configs explicitly.
        with Pool(workers, initializer=init_worker, initargs=(neighbor_configs,)) as pool:
            return pool.map(score_row, jobs, chunksize=16)
    init_worker(neighbor_configs)
    return [score_row(j) for j in jobs]


def cached_scores(rows, stops, neighbor_configs, workers, key, refresh=False):
    """``score_rows`` with a per-run-folder cache (tube_cache/<key>_v<VERSION>.json), so unchanged runs load instantly."""
    trace_key = lambda r: f"{r['source']}/{Path(r['npz']).parent.name}/{r['seed_index']}"
    folders = {}
    for i, r in enumerate(rows):
        folders.setdefault(Path(r['npz']).parents[2], []).append(i)
    scores = [None]*len(rows)
    for folder, members in folders.items():
        path = folder/'tube_cache'/f'{key}_v{tube_scoring.VERSION}.json'
        cache = {} if refresh or not path.exists() else json.loads(path.read_text())
        todo = [i for i in members if trace_key(rows[i]) not in cache]
        if todo:
            fresh = score_rows([rows[i] for i in todo], [stops[i] for i in todo], neighbor_configs, workers)
            cache.update({trace_key(rows[i]): score for i, score in zip(todo, fresh)})
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(cache, default=jsonable))
        for i in members:
            scores[i] = cache[trace_key(rows[i])]
    return scores


def replay_stops(rows, threshold, gate_plane):
    stops = []
    for row in rows:
        z = np.load(row['npz'])
        stops.append(tube_scoring.replay_stop(z['travelled'], z['confidence'][:, gate_plane-1], threshold,
                                              final_was_stop=row['reason'] not in tube_scoring.EVALUATION_STOPS))
    return stops


TRACE_FIELDS = ('outcome', 'end', 'reason', 'verified_length', 'wrong_length', 'total_length', 'available', 'remaining',
                'loss_at', 'loss_kind', 'continued_after_loss', 'excursions', 'reached_end')


def write_traces(path, runs):
    """One row per trace and run; ``image`` is the run's strip render (render.py) when it exists."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('run', 'source', 'cohort', 'seed_index', 'fiber_name', 'sign', 't0')+TRACE_FIELDS+('npz', 'image'))
        for label, (rows, _) in runs.items():
            for r in rows:
                npz = Path(r['npz'])
                image = npz.parents[2]/'images'/f"{r['source']}_{r.get('cohort', npz.parent.name)}_{r['seed_index']:03d}.png"
                writer.writerow((label, r['source'], r.get('cohort', npz.parent.name), r['seed_index'], r['fiber_name'], r['sign'],
                                 r['t0'])+tuple(r['tube'].get(k) for k in TRACE_FIELDS)
                                + (str(npz), str(image) if image.exists() else ''))


def cell(value, digits):
    v = value[0]
    if len(value) == 1 or digits == 0:
        return f'{v:.{digits}f}'
    return f'{v:.{digits}f} [{value[1]:.{digits}f}, {value[2]:.{digits}f}]'


def print_tables(report, labels):
    for source, entry in report['sources'].items():
        unit = next(iter(entry['runs'].values()))['unit']
        um = next(iter(entry['runs'].values()))['um_per_voxel']
        print(f"\n== {source}  ({next(iter(entry['runs'].values()))['fibers']} fibers; rates {unit}; offsets in "
              f"{'um' if um else 'trace voxels'})")
        width = 26
        print(f"{'':22}" + ''.join(f'{label:>{width}}' for label in labels if label in entry['runs'])
              + ''.join(f"{'diff '+l:>{width}}" for l in entry.get('paired', {})))
        for key, name, digits in HEADLINE:
            cells = [cell(entry['runs'][l]['metrics'][key], digits) for l in labels if l in entry['runs']]
            for label, diff in entry.get('paired', {}).items():
                d = diff['metrics'][key]
                star = '*' if d[1] > 0 or d[2] < 0 else ''
                cells.append(f'{d[0]:+.{max(digits, 1)}f}{star}')
            print(f'{name:22}' + ''.join(f'{c:>{width}}' for c in cells))
        runs = [entry['runs'][l] for l in labels if l in entry['runs']]
        for distance in runs[0]['mistake_free']:
            cells = ['-' if r['mistake_free'][distance] is None else
                     f"{r['mistake_free'][distance][0]:.3f} (n={r['mistake_free'][distance][1]})" for r in runs]
            print(f"{'no mistake by '+distance:22}" + ''.join(f'{c:>{width}}' for c in cells))
        cells = ['-' if r['mistake_at_median'] is None else f"{r['mistake_at_median']:.1f}" for r in runs]
        print(f"{'median mistake at':22}" + ''.join(f'{c:>{width}}' for c in cells))
    for source, entry in report['sources'].items():
        if 'curves' not in entry:
            continue
        print(f'\n== {source} operating curves (replayed thresholds; rates {next(iter(entry["runs"].values()))["unit"]})')
        print(f"{'run':16}{'threshold':>10}{'losses':>10}{'premature':>11}{'precision':>11}{'coverage':>10}")
        for label, curve in entry['curves'].items():
            for threshold, m in curve:
                print(f"{label:16}{threshold:>10.2f}{m['losses_rate']:>10.3f}{m['premature_stops_rate']:>11.3f}"
                      f"{m['precision']:>11.3f}{m['fiber_coverage']:>10.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+', help='LABEL=FOLDER[,FOLDER2]')
    ap.add_argument('--config', help='dataset config of the evaluation (voxel sizes, neighbour fibers)')
    ap.add_argument('--rescore', action='store_true', help='rescore every row from its saved trace, ignoring the cache')
    ap.add_argument('--no-switch-labels', action='store_true')
    ap.add_argument('--replay', nargs='*', type=float, default=[])
    ap.add_argument('--gate-plane', type=int, help="override the protocol's gate plane (runs before it was recorded: 16)")
    ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out')
    ap.add_argument('--traces', help='CSV of every trace: run, source, seed, outcome and tube scores, npz and image paths')
    args = ap.parse_args(argv)
    configs, neighbor_configs = {}, None
    if args.config:
        document, _ = read_dataset_config(args.config)
        configs = {s['name']: s for s in document['sources']}
        if not args.no_switch_labels:
            same = lambda a, b: (str(a['ct']).rstrip('/') == str(b['ct']).rstrip('/')
                                 and float(a.get('grid_scale', 8.)) == float(b.get('grid_scale', 8.)))
            neighbor_configs = {name: [s for s in document['sources'] if same(s, c)] for name, c in configs.items()}
    labels, runs = [], {}
    for spec in args.runs:
        label, folders = spec.split('=', 1)
        rows, protocol = load_run(folders)
        missing = [i for i, r in enumerate(rows) if args.rescore or r.get('tube', {}).get('version') != tube_scoring.VERSION]
        if missing:
            print(f'{label}: scoring {len(missing)} of {len(rows)} traces (cached where unchanged)', flush=True)
            scores = cached_scores([rows[i] for i in missing], [None]*len(missing), neighbor_configs, args.workers,
                                   'base_labelled' if neighbor_configs else 'base', refresh=args.rescore)
            for i, score in zip(missing, scores):
                rows[i]['tube'] = score
        labels.append(label)
        runs[label] = (rows, protocol)
    def run_info(spec, protocol):
        policy = protocol.get('operating_policy', {})
        return dict(folders=spec.split('=', 1)[1], checkpoint=protocol.get('checkpoint'), step=protocol.get('step'),
                    confidence=policy.get('confidence'), n_commit=policy.get('n_commit'), gate=policy.get('gate'),
                    refit_retry=bool(protocol.get('args', {}).get('refit_retry', False)),
                    batch=protocol.get('batch'), annotation_end_stop=protocol.get('annotation_end_stop'),
                    seeds_sha256=protocol.get('seeds_sha256'))
    report = dict(settings=tube_scoring.settings(),
                  runs={l: run_info(s, runs[l][1]) for l, s in zip(labels, args.runs)}, sources={})
    for source in SOURCES:
        um = tube_scoring.um_per_voxel(configs[source]) if source in configs else None
        entry = dict(runs={}, paired={})
        for label in labels:
            members = [r for r in runs[label][0] if r['source'] == source]
            if members:
                entry['runs'][label] = tube_scoring.summarize_tube(members, um, args.boot)
        if not entry['runs']:
            continue
        reference = labels[0]
        for label in labels[1:]:
            a = [r for r in runs[reference][0] if r['source'] == source]
            b = [r for r in runs[label][0] if r['source'] == source]
            if a and b:
                entry['paired'][label] = tube_scoring.compare_tube(a, b, um, args.boot)
        report['sources'][source] = entry
    for label in labels:
        rows, protocol = runs[label]
        if not args.replay:
            break
        if protocol.get('args', {}).get('refit_retry'):
            print(f'{label}: refit retry was on; replay skipped')
            continue
        gate_plane = args.gate_plane or protocol.get('gate_plane', 16)
        base = protocol['operating_policy']['confidence']
        for threshold in sorted(t for t in args.replay if t >= base):
            stops = replay_stops(rows, threshold, gate_plane)
            scores = cached_scores(rows, stops, None, args.workers, f'replay{threshold:g}_gate{gate_plane}', refresh=args.rescore)
            replayed = [dict(r, tube=s) for r, s in zip(rows, scores)]
            for source, entry in report['sources'].items():
                members = [r for r in replayed if r['source'] == source]
                um = tube_scoring.um_per_voxel(configs[source]) if source in configs else None
                if members:
                    metrics = tube_scoring.summarize_tube(members, um, boot=0)['metrics']
                    entry.setdefault('curves', {}).setdefault(label, []).append((threshold, {k: v[0] for k, v in metrics.items()}))
            print(f'{label}: replayed threshold {threshold}', flush=True)
    print_tables(report, labels)
    if args.traces:
        write_traces(args.traces, runs)
    if args.out:
        Path(args.out).write_text(json.dumps(report, default=jsonable, indent=1))


if __name__ == '__main__':
    main()
