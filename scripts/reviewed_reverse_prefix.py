"""Matched trace-only reverse-window experiments; shared frozen inference cache."""
import copy
import json
import time
from pathlib import Path
from reviewed_fiber_evaluation import ROOT, REPO, execute_evaluation
from lasagna.inference_provenance import atomic_write, sha256_file
from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import consistency_report

MODEL = 'halfsigma_sideband_best_raw_20261002'
EXPERIMENT = ROOT/'reverse_prefix_40500_v1'
VARIANTS = [('window256_terminal', 256, False),
            ('prefix256_terminal', 256, True),
            ('prefix128_terminal', 128, True),
            ('prefix512_terminal', 512, True)]


def workload(base, name, interval, prefix, experiment=EXPERIMENT):
    config = copy.deepcopy(base)
    if any('reverse' in arg or 'consistency' in arg for arg in config['evaluation']['trace_args']):
        raise ValueError('expected unfiltered baseline')
    config['output_root'] = str(experiment/name)
    config['evaluation']['gpu_manager']['label'] = f'reviewed-{name}'
    config['evaluation']['trace_args'] += ['--replay-reverse-interval-base', str(interval),
                                         '--replay-reverse-all-endpoints']
    if prefix:
        config['evaluation']['trace_args'].append('--replay-reverse-to-start')
    return config


def report(experiment=EXPERIMENT, variants=VARIANTS, extra_folders=(),
           report_name='reviewed_reverse_prefix_40500_20261002', note='', action='reviewed_reverse_prefix'):
    results = {}
    folders = [('unfiltered', ROOT/MODEL), ('window256_original', ROOT/'reverse256_v1'/MODEL)]
    folders += list(extra_folders)
    folders += [(name, experiment/name) for name, _, _ in variants]
    for name, folder in folders:
        path = folder/'full/summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        if not summary.get('completed'):
            continue
        if summary['cases'] != 182:
            raise ValueError('cohort changed')
        result = consistency_report(summary)
        result['protocol'] = name
        result['summary_path'] = str(path)
        # Replay geometry keeps pre-rollback terminal failure evidence.
        conversions = 0
        for replay in (folder/'full/cases').glob('*/trace.json'):
            document = json.loads(replay.read_text())
            conversions += sum('filtered_distance_failure_xyz_arc' in s for s in document['replay']['segments'])
        result['terminal_distance_failures_filtered'] = conversions
        timing = folder/'timing.json'
        if timing.exists():
            result['wall_seconds'] = json.loads(timing.read_text())['wall_seconds']
        inference = folder/'full/inference_summary.json'
        if inference.exists():
            result['inference_summary_path'] = str(inference)
            inference_data = json.loads(inference.read_text())
            result['new_inference_tiles'] = sum(s['inferred_tiles'] for s in inference_data['sources'].values())
            result['new_published_chunks'] = sum(s['published_chunks'] for s in inference_data['sources'].values())
        results[name] = result
    atomic_write(experiment/'comparison.json', results, durable=False)
    target = REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks'/report_name
    atomic_write(target.with_suffix('.json'), results, durable=False)
    lines = ['# Reverse checks to segment start — raw u40500', '',
             'Same frozen model/cache, 91 reviewed fibers / 182 directions, beam8, reverse ellipse20/40. '
             'Window checks only the latest accepted interval; prefix checks the entire current forward segment. '
             'All terminal endpoints are checked unless already checked at the same point. One-point segments cannot be checked. '
             'Intervals are forward arc travel in base voxels, checked at the first committed step reaching the interval. No convergence shortcut.', '', note, '',
             'Acceptance uses forward trace geometry only, never GT. However GT still determines ordinary evaluation '
             'failure endpoints and reseeding. Thus terminal failure-to-filter conversions are oracle-timed diagnostics, '
             'not demonstrated deployable GT-free detection. Retained coverage excludes rejected/skipped intervals. '
             'Ordinary restart episodes use existing80-voxel grouping; filtered stops are ungrouped.', '',
             '| Variant | Ordinary restarts | Filtered stops | Total interruptions | Retained % | Reverse checks | Terminal failure→filtered | Wall s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        c = r['combined']
        lines.append(f'| {name} | {c["ordinary_restarts"]} | {c["filtered_stops"]} | '
                     f'{c["ordinary_restarts"]+c["filtered_stops"]} | {c["retained_percent"]:.4f} | '
                     f'{c["reverse_checks"]} | {r["terminal_distance_failures_filtered"]} | '
                     f'{round(r["wall_seconds"], 1) if "wall_seconds" in r else "—"} |')
    lines += ['', '## Per scroll', '', '| Scroll / variant | Ordinary restarts | Filtered stops | Retained % |', '|---|---:|---:|---:|']
    for scroll in results['unfiltered']['scrolls']:
        for name, r in results.items():
            c = r['scrolls'][scroll]
            lines.append(f'| {scroll} / {name} | {c["ordinary_restarts"]} | {c["filtered_stops"]} | {c["retained_percent"]:.4f} |')
    lines += ['', '## Run', '', f'Set control action `{action}`, then:', '',
              '```bash', 'bash /home/hendrik/villa3/scripts/run_training_failure_mining', '```', '',
              f'Workloads, output and timing: {experiment}. Recorded wall times include startup, cache filling and resource waits; these are not isolated tracer-speed benchmarks.']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    return results


def evaluate_config(config, name):
    output = Path(config['output_root'])
    path = output/'workload.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('immutable experiment changed')
    atomic_write(path, config, durable=False)
    start = time.monotonic()
    execute_evaluation(path, config, name)
    if not (output/'timing.json').exists():
        atomic_write(output/'timing.json', dict(wall_seconds=time.monotonic()-start), durable=False)


def main():
    base = json.loads((ROOT/MODEL/'workload.json').read_text())
    if sha256_file(base['checkpoint']) != base['checkpoint_sha256']:
        raise ValueError('frozen checkpoint changed')
    for name, interval, prefix in VARIANTS:
        config = workload(base, name, interval, prefix)
        evaluate_config(config, name)
        print(f'REVERSE RESULT {name}: {json.dumps(report()[name])}', flush=True)


if __name__ == '__main__':
    main()
