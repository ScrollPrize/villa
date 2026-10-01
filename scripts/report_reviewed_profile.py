"""Read-only analysis of the reviewed chunk-size sweep; emits JSON to stdout."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from reviewed_fiber_evaluation import ROOT


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {} if not len(a) else dict(n=len(a), sum=float(a.sum()), mean=float(a.mean()),
                                    p50=float(np.quantile(a, .5)), p95=float(np.quantile(a, .95)))


def compare_traces(full, baseline):
    from scipy.spatial import cKDTree
    changed_counts = []
    endpoint_distances = []
    path_means = []
    for path in sorted((full/'cases').glob('*/trace.json')):
        a = json.loads(path.read_text())
        b = json.loads((baseline/'cases'/path.parent.name/'trace.json').read_text())
        for key in ('fiber_path', 'reverse', 'source_id'):
            assert a['provenance'][key] == b['provenance'][key]
        assert a['reference_points_base_xyz'] == b['reference_points_base_xyz']
        pa = np.concatenate([s['trace_points_base_xyz'] for s in a['replay']['segments']])
        pb = np.concatenate([s['trace_points_base_xyz'] for s in b['replay']['segments']])
        # Symmetric nearest sampled point distance: geometry diagnostic, not the acceptance metric.
        path_means.append(float(np.mean(np.r_[cKDTree(pa).query(pb)[0], cKDTree(pb).query(pa)[0]])))
        endpoint_distances.append(float(np.linalg.norm(pa[-1]-pb[-1])))
    def counts(folder):
        s = json.loads((folder/'summary.json').read_text())
        return {(v['source_id'], c['source_index'], c['direction'], c.get('run_index',0)): c['restart_count']
                for v in s['sources'] for c in v['cases']}
    ac, bc = counts(full), counts(baseline)
    assert ac.keys() == bc.keys()
    changed_counts = [dict(key=k, baseline=bc[k], candidate=ac[k]) for k in ac if ac[k]!=bc[k]]
    return dict(changed_restart_cases=changed_counts,
                per_case_mean_symmetric_sample_distance_base=distribution(path_means),
                terminal_point_distance_base=distribution(endpoint_distances))


def analyze(output):
    full = output / 'full'
    summary = json.loads((full / 'summary.json').read_text())
    inference_path = full / 'inference_summary.json'
    inference = json.loads(inference_path.read_text())
    profile = json.loads((full / 'pipeline_profile.json').read_text())
    if profile.get('scope', 'batch') != 'batch' or not profile['worker_samples']:
        raise ValueError('Report requires a fresh per-evaluation nonempty profile, not a cumulative session')
    queues = [json.loads(line) for line in (full / 'inference_queues.jsonl').read_text().splitlines()]
    first_submit = next(r['elapsed_seconds'] for r in queues if r['gpu_submitted_uncollected'])
    end = inference_path.stat().st_mtime
    start = end - inference['wall_seconds']
    gpu = []
    with (output / 'gpu_utilization.csv').open() as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            t = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
            if start + first_submit <= t <= end:
                gpu.append(dict(index=int(row['index']), utilization=float(row['utilization.gpu [%]']), time=t))
    workers = profile['worker_samples']
    stages = profile['stage_samples']
    pool = inference['worker_pool']
    monitor = json.loads((output/'monitor_window.json').read_text())
    return dict(output=str(output), wall_seconds=inference['wall_seconds'],
        command_seconds=monitor['end_unix']-monitor['start_unix'],
        pipeline_complete_seconds=queues[-1]['elapsed_seconds'],
        profile_serialization_seconds=inference['wall_seconds']-queues[-1]['elapsed_seconds'],
        first_gpu_submit_seconds=first_submit, post_submit_seconds=inference['wall_seconds']-first_submit,
        gpu_window_note='Approximate coordinator interval anchored to inference_summary mtime; exclude pre-grant samples via first GPU submission. Includes model initialization/tail. GPU utilization is sampled kernel-active time, not FLOP utilization.',
        gpu_utilization=distribution([r['utilization'] for r in gpu]),
        per_gpu_utilization={i: distribution([r['utilization'] for r in gpu if r['index']==i]) for i in range(6)},
        tiles=pool['tiles'], batch_sizes=pool['batch_size_counts'],
        published_chunks=sum(s['published_chunks'] for s in inference['sources'].values()),
        coordinator_seconds={k:v for k,v in pool.items() if k.startswith('pump_')},
        worker_seconds=pool['worker_seconds'],
        worker_stages={k: distribution([s[k] for s in workers if k in s]) for k in workers[0] if k not in ('worker_index', 'batch_size')},
        cpu_stages={k: {field: distribution([s[field] for s in stages if s['stage']==k])
                       for field in ('wall_seconds','thread_cpu_seconds','off_cpu_seconds_estimate')}
                    for k in sorted({s['stage'] for s in stages})},
        accumulator_seconds=distribution(profile['accumulator_seconds']),
        accumulator_queue_plus_ack_seconds=distribution(profile['accumulator_queue_plus_ack_seconds']),
        final_queues=queues[-1],
        restarts=summary['failures'], evaluated_cases=summary['evaluated_cases'],
        mean_mm=summary['mean_segment_length_mm'],
        versus_original_u31500=compare_traces(full, ROOT/'halfsigma_current_raw_v3/full'),
        source_restarts={s['source_id']: sum(c['restart_count'] for c in s['cases']) for s in summary['sources']})


if __name__ == '__main__':
    root = ROOT / 'profile_chunks_u31500_v1'
    print(json.dumps([analyze(p) for p in sorted(root.glob('chunk*'))
                      if (p/'full/pipeline_profile.json').exists()], indent=2))
