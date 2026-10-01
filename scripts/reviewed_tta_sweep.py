"""Two-product thirteen-forward TTA, compared to ordinary float32 inference."""
import copy
import json
import time
from pathlib import Path
from reviewed_fiber_evaluation import ROOT, REPO, execute_evaluation
from lasagna.inference_provenance import atomic_write
from vesuvius.neural_tracing.fiber_trace_3d.tta import VARIANTS

EXPERIMENT = ROOT/'tta13_u31500_v2'
RUNS = ('baseline', *VARIANTS)


def report():
    from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import complete_report
    results = {}
    for name, folder in [('previous_bf16', ROOT/'halfsigma_current_raw_v3'),
                         *((v, EXPERIMENT/v) for v in RUNS)]:
        path = folder/'full/summary.json'
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        if not summary.get('completed'):
            continue
        assert summary['cases'] == 182 and summary['eligible_runs'] == 91
        results[name] = {'summary_path': str(path), **complete_report(summary)}
    atomic_write(EXPERIMENT/'comparison.json', results, durable=False)
    target = REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks/reviewed_tta13_20261001'
    atomic_write(target.with_suffix('.json'), results, durable=False)
    lines = ['# Reviewed 13-transform TTA', '',
             'Frozen u31500, reviewed91fibers/182directions, GPUs0–5. Same beam8 tracing/thresholds as previous evaluation. '
             'Mean and median share13forwards per tile and exactly14accumulated channels. Baseline is a separate ordinary7channel run. Strict float32 (TF32 disabled), batch1, float32 accumulation. '
             'Previous bf16 row is historical context, not the matched baseline.', '',
             'Directions reframed analytically in sign-invariant double-angle coordinates; mean or componentwise midpoint median '
             'of cos2/sin2, capped to unit disk then re-encoded. Presence ordinary mean/median. Median is not a spherical geometric median. '
             'Tile aggregates are then blended by the existing accumulator. No embeddings; pilot limited to output stride1 and one branch.', '',
             'Mean is combined directed reference length/(restarts+1); quantiles include terminal segments. '
             'This reviewed hard-region cohort is not proven to be an independent training holdout.', '',
             '| Variant | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |',
             '|---|---:|---:|---:|---:|---:|---:|']
    def row(name, stats):
        return f'| {name} | {stats["failures"]} | {stats["failures"]/stats["directed_reference_mm"]:.5f} | ' + ' | '.join(
            f'{stats[k]:.3f}' for k in ('mean_segment_length_mm','median_segment_length_mm','p05_segment_length_mm','p95_segment_length_mm')) + ' |'
    for name, result in results.items():
        lines.append(row(name, result['global_result']))
    lines += ['', '## Per scroll', '', '| Scroll / variant | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for scroll in results['previous_bf16']['scrolls']:
        for name, result in results.items():
            lines.append(row(f'{scroll} / {name}', result['scrolls'][scroll]))
    lines += ['', '## Reproduction', '',
              'Set scripts/training_failure_mining_control.json action to reviewed_tta_sweep, then run:', '',
              '```bash', 'bash /home/hendrik/villa3/scripts/run_training_failure_mining', '```', '',
              f'Outputs/configs: {EXPERIMENT}. Missing rows are not yet completed. Mean fills the shared two-product cache; median reuses it and only infers missing coverage. Their wall times are not independent TTA inference benchmarks.']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    return results


def main():
    base = json.loads((ROOT/'halfsigma_current_raw_v3/workload.json').read_text())
    for name in RUNS:
        config = copy.deepcopy(base)
        config['output_root'] = str(EXPERIMENT/name)
        evaluation = config['evaluation']
        evaluation['inference_cache_namespace'] = ('reviewed_u31500_fp32_baseline_v2' if name == 'baseline'
                                                 else 'reviewed_u31500_tta13_v2')
        evaluation['gpu_manager']['label'] = f'reviewed-tta-{name}'
        settings = evaluation['inference']
        # Two products:14fp32+weight (60bytes/vx); baseline7fp32+weight(32).
        # Preserve <= original504GiB hot payload and <= original spill segment size.
        record_bytes = 32 if name == 'baseline' else 60
        settings['accumulator_hot_slots'] = settings['accumulator_hot_slots'] * 18 // record_bytes
        settings['accumulator_spill_segment_slots'] = settings['accumulator_spill_segment_slots'] * 18 // record_bytes
        settings.update(precision='off', allow_tf32=False, accumulator_dtype='float32',
                        max_batch_size=1, profile_pipeline=True)
        if name != 'baseline':
            settings['tta'] = name
        # Original shared-file result ring and copy path; all workers/queues retained.
        output = Path(config['output_root'])
        output.mkdir(parents=True, exist_ok=True)
        path = output/'workload.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('immutable TTA workload changed')
        atomic_write(path, config, durable=False)
        full = output/'full'
        summary_path = full/'summary.json'
        if full.exists() and not (summary_path.exists() and json.loads(summary_path.read_text()).get('completed')):
            archived = output/f'interrupted_full_{time.time_ns()}'
            full.rename(archived)
            print(f'Preserved incomplete trace output: {archived}; inference cache retained', flush=True)
        execute_evaluation(path, config, name)
        summary = report()[name]
        print(f'TTA RESULT {name}: {json.dumps(summary)}', flush=True)


if __name__ == '__main__':
    main()
