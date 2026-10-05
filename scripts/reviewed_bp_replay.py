"""Replay the historical crop using the two frozen raw40500 policies."""
import json
import math

from reviewed_bp_prefix import CONFIGS, OUT, REPO, save_config
from lasagna.inference_provenance import atomic_write
from vesuvius.neural_tracing.fiber_trace_3d.crop_benchmark import run


def make_config(policy):
    if policy not in ('unfiltered', 'prefix512'):
        raise ValueError(policy)
    template = json.loads((CONFIGS/'crop_benchmark_beam_20260923.json').read_text())
    pilot = json.loads((OUT/'beam-unfiltered.config.json').read_text())
    pilot_args = pilot['commands'][0]['args']
    args = list(template['commands'][0]['args'])
    for flag in ('--fiber-manifest', '--normal-manifest'):
        args[args.index(flag)+1] = pilot_args[pilot_args.index(flag)+1]
    name = f'replay-{policy}'
    args[args.index('--output')+1] = str(OUT/f'{name}.json')
    # Match the BP pilot's actual trace resolution and production tuning.
    for flag in ('--inference-scaledown-power', '--beam-width', '--beam-lookahead-steps',
                 '--step-voxels', '--smoothness-normal-weight', '--smoothness-tangent-weight',
                 '--embedding-weight'):
        args += [flag, pilot_args[pilot_args.index(flag)+1]]
    if policy == 'prefix512':
        args += ['--replay-reverse-interval-base', '512', '--replay-reverse-to-start',
                 '--replay-reverse-all-endpoints']
    return dict(output_root=str(OUT), reference_dir=template['reference_dir'],
        input_files=pilot['input_files'], measurement_date='2026-10-02',
        commands=[dict(name=name, binary='vc_fiber_trace_chunk', args=args)])


def cohort(document):
    return [(c['source_index'], c['run_index'], c['direction'], c['reference_length_base_voxels'])
            for c in document['cases']]


def main():
    historical = json.loads((OUT.parent/'20260923_v1/beam-7ff0ce6c.json').read_text())
    rows = []
    publications = []
    for policy in ('unfiltered', 'prefix512'):
        config = make_config(policy)
        save_config(OUT/f'replay-{policy}.config.json', config)
        run(config)
        record = json.loads((OUT/f'replay-{policy}.run.json').read_text())
        previous = json.loads((OUT/'bp-unfiltered.run.json').read_text())
        if record['references'] != previous['references']:
            raise ValueError('reference hashes changed since BP pilot')
        path = OUT/f'replay-{policy}.json'
        result = json.loads(path.read_text())
        if cohort(result) != cohort(historical):
            raise ValueError('historical crop cohort changed')
        if not all(c['evaluation_complete'] for c in result['cases']):
            raise ValueError('incomplete replay')
        summary = result['summary']
        failures = summary['total_failures']
        score = 100 / (failures + 1)
        if not math.isclose(summary['mean_segment_length_percent'], score):
            raise ValueError('replay metric changed')
        rows.append(dict(policy=policy, replay_percent=score, failures=failures,
            filtered_stops=sum(len(c['filtered_stops']) for c in result['cases']),
            reverse_checks=sum(c['reverse_checks'] for c in result['cases']),
            retained_percent=100*sum(c['traced_length_base_voxels'] for c in result['cases']) /
                sum(c['reference_length_base_voxels'] for c in result['cases']),
            cases=len(result['cases']), summary=summary,
            wall_seconds=json.loads((OUT/f'replay-{policy}.run.json').read_text())['wall_seconds'],
            result=str(path)))
        print('REPLAY RESULT '+json.dumps(rows[-1]), flush=True)
        publish = dict(config, commands=[], publish_replay_plots=[dict(
            id=f'raw40500-1024-{policy}', label=f'Raw40500 beam8 {policy} (1024)',
            result=path.name, run_record='docs/fiber_benchmark_runs/2026-10-02-prefix512-replay.md')])
        publications.append(publish)
    target = REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-02-prefix512-replay'
    atomic_write(target.with_suffix('.json'), rows, durable=False)
    atomic_write(OUT/'replay_comparison.json', rows, durable=False)
    lines = ['# Raw40500 historical1024 crop replay', '',
        'Same frozen raw40500 manifest and tuning as the BP pilot, 26 tagged references clipped to the historical '
        '1024³ crop. Exact replay case identities and lengths validated against the September23 beam replay. '
        'Cached predictions; no new inference or native changes.128 requested CPU threads, RelWithDebInfo, one run each.', '',
        'Score is100/(ordinary failures+1). Filtered stops are separate, not ordinary errors. '
        'Full-prefix512/no-history uses the shared replay filter, including terminal endpoints; GT controls '
        'evaluation termination and reseeding. This is not a GT-free deployment score. No short-piece filtering.', '',
        '| Policy | Replay % ↑ | Ordinary failures | Filtered stops | Retained % | Reverse checks | Wall s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['policy']} | {row['replay_percent']:.4f}% | {row['failures']} | "
            f"{row['filtered_stops']} | {row['retained_percent']:.4f}% | {row['reverse_checks']} | {row['wall_seconds']:.2f} |")
    lines += ['', 'Approved runner: set control action `reviewed_bp_replay`, then '
        '`bash /home/hendrik/villa3/scripts/run_training_failure_mining`. Configs, commands, hashes, logs and outputs: '
        f'`{OUT}`. See adjacent JSON for complete native summaries.']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    for publish in publications:
        run(publish)


if __name__ == '__main__':
    main()
