"""CPU-only independent readout of the two frozen raw40500 crop traces."""
import csv
import io
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO/'vesuvius/src'), str(REPO)]
from lasagna.inference_provenance import atomic_write
from vesuvius.neural_tracing.fiber_trace_3d.crop_benchmark import run
from vesuvius.neural_tracing.fiber_trace_3d.density_penalty_sweep import independent_metrics, independent_summary_rows
from vesuvius.neural_tracing.fiber_trace_3d.local_cost_sweep import ladder_shift_metrics

OUT = Path('/ephemeral/hendrik/fiber_crop_bench/20261002_raw40500_prefix512_v1')


def independent_args(arguments):
    args = list(arguments)
    for flag, arity in (('--reference-prune-offenders',0), ('--reference-prune-policy',1),
                        ('--reference-oracle-magnitude-weight',1),
                        ('--reference-oracle-accept-message-limit',0), ('--reference-oracle-rounds',1)):
        if flag in args:
            index = args.index(flag)
            if arity and (index+arity >= len(args) or args[index+1].startswith('--')):
                raise ValueError(f'missing value for {flag}')
            del args[index:index+arity+1]
    if '--reference-independent' not in args:
        args.append('--reference-independent')
    return args


def initial_states(path):
    keys = ('piece','original_trace','source_piece','begin_arc_base','end_arc_base',
            'winding_valid','winding_relative_map','winding_final_class')
    with path.open() as f:
        return [tuple(row[k] for k in keys) for row in csv.DictReader(f)]


def readout_metrics(log):
    result = independent_metrics(log)
    summaries = independent_summary_rows(log)
    if len(summaries) != 26 or {float(r['reference']) for r in summaries} != {i/2 for i in range(26)}:
        raise ValueError('expected26 reference rows')
    result.update(references=26,
        correct_references=sum(float(r['reference']) == float(r['winding']) for r in result['ladder']),
        missing_references=[r['reference'] for r in summaries if r['evaluated'] == '0'],
        tied_rows=sum(int(r['best_ties']) > 1 for r in summaries),
        unbounded_rows=sum(r['lower_unbounded_best']=='1' or r['upper_unbounded_best']=='1' for r in summaries))
    observed = ladder_shift_metrics(result['ladder'])
    result['observed_ladder_shift'] = dict(**observed, references=len(result['ladder']),
        note='missing references omitted; bridges across gaps, not the full26-reference metric')
    result.update(observed if not result['independent_missing'] else dict(
        ladder_shifts=None,ladder_shift_percent=None, shift_unavailable='incomplete reference ladder'))
    return result


def main():
    results = {}
    for policy in ('unfiltered', 'prefix512'):
        config = json.loads((OUT/f'bp-{policy}.config.json').read_text())
        config.pop('publish_bp_plots', None)
        case = config['commands'][0]
        args = case['args'] = independent_args(case['args'])
        name = f'bp-{policy}-independent'
        case['name'] = name
        args[args.index('--output')+1] = str(OUT/name)
        path = OUT/f'{name}.config.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('independent configuration changed')
        atomic_write(path,config,durable=False)
        record = OUT/f'{name}.run.json'
        if not (record.exists() and json.loads(record.read_text())['returncode'] == 0):
            run(config)
        before = json.loads((OUT/f'bp-{policy}.run.json').read_text())
        after = json.loads(record.read_text())
        if before['references'] != after['references'] or before['inputs'] != after['inputs']:
            raise ValueError('reference/input hashes changed')
        old = initial_states(OUT/f'bp-{policy}_consistency.csv')
        new = initial_states(OUT/f'{name}_consistency.csv')
        if old != new:
            raise ValueError(f'{policy}: initial BP states differ from prior solve')
        log = (OUT/f'{name}.log').read_text()
        result = readout_metrics(log)
        result.update(initial_states_identical=True, initial_pieces=len(new), references=26,
            wall_seconds=after['wall_seconds'])
        results[policy] = result
        atomic_write(OUT/'independent_results.json',results,durable=False)
        print(policy+': '+json.dumps({k:v for k,v in result.items() if k!='ladder'}),flush=True)
    report_path = REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-02-prefix512-independent.json'
    atomic_write(report_path,results,durable=False)
    lines = ['# Independent BP accuracy and shift — raw40500 1024³\n',
        'Same frozen source traces, normal/reference hashes and hard-both-sign BP objective. No reference nodes/conditioning/pruning in the solve. Independent reference ranking follows the frozen solve; global gauge alignment uses GT for scoring only. Initial piece states were checked exactly against the previous run. No inference or retracing.\n',
        '| Variant | Exact /26 | Accuracy | Missing | Observed-only shifts | Observed-only shift score | CPU seconds |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for policy,r in results.items():
        s=r['observed_ladder_shift']
        lines.append(f"| {policy} | {r['correct_references']}/26 | {r['independent_percent']:.4f}% | {r['independent_missing']} | {s['ladder_shifts']} / {s['references']} | {s['ladder_shift_percent']:.4f}% | {r['wall_seconds']:.2f} |")
    lines += ['', 'Full26-reference shift score is unavailable if references lack admitted evidence. Observed-only diagnostic sorts the remaining references, skips missing ones, and uses100×(1−K/Nobserved); it must not be compared as the full26score. Accuracy always retains denominator26. Ties/unbounded candidates and full ladders are saved in the adjacentJSON.',
              '', '## Ladders', '', '| Reference | Unfiltered | Prefix512 |', '|---:|---:|---:|']
    ladders = {p:{float(r['reference']):r['winding'] for r in v['ladder']} for p,v in results.items()}
    for i in range(26):
        ref=i/2
        lines.append(f"| {ref:g} | {ladders['unfiltered'].get(ref,'missing')} | {ladders['prefix512'].get(ref,'missing')} |")
    lines += ['', 'Approved command: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, action `reviewed_bp_independent`. Exact configs, raw logs, provenance and timings are beside the previous crop outputs.']
    report_path.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
