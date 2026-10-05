"""Fast cached1024 historical independent BP readouts; no inference/retracing."""
import json
from pathlib import Path

from reviewed_bp_independent import REPO, independent_args, initial_states, readout_metrics
from lasagna.inference_provenance import atomic_write, sha256_file
from vesuvius.neural_tracing.fiber_trace_3d.crop_benchmark import run

ROOT = Path('/ephemeral/hendrik/fiber_crop_bench')
OUT = ROOT/'20261002_fast_historical_independent_v1'
CASES = [
    ('u22000 original BP', '20260923_v1/bp-beam-new', False),
    ('u22000 hard perpendicular', '20260923_v1/bp-beam-new1024-independent-reference', True),
    ('u22000 hard both signs', '20260923_v1/bp-beam-new-perpsign-p1hardsign', False),
    ('Qual supplied fibers', '20260923_qual_v1/qual-bp', False),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, relative, reuse in CASES:
        prior = ROOT/relative
        before = json.loads(Path(str(prior)+'.run.json').read_text())
        for path, expected in {**before['inputs'], **before['references']}.items():
            if sha256_file(path) != expected:
                raise ValueError(f'historical input changed: {path}')
        args = independent_args(before['command'][1:])
        trace = Path(args[1])
        trace_hashes = {str(p.relative_to(trace)): sha256_file(p)
                        for p in sorted(trace.rglob('*')) if p.is_file()}
        if not trace_hashes:
            raise ValueError(f'empty trace cache: {trace}')
        name = prior.name+'-independent'
        target = prior if reuse else OUT/name
        if not reuse:
            args[args.index('--output')+1] = str(target)
            config = dict(output_root=str(OUT),
                reference_dir=args[args.index('--reference-fiber-dir')+1],
                input_files=list(before['inputs']),
                commands=[dict(name=name, binary='vc_fiber_trace_chunk', args=args)])
            config_path = OUT/f'{name}.config.json'
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise ValueError('saved configuration changed')
            atomic_write(config_path, config, durable=False)
            record_path = Path(str(target)+'.run.json')
            if not (record_path.exists() and json.loads(record_path.read_text())['returncode'] == 0):
                run(config)
        after = json.loads(Path(str(target)+'.run.json').read_text())
        if before['references'] != after['references'] or before['inputs'] != after['inputs']:
            raise ValueError('reference/input inventory changed')
        if trace_hashes != {str(p.relative_to(trace)): sha256_file(p)
                           for p in sorted(trace.rglob('*')) if p.is_file()}:
            raise ValueError('trace changed during evaluation')
        old = initial_states(Path(str(prior)+'_consistency.csv'))
        new = initial_states(Path(str(target)+'_consistency.csv'))
        metrics = readout_metrics(Path(str(target)+'.log').read_text())
        metrics.update(reused_saved_readout=reuse, wall_seconds=0 if reuse else after['wall_seconds'],
            historical_wall_seconds=before['wall_seconds'], initial_states_identical=old==new,
            initial_pieces=len(new), historical_pieces=len(old),
            changed_state_rows=sum(a!=b for a,b in zip(old,new))+abs(len(old)-len(new)),
            trace_hashes=trace_hashes, historical_record=str(prior)+'.run.json',
            result_record=str(target)+'.run.json')
        results[label] = metrics
        atomic_write(OUT/'results.json', results, durable=False)
        print('RESULT '+label+' '+json.dumps({k:v for k,v in metrics.items()
              if k not in ('ladder','trace_hashes')}), flush=True)
    report = REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-02-fast-historical-independent'
    atomic_write(report.with_suffix('.json'), results, durable=False)
    lines = ['# Fast historical1024 independent BP metrics', '',
        'Only u22000 beam8 and Qual. Cached traces; no inference, retracing or oracle conditioning. '
        'Historical command/objective flags retained; only pruning/oracle flags removed and independent readout enabled. '
        'Current native binary,128CPU threads, RelWithDebInfo, one run per new result. '
        'Historical hard-perpendicular independent log reused without a solve. Input/reference hashes checked; '
        'trace payload hashes recorded and unchanged during each run. Initial state equality is audited below.', '',
        '| Variant | Non-oracle accuracy | Correct/26 | Missing | Observed shift score | Shifts / observed refs | Tied / unbounded | Initial states identical | New CPU wall s |',
        '|---|---:|---:|---:|---:|---:|---:|---|---:|']
    for label,r in results.items():
        s = r['observed_ladder_shift']
        lines.append(f"| {label} | {r['independent_percent']:.4f}% | {r['correct_references']}/26 | "
            f"{r['independent_missing']} | {s['ladder_shift_percent']:.4f}% | {s['ladder_shifts']} / {s['references']} | "
            f"{r['tied_rows']} / {r['unbounded_rows']} | {r['initial_states_identical']} | {r['wall_seconds']:.2f} |")
    lines += ['', 'Accuracy uses all26 references. Shift score100*(1-K/Nobserved) omits missing references and bridges their gaps; '
        'full26shift is unavailable for incomplete ladders. Global sign/offset alignment uses GT for scoring only. '
        'Tied ranks, if any, are diagnostics, not uniquely determined assignments.', '',
        'Any row with initial states identical=False is a current-binary rerun of the historical objective, '
        'not an exact reconstruction of the old solve. Historical oracle/constraint scores are not overwritten. '
        'The adjacent JSON reports changed_state_rows and both source records.', '', '## Ladders', '',
        '| Reference | '+' | '.join(results)+' |', '|---|'+'---:|'*len(results)]
    ladders = {k:{float(r['reference']):r['winding'] for r in v['ladder']} for k,v in results.items()}
    for i in range(26):
        ref=i/2
        lines.append(f'| {ref:g} | '+' | '.join(str(ladders[k].get(ref,'missing')) for k in results)+' |')
    lines += ['', 'Run: control action `historical_bp_independent`, then '
        '`bash /home/hendrik/villa3/scripts/run_training_failure_mining`.', f'Artifacts: `{OUT}`.']
    report.with_suffix('.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
