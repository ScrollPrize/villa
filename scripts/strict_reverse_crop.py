"""Matched raw40500 reverse-radius sweep, reusing the crop benchmark machinery."""
import copy
import json
import subprocess
from pathlib import Path

from reviewed_bp_prefix import OUT as BASE, REPO, save_config
from reviewed_bp_replay import cohort, make_config
from reviewed_bp_independent import independent_args, initial_states, readout_metrics
from lasagna.inference_provenance import atomic_write, sha256_file
from vesuvius.neural_tracing.fiber_trace_3d.crop_benchmark import run, bp_metrics

OUT = BASE.parent/'20261002_strict_reverse_crop_v1'
VARIANTS = [('ellipse10x20',10,20), ('ellipse5x10',5,10),
            ('isotropic20',20,20), ('isotropic10',10,10), ('isotropic5',5,5)]
REPORT = 'docs/fiber_benchmark_runs/2026-10-02-strict-reverse-crop.md'


def status():
    output = subprocess.check_output(['ps','-eo','pid=,etime=,pcpu=,pmem=,args='],text=True)
    for line in output.splitlines():
        if str(OUT) in line or 'scripts/strict_reverse_crop.py' in line:
            print(line,flush=True)


def configured(template, name, suffix, updates, extra=()):
    config = copy.deepcopy(template)
    config.pop('publish_bp_plots', None)
    config.pop('publish_replay_plots', None)
    config['output_root'] = str(OUT)
    case = config['commands'][0]
    case['name'] = name
    args = case['args']
    for flag, value in dict(updates, **{'--output':str(OUT/(name+suffix))}).items():
        if flag in args:
            args[args.index(flag)+1] = str(value)
        else:
            args += [flag,str(value)]
    args += list(extra)
    return config


def execute(config):
    name = config['commands'][0]['name']
    save_config(OUT/f'{name}.config.json', config)
    record = OUT/f'{name}.run.json'
    binary = REPO/'volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_chunk'
    baseline_record = json.loads((BASE/'bp-prefix512.run.json').read_text())
    expected_refs = baseline_record['references']
    current_refs = {str(p):sha256_file(p) for p in sorted(Path(config['reference_dir']).glob('*.json'))}
    if current_refs != expected_refs:
        raise ValueError('reference inventory changed')
    for path,digest in baseline_record['inputs'].items():
        if sha256_file(path) != digest:
            raise ValueError('baseline input manifest changed')
    for path, digest in expected_refs.items():
        if sha256_file(path) != digest:
            raise ValueError('reference changed')
    reuse = False
    if record.exists():
        old = json.loads(record.read_text())
        libs = {str(p):sha256_file(p) for p in sorted((binary.parent.parent/'core').glob('libvc_fiber_tracer.*')) if p.is_file()}
        reuse = (old['returncode']==0 and old['command']==[str(binary),*config['commands'][0]['args']]
                 and old['binary_sha256']==sha256_file(binary) and old['shared_libraries']==libs
                 and old['references']==expected_refs
                 and old['inputs']=={p:sha256_file(p) for p in config['input_files']})
    if not reuse:
        artifact = Path(config['commands'][0]['args'][config['commands'][0]['args'].index('--output')+1])
        if artifact.suffix == '.zarr' and artifact.exists():
            raise ValueError('stale dense artifact: use a new output namespace; refusing overwrite')
        run(config)
    result = json.loads(record.read_text())
    if result['references'] != expected_refs:
        raise ValueError('reference inventory changed during execution')
    for path,digest in result['inputs'].items():
        if sha256_file(path) != digest:
            raise ValueError('input changed during execution')
    return result


def detailed_result(root, name, timings):
    result = json.loads((root/f'replay-{name}.json').read_text())
    artifact = json.loads((root/f'beam-{name}.zarr/.zattrs').read_text())['processing']['artifact']
    summary = artifact['trace_summary']
    metrics = bp_metrics((root/f'bp-{name}.log').read_text())
    metrics['oracle_missing_references'] = metrics.pop('missing_references')
    independent_log = root/(f'bp-{name}-independent.log' if root == BASE else f'independent-{name}.log')
    metrics.update(readout_metrics(independent_log.read_text()))
    retained = summary['retained_length_base']
    discarded = summary['reverse_skipped_length_base']
    metrics.update(timings=timings, failures=result['summary']['total_failures'],
        replay_percent=100/(result['summary']['total_failures']+1),
        filtered_stops=sum(len(c['filtered_stops']) for c in result['cases']),
        replay_reverse_checks=sum(c['reverse_checks'] for c in result['cases']),
        retained_percent=result['summary']['length_weighted_success_percent'],
        reference_length_base=result['summary']['directed_reference_length_base_voxels'],
        reference_credited_base=result['summary']['traced_length_base_voxels'],
        trace_count=artifact['trace_count'], dense_retained_base=retained,dense_discarded_base=discarded,
        dense_retained_percent=100*retained/(retained+discarded) if retained+discarded else 0,
        dense_reverse_attempts=summary['reverse_attempts'],dense_reverse_failures=summary['reverse_failures'])
    return metrics


def report(rows):
    # Re-read authoritative artifacts so report-only refresh needs no solve.
    rows = {name:{**r,**detailed_result(OUT,name,r['timings'])} for name,r in rows.items()}
    baseline_times = {stage:json.loads((BASE/f'{prefix}.run.json').read_text())['wall_seconds']
        for stage,prefix in [('trace','beam-prefix512'),('replay','replay-prefix512'),
            ('bp','bp-prefix512'),('independent','bp-prefix512-independent')]}
    baseline = detailed_result(BASE,'prefix512',baseline_times)
    baseline.update(normal_radius_base=20,tangential_radius_base=40,initial_states_identical=True)
    rows = {'baseline20x40':baseline,**rows}
    atomic_write(OUT/'results.json', rows, durable=False)
    target = REPO/'volume-cartographer'/REPORT
    atomic_write(target.with_suffix('.json'),rows,durable=False)
    lines = ['# Strict reverse-consistency tolerance — raw40500,1024³ crop', '',
        'Fixed full-prefix512/no-history, beam8, cache/model/GT cohort and hard-both BP objective. '
        'Only reverse-check radii change (normal/in-sheet, base voxels). Equal radii use the same metric '
        'with isotropic Euclidean tolerance; correspondence/progress rules remain unchanged. '
        'GT acceptance and reseeding unchanged. No short-piece filtering. Cached CPU work,128threads, RelWithDebInfo.', '',
        '| Variant | Replay ↑ | BP constraint ↑ | Oracle accuracy ↑ | Non-oracle accuracy ↑ | Observed shift ↑ |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        lines.append(f"| {name} | {r['replay_percent']:.4f}% | {r['constraint_score_percent']:.4f}% | "
            f"{r['reference_score_percent']:.4f}% | {r['independent_percent']:.4f}% | "
            f"{r['observed_ladder_shift']['ladder_shift_percent']:.4f}% |")
    lines += ['', '| Variant | Ordinary errors | Filtered stops | Reference retained % | Dense candidate retained % | Traces | Missing / ties / unbounded | Shifts / observed | Trace+replay+BP+independent s |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        s=r['observed_ladder_shift']
        lines.append(f"| {name} | {r['failures']} | {r['filtered_stops']} | {r['retained_percent']:.4f}% | "
            f"{r['dense_retained_percent']:.4f}% | {r['trace_count']} | {r['independent_missing']} / {r['tied_rows']} / {r['unbounded_rows']} | "
            f"{s['ladder_shifts']} / {s['references']} | {sum(r['timings'].values()):.2f} |")
    lines += ['', 'Replay100/(ordinary errors+1) excludes filtered stops. Reference coverage is accepted reference arc; '
        'dense retained fraction uses that variant’s own candidates. Both may change nonmonotonically when filtering changes '
        'restart/coverage decisions. Oracle accuracy is GT-assisted. Independent accuracy uses26references; '
        'observed shift omits missing references and bridges gaps. Missing/tied/unbounded readouts and ladders are in JSON. '
        'Full26shift unavailable for missing evidence. Independent initial states must match that variant’s oracle-run initial solve.', '',
        'Constraint score is −100×problematic/retained-fulfilled (closer to zero is better), not a removal percentage. '
        'Accuracy counts exact assignments among all26 references; shift is100×(1−offset changes/observed references). '
        'Non-oracle scores use the deterministic rank1 readout. Isotropic5 has an ambiguous reference8 '
        '(four equal best candidates in the evaluated range and an upper-unbounded best region); its accuracy/shift '
        'must not be interpreted as a uniquely determined ladder. Other new variants have no tied/unbounded readouts.', '',
        'Interpretation: isotropic20 is the strongest overall candidate here: substantially better constraint and shift '
        'scores with near-baseline reference retention and the same six filtered stops. Half-ellipse10/20 has the '
        'highest non-oracle accuracy among the new variants but a worse constraint score than baseline. '
        'Isotropic5 over-filters severely. These are single-run crop results, not a rerun of the full91-piece '
        'reviewed benchmark; no production default has changed.', '',
        f'Artifacts: `{OUT}`. Approved command: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, '
        'control action `strict_reverse_crop`. One run per variant, prior baseline reused after replay validation.']
    target.write_text('\n'.join(lines)+'\n')
    historical_path = target.parent/'2026-10-02-prefix512.md'
    historical = historical_path.read_text().splitlines()
    historical = [line for line in historical if not line.startswith('| raw40500 strict ')]
    anchor = next(i for i,line in enumerate(historical)
        if line.startswith('| raw40500 beam8, full-prefix512 filter |'))
    additions = []
    for name,r in rows.items():
        if name == 'baseline20x40':
            continue
        additions.append(f"| raw40500 strict {name}, full-prefix512 | {r['replay_percent']:.2f}% | "
            f"{r['constraint_score_percent']:.2f}% | {r['reference_score_percent']:.2f}% | "
            f"{r['independent_percent']:.2f}% | {r['observed_ladder_shift']['ladder_shift_percent']:.2f}% |")
    historical[anchor+1:anchor+1] = additions
    link = 'Strict-radius extension: [matched matrix, filter stops, coverage and timings](2026-10-02-strict-reverse-crop.md).'
    if link not in historical:
        historical += ['', link]
    historical_path.write_text('\n'.join(historical)+'\n')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    historical = json.loads((BASE/'replay-prefix512.json').read_text())
    replay_template = make_config('prefix512')
    control = configured(replay_template,'replay-baseline','.json',{})
    execute(control)
    current = json.loads((OUT/'replay-baseline.json').read_text())
    if current['cases'] != historical['cases'] or current['summary'] != historical['summary']:
        raise ValueError('default baseline changed; do not mix historical baseline')
    explicit = configured(replay_template,'replay-explicit-baseline','.json',
        {'--replay-reverse-normal-base':20,'--replay-reverse-tangential-base':40})
    execute(explicit)
    if json.loads((OUT/'replay-explicit-baseline.json').read_text())['cases'] != current['cases']:
        raise ValueError('explicit baseline radii changed default replay')
    trace_template = json.loads((BASE/'beam-prefix512.config.json').read_text())
    bp_template = json.loads((BASE/'bp-prefix512.config.json').read_text())
    rows = {}
    for name,n,t in VARIANTS:
        timings = {}
        config = configured(replay_template,'replay-'+name,'.json',
            {'--replay-reverse-normal-base':n,'--replay-reverse-tangential-base':t})
        timings['replay'] = execute(config)['wall_seconds']
        result = json.loads((OUT/f'replay-{name}.json').read_text())
        if cohort(result)!=cohort(historical) or not all(c['evaluation_complete'] for c in result['cases']):
            raise ValueError('replay cohort incomplete/changed')
        trace = OUT/f'beam-{name}.zarr'
        config = configured(trace_template,'beam-'+name,'.zarr',{'--reverse-normal':n,'--reverse-tangential':t})
        timings['trace'] = execute(config)['wall_seconds']
        attrs = json.loads((trace/'.zattrs').read_text())['processing']['artifact']
        summary = attrs['trace_summary']
        config = configured(bp_template,'bp-'+name,'',{})
        config['commands'][0]['args'][1] = str(trace)
        config['input_files'] += [str(trace/'.zattrs')]
        timings['bp'] = execute(config)['wall_seconds']
        metrics = bp_metrics((OUT/f'bp-{name}.log').read_text())
        independent = configured(config,'independent-'+name,'',{})
        independent['commands'][0]['args'] = independent_args(independent['commands'][0]['args'])
        timings['independent'] = execute(independent)['wall_seconds']
        if initial_states(OUT/f'bp-{name}_consistency.csv') != initial_states(OUT/f'independent-{name}_consistency.csv'):
            raise ValueError('independent initial states changed')
        metrics.update(readout_metrics((OUT/f'independent-{name}.log').read_text()))
        failures = result['summary']['total_failures']
        metrics.update(normal_radius_base=n,tangential_radius_base=t, timings=timings,
            failures=failures, replay_percent=100/(failures+1),
            filtered_stops=sum(len(c['filtered_stops']) for c in result['cases']),
            retained_percent=result['summary']['length_weighted_success_percent'],
            trace_count=attrs['trace_count'], dense_retained_percent=100*summary['retained_length_base']/
                (summary['retained_length_base']+summary['reverse_skipped_length_base']),
            initial_states_identical=True)
        rows[name] = metrics
        report(rows)
        print('VARIANT COMPLETE '+name+' '+json.dumps({k:v for k,v in metrics.items() if k!='ladder'}),flush=True)
    publications = []
    for name,_,_ in VARIANTS:
        entry = dict(id=f'raw40500-strict-{name}',label=f'Raw40500 prefix512 {name}',run_record=REPORT)
        publications.append(dict(output_root=str(OUT),reference_dir=replay_template['reference_dir'],commands=[],
            measurement_date='2026-10-02',publish_replay_plots=[dict(entry,result=f'replay-{name}.json')],
            publish_bp_plots=[dict(entry,log=f'bp-{name}.log')]))
    for config in publications:
        run(config)


if __name__ == '__main__':
    main()
