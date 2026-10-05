"""Local event BP ablation on immutable isotropic20 traces; no retracing."""
import json
import re
import subprocess
from pathlib import Path
import strict_reverse_crop as shared
from reviewed_bp_independent import independent_args, initial_states, readout_metrics
from lasagna.inference_provenance import atomic_write
from vesuvius.neural_tracing.fiber_trace_3d.crop_benchmark import bp_metrics, run

SOURCE = shared.OUT
OUT = SOURCE.parent/'20261002_iso20_event_bp_v1'
REPORT = 'docs/fiber_benchmark_runs/2026-10-02-iso20-event-bp.md'


def configuration(template):
    config = shared.configured(template,'bp-events','',{},extra=['--local-constraint-events'])
    return config


def independent_diagnostic():
    shared.OUT=OUT
    config=configuration(json.loads((SOURCE/'bp-isotropic20.config.json').read_text()))
    config=shared.configured(config,'independent-events','',{})
    config['commands'][0]['args']=independent_args(config['commands'][0]['args'])
    shared.execute(config)
    publish_partial()


def publish_partial():
    result=readout_metrics((OUT/'independent-events.log').read_text())
    baseline=json.loads((SOURCE/'results.json').read_text())['isotropic20']
    failed=json.loads((OUT/'bp-events.run.json').read_text())
    independent=json.loads((OUT/'independent-events.run.json').read_text())
    result.update(status='partial_oracle_calibration_failure',oracle_error='Reference-connected BP gauge has no calibration',
        constraint_score_percent=None,reference_score_percent=None,replay_percent=100,
        replay_inherited=True,bp_seconds=failed['wall_seconds'],independent_seconds=independent['wall_seconds'],
        initial_states_identical=None)
    log=(OUT/'independent-events.log').read_text()
    result['event_diagnostics']=[line for line in log.splitlines() if line.startswith('local constraint events:') or 'missing_owner' in line]
    target=shared.REPO/'volume-cartographer'/REPORT
    rows={'overlapping512_128':baseline,'local_events':result}
    atomic_write(OUT/'results.json',rows,durable=False)
    atomic_write(target.with_suffix('.json'),rows,durable=False)
    table=['| BP construction | Replay | Constraint | Oracle accuracy | Non-oracle accuracy | Observed shift |',
        '|---|---:|---:|---:|---:|---:|',
        '| Overlapping512/128 | 100.00% | -25.24% | 80.77% | 50.00% | 83.33% |',
        f"| Local events | 100.00% | — | — | {result['independent_percent']:.2f}% | {result['observed_ladder_shift']['ladder_shift_percent']:.2f}% |"]
    text=['# Isotropic20 local-event splitting — partial benchmark','',
        'Same saved raw40500 isotropic20/full-prefix512/no-history traces; only --local-constraint-events added '
        'to hard-both BP. Existing mode bundles certification/refinement, midpoint cuts between orthogonal '
        'events, parallel-category transition cuts, no overlap, continuity and source-based reference cross constraints. '
        'Parallel factors retained. Local defaults: spacing64,tangent16,parallel span64,reciprocal tolerance8 '
        'base voxels; angular30degrees,winding tolerance0.25,refinement2. No native solver edits or GPU/inference/retracing.','',
        *table,'','The oracle attempt failed after its initial500-message solve: '+result['oracle_error']+'. '
        'No constraint/oracle scores exist. No arbitrary calibration or constraint deletion was applied. '
        'The independent native run succeeded. Its additional empty satellite-gauge summaries have no admitted '
        'or evaluated evidence and are ignored by the parser, while all26references remain in the denominator; '
        'multiple genuinely scored gauges remain an error.','',
        'Independent accuracy3/26; missing4and12. Shift23offset changes/24observed references. '
        'Two references(3.5and8) have tied/unbounded best assignments; scores use deterministic rank1. '
        'This is not a uniquely determined ladder. Initial-state equality could not be checked because '
        'the oracle attempt failed before writing its consistencyCSV. Both used identical initial configurations.','',
        'Replay is inherited unchanged:0ordinary errors,6filtered stops,93.7463%retained reference distance. '
        'Graph1879source traces;118010pieces versus2732baseline;354595constraints versus243615, '
        'including116131continuity links. The constraint denominator itself changes in this ablation.','',
        *result['event_diagnostics'],'',
        f"Oracle attempt {failed['wall_seconds']:.2f}s (failed); independent {independent['wall_seconds']:.2f}s. "
        'Cached CPU128threads,RelWithDebInfo,one repetition. No complete oracle plot point published.','',
        f'Artifacts: `{OUT}`. Approved runner `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, '
        'action `iso20_event_bp`; completed independent/report-only refresh action `iso20_event_readout`.']
    target.write_text('\n'.join(text)+'\n')
    historical=target.parent/'2026-10-02-prefix512.md'
    lines=[line for line in historical.read_text().splitlines() if not line.startswith('| raw40500 iso20 local events |')]
    anchor=next(i for i,line in enumerate(lines) if line.startswith('| raw40500 strict isotropic20,'))
    lines.insert(anchor+1,table[-1].replace('| Local events |','| raw40500 iso20 local events |'))
    note='Local-event result is partial: [oracle calibration failure and non-oracle ambiguity details](2026-10-02-iso20-event-bp.md).'
    if note not in lines: lines+=['',note]
    historical.write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2),flush=True)


def main():
    shared.OUT = OUT
    OUT.mkdir(parents=True,exist_ok=True)
    template = json.loads((SOURCE/'bp-isotropic20.config.json').read_text())
    config = configuration(template)
    try:
        record = shared.execute(config)
    except subprocess.CalledProcessError:
        if 'Reference-connected BP gauge has no calibration' not in (OUT/'bp-events.log').read_text():
            raise
        independent_diagnostic()
        return
    independent = shared.configured(config,'independent-events','',{})
    independent['commands'][0]['args'] = independent_args(independent['commands'][0]['args'])
    independent_record = shared.execute(independent)
    if initial_states(OUT/'bp-events_consistency.csv') != initial_states(OUT/'independent-events_consistency.csv'):
        raise ValueError('Independent initial states differ')
    baseline = json.loads((SOURCE/'results.json').read_text())['isotropic20']
    log = (OUT/'bp-events.log').read_text()
    result = bp_metrics(log)
    result['oracle_missing_references'] = result.pop('missing_references')
    result.update(readout_metrics((OUT/'independent-events.log').read_text()))
    result.update(replay_percent=baseline['replay_percent'],filtered_stops=baseline['filtered_stops'],
        retained_percent=baseline['retained_percent'],replay_inherited=True,initial_states_identical=True,
        bp_seconds=record['wall_seconds'],independent_seconds=independent_record['wall_seconds'],
        local_summary=re.findall(r'^local constraint events:.*$',log,re.M),
        source_event_summary=[line for line in log.splitlines() if 'missing_owner' in line],
        cohort_summary=re.findall(r'^fiber direction BP-only cohort.*$',log,re.M))
    rows={'overlapping512_128':baseline,'local_events':result}
    atomic_write(OUT/'results.json',rows,durable=False)
    target=shared.REPO/'volume-cartographer'/REPORT
    atomic_write(target.with_suffix('.json'),rows,durable=False)
    lines=['# Isotropic20: overlapping pieces versus local constraint events','',
        'Same raw40500 isotropic20 full-prefix512/no-history beam traces and hard-both BP objective. '
        'Added only --local-constraint-events. This existing mode includes source-level event certification, '
        'local consistency checks/refinement, event-midpoint segmentation and zero overlap; it is not merely '
        'a change of piece length. Parallel factors remain enabled. Local defaults: spacing64, tangent span16, '
        'parallel span64, reciprocal tolerance8base voxels; angular30degrees, winding tolerance0.25, '
        'two refinement rounds. Splits include parallel-category transitions and adjacent pieces get continuity. '
        'No inference or retracing.','',
        '| BP construction | Replay | Constraint | Oracle accuracy | Non-oracle accuracy | Observed shift |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        lines.append(f"| {name} | {r['replay_percent']:.2f}% | {r['constraint_score_percent']:.2f}% | "
            f"{r['reference_score_percent']:.2f}% | {r['independent_percent']:.2f}% | "
            f"{r['observed_ladder_shift']['ladder_shift_percent']:.2f}% |")
    lines += ['',f"Event BP {result['bp_seconds']:.2f}s; independent {result['independent_seconds']:.2f}s. "
        f"Missing/tied/unbounded readouts: {result['independent_missing']}/{result['tied_rows']}/{result['unbounded_rows']}.",
        '',*result['cohort_summary'],*result['local_summary'],*result['source_event_summary'],'',
        'Replay inherited unchanged: zero ordinary errors,6filtered stops,93.7463%reference distance retained. '
        'Accuracy denominator26. Observed shift omits missing references and bridges gaps; denominator and ladder '
        'are recorded in JSON. Constraint score is −100×problematic/retained-fulfilled, not removal percentage. '
        'Oracle is GT-assisted; independent is reference-free BP followed by frozen-neighbor readout. '
        'Initial states checked identical between the two solves. Rank1 scores may be ambiguous if ties/unbounded are nonzero.','',
        f'Artifacts: `{OUT}`. Approved command: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`, '
        'action `iso20_event_bp`. Cached CPU,128threads,RelWithDebInfo,one repetition.']
    target.write_text('\n'.join(lines)+'\n')
    historical=target.parent/'2026-10-02-prefix512.md'
    text=historical.read_text().splitlines()
    text=[line for line in text if not line.startswith('| raw40500 iso20 local events |')]
    anchor=next(i for i,line in enumerate(text) if line.startswith('| raw40500 strict isotropic20,'))
    text.insert(anchor+1,next(line for line in lines if line.startswith('| local_events |')).replace(
        '| local_events |','| raw40500 iso20 local events |'))
    historical.write_text('\n'.join(text)+'\n')
    entry=dict(id='raw40500-iso20-events',label='Raw40500 iso20 local events',run_record=REPORT)
    run(dict(output_root=str(OUT),reference_dir=config['reference_dir'],commands=[],measurement_date='2026-10-02',
        publish_bp_plots=[dict(entry,log='bp-events.log')],
        publish_replay_plots=[dict(entry,result=str(SOURCE/'replay-isotropic20.json'))]))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
