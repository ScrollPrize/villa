"""Orthogonal-only event segmentation sweep on frozen isotropic20 traces."""
import csv
import json
import re
import subprocess
import numpy as np
import strict_reverse_crop as shared
from iso20_event_bp import SOURCE
from reviewed_bp_independent import independent_args, initial_states, readout_metrics
from lasagna.inference_provenance import atomic_write
from vesuvius.neural_tracing.fiber_trace_3d.crop_benchmark import bp_metrics, run

OUT=SOURCE.parent/'20261002_iso20_orthogonal_minimum_v1'
MINIMUMS=(8,16,32,64,128,256)
REPORT='docs/fiber_benchmark_runs/2026-10-02-iso20-orthogonal-minimum.md'


def configuration(template,minimum):
    return shared.configured(template,f'bp-min{minimum}','',{},extra=[
        '--local-constraint-events','--local-min-piece-length',str(minimum)])


def fmt(value):
    return '—' if value is None else f'{value:.2f}%'


def matching_failed_oracle(config, independent_record):
    """Reuse a recorded known failure only with identical current provenance."""
    case=config['commands'][0]
    path=OUT/f"{case['name']}.run.json"
    log=OUT/f"{case['name']}.log"
    if not path.exists() or not log.exists():return None
    old=json.loads(path.read_text())
    expected=[independent_record['command'][0],*case['args']]
    keys=('binary_sha256','shared_libraries','inputs','references')
    if (old['returncode']!=0 and old['command']==expected and
        all(old[k]==independent_record[k] for k in keys) and
        'Reference-connected BP gauge has no calibration' in log.read_text()):
        return old
    return None


def report(rows):
    baseline=json.loads((SOURCE/'results.json').read_text())['isotropic20']
    old=json.loads((SOURCE.parent/'20261002_iso20_event_bp_v1/results.json').read_text())['local_events']
    comparison={'overlapping512/128':baseline,'previous event mode':old,**rows}
    path=shared.REPO/'volume-cartographer'/REPORT
    atomic_write(OUT/'results.json',comparison,durable=False)
    atomic_write(path.with_suffix('.json'),comparison,durable=False)
    lines=['# Isotropic20 orthogonal-only minimum-length sweep','',
        'Same saved raw40500/full-prefix512/isotropic20/no-history traces. Only segmentation changes: '
        'no parallel-transition cuts; greedy ascending orthogonal-event midpoint candidates must leave '
        'both previous piece and final tail at least the configured minimum. Short whole source traces '
        'remain intact. Multiple parallel/orthogonal measurements may share a piece. No changes to '
        'certification/refinement/evidence weights, defect/break costs, reference policy or500message budget. '
        '128CPU threads,RelWithDebInfo,one repetition; no inference or retracing.','',
        '| Construction | Replay | Constraint | Oracle accuracy | Non-oracle accuracy | Observed shift |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in comparison.items():
        lines.append(f"| {name} | 100.00% | {fmt(r.get('constraint_score_percent'))} | "
            f"{fmt(r.get('reference_score_percent'))} | {fmt(r['independent_percent'])} | "
            f"{fmt(r['observed_ladder_shift']['ladder_shift_percent'])} |")
    lines+=['','| Minimum | Pieces | Median length | Minimum length | Short intact sources | Constraints | Active components | Missing/tied/unbounded | Oracle / independent seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        lines.append(f"| {r['minimum']} | {r['pieces']} | {r['median_length']:.3f} | {r['min_length']:.3f} | "
            f"{r['short_intact_sources']} | {r['constraints']} | {r['active_components']} | "
            f"{r['independent_missing']}/{r['tied_rows']}/{r['unbounded_rows']} | "
            f"{r['oracle_seconds']:.2f} / {r['independent_seconds']:.2f} |")
    lines+=['','Replay inherited unchanged:0ordinary errors,6filtered stops,93.7463%reference distance retained. '
        'Accuracy denominator26; observed shift omits missing references and bridges their gaps. '
        'Rank1scores are not uniquely determined when tied/unbounded readouts exist. '
        'Constraint score−100×problematic/retainedfulfilled (higher better),not removal percent. '
        'Oracle calibration failures remain unavailable; no arbitrary offsets or constraint removal. '
        'All initial solves retain500messages; convergence details and full ladders in JSON.','']
    for name,r in rows.items():
        lines += [f"## {name}", '',r.get('oracle_error','Oracle and independent initial states match.'),
            *r['diagnostics'],'']
    lines += [f'Artifacts:`{OUT}`. Approved command:`bash /home/hendrik/villa3/scripts/run_training_failure_mining`, '
        'control action `iso20_minimum_sweep`. Native build/tests:action `build`.']
    path.write_text('\n'.join(lines)+'\n')
    history=path.parent/'2026-10-02-prefix512.md'
    h=[line for line in history.read_text().splitlines() if not line.startswith('| raw40500 iso20 orth-min')]
    anchor=next(i for i,line in enumerate(h) if line.startswith('| raw40500 iso20 local events |'))
    new=[line.replace('| min','| raw40500 iso20 orth-min',1) for line in lines if line.startswith('| min')]
    h[anchor+1:anchor+1]=new
    note='Orthogonal-only minimum-length sweep: [full results and caveats](2026-10-02-iso20-orthogonal-minimum.md).'
    if note not in h:h+=['',note]
    history.write_text('\n'.join(h)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    shared.OUT=OUT
    template=json.loads((SOURCE/'bp-isotropic20.config.json').read_text())
    rows={}
    for minimum in MINIMUMS:
        config=configuration(template,minimum)
        independent=shared.configured(config,f'independent-min{minimum}','',{})
        independent['commands'][0]['args']=independent_args(independent['commands'][0]['args'])
        record=shared.execute(independent)
        log=(OUT/f'independent-min{minimum}.log').read_text()
        r=readout_metrics(log)
        with (OUT/f'independent-min{minimum}_local_event_pieces.csv').open() as f:
            pieces=list(csv.DictReader(f))
        lengths=np.array([float(p['end_arc_base'])-float(p['begin_arc_base']) for p in pieces])
        counts={}
        for p in pieces:counts[p['original_trace']]=counts.get(p['original_trace'],0)+1
        short=[p for p,l in zip(pieces,lengths) if l<minimum-1e-7]
        assert all(counts[p['original_trace']]==1 and float(p['begin_arc_base'])==0 for p in short)
        summary=re.search(r'local constraint events: pieces=(\d+) constraints=(\d+).*transition_cuts=(\d+)',log)
        assert summary and int(summary[3])==0 and int(summary[1])==len(pieces)
        connectivity=log.split('BP hard-order connectivity\n')[1].split('BP final states')[0]
        active=next(line.split() for line in connectivity.splitlines() if line.startswith('active '))
        r.update(minimum=minimum,pieces=len(pieces),median_length=float(np.median(lengths)),
            min_length=float(lengths.min()),total_length=float(lengths.sum()),short_intact_sources=len(short),
            constraints=int(summary[2]),active_components=int(active[3]),independent_seconds=record['wall_seconds'],
            replay_percent=100,replay_inherited=True,
            diagnostics=[line for line in log.splitlines() if line.startswith('local constraint events:') or
                'missing_owner' in line or line.startswith('message_limit ') or line.startswith('converged ')])
        try:
            failed=matching_failed_oracle(config,record)
            if failed:
                raise subprocess.CalledProcessError(failed['returncode'],failed['command'])
            oracle=shared.execute(config)
        except subprocess.CalledProcessError:
            err=(OUT/f'bp-min{minimum}.log').read_text()
            if 'Reference-connected BP gauge has no calibration' not in err:raise
            oracle=json.loads((OUT/f'bp-min{minimum}.run.json').read_text())
            r.update(constraint_score_percent=None,reference_score_percent=None,
                oracle_error='Reference-connected BP gauge has no calibration',initial_states_identical=None)
        else:
            oracle_metrics=bp_metrics((OUT/f'bp-min{minimum}.log').read_text())
            oracle_metrics['oracle_missing_references']=oracle_metrics.pop('missing_references')
            r.update(oracle_metrics)
            assert initial_states(OUT/f'bp-min{minimum}_consistency.csv')==initial_states(OUT/f'independent-min{minimum}_consistency.csv')
            r['initial_states_identical']=True
            entry=dict(id=f'raw40500-iso20-orth-min{minimum}',label=f'Raw40500 iso20 orth min{minimum}',run_record=REPORT)
            run(dict(output_root=str(OUT),reference_dir=config['reference_dir'],commands=[],measurement_date='2026-10-02',
                publish_bp_plots=[dict(entry,log=f'bp-min{minimum}.log')],
                publish_replay_plots=[dict(entry,result=str(SOURCE/'replay-isotropic20.json'))]))
        r['oracle_seconds']=oracle['wall_seconds']
        rows[f'min{minimum}']=r
        report(rows)
        print('COMPLETE '+json.dumps({k:v for k,v in r.items() if k!='ladder'}),flush=True)


if __name__=='__main__':main()
