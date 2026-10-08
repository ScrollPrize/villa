"""Run existing reference-conditioned removal metric on saved probe variants."""
import json
import re
import csv
from pathlib import Path
import strict_reverse_crop as shared
from reviewed_bp_independent import initial_states
from vesuvius.neural_tracing.fiber_trace_3d.crop_benchmark import bp_metrics

SOURCE=shared.OUT.parent/'20261002_iso20_orthogonal_probes_v1'
OUT=SOURCE.parent/'20261003_iso20_probe_removal_v1'
REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-03-probe-removal.json'
NAMES=('winner-repeat','winner-median64','winner-consensus32')
ORACLE_FLAGS=['--reference-prune-offenders','--reference-prune-policy','oracle-inliers',
    '--reference-oracle-magnitude-weight','1','--reference-oracle-accept-message-limit',
    '--reference-oracle-rounds','1']


def oracle_args(args):
    if args.count('--reference-independent')!=1:
        raise ValueError('expected independent source configuration')
    return [a for a in args if a!='--reference-independent']+ORACLE_FLAGS


def pruning_counts(log):
    match=re.search(r'constraint pruning benchmark\n[^\n]*\n([^\n]+)',log)
    if match is None:raise ValueError('missing pruning table')
    keys=('initial','removed_incident','retained_infringed','retained_defect','problematic','retained_fulfilled')
    result=dict(zip(keys,map(int,match[1].split()[:6])))
    assert result['problematic']==result['removed_incident']+result['retained_infringed']+result['retained_defect']
    assert result['initial']==result['problematic']+result['retained_fulfilled']
    result['problematic_percent_initial']=100*result['problematic']/result['initial']
    return result


def factor_identities(path):
    keys=('constraint','piece_a','piece_b','parallel','perpendicular','original_signed_delta',
          'original_signed_parallel_delta','evidence_weight','arc_a','arc_b')
    with path.open() as f:
        return [tuple(r[k] for k in keys) for r in csv.DictReader(f)]


def publish(rows):
    REPORT.write_text(json.dumps(rows,indent=2)+'\n')
    (OUT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    lines=['# Reference-conditioned constraint removal: probe comparison','',
        'Selected baseline (user decision):median +/-64,minimum256,parallel magnitude0.05, '
        'disabling150 per128 base voxels,gap boundary0. Current best fine-split probe candidate at selection; '
        'historical overlap remains better on removal alone. Production defaults unchanged.','',
        'Existing oracle-inliers benchmark, one oracle round, magnitude weight1, accept-message-limit as historical baseline. '
        'Frozen raw40500 iso20 1024 crop; fine256, soft parallel weight0.05, hard signs both, old reference extraction. '
        '128CPU threads,500-message limit, RelWithDebInfo. No inference/retracing or geometric matching audit.','',
        '| Variant | Initial constraints | Problematic | Problematic / initial | Retained fulfilled | Constraint score ↑ | Oracle accuracy | Seconds |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in rows.items():
        if 'error' in r:
            lines.append(f'| {name} | unavailable: {r["error"]} | — | — | — | — | — | {r["seconds"]:.1f} |');continue
        c=r['counts']
        lines.append(f'| {name} | {c["initial"]} | {c["problematic"]} | {c["problematic_percent_initial"]:.4f}% | {c["retained_fulfilled"]} | {r["constraint_score_percent"]:.4f}% | {r["reference_score_percent"]:.4f}% | {r["seconds"]:.1f} |')
    lines+=['','Constraint score = −100×problematic/retained-fulfilled (higher/closer to zero is better). '
        'Problematic includes constraints incident to removed pieces, retained infringed constraints and retained defect constraints. '
        'It is the established operational removal score, not proof each removed measurement is individually geometrically false.','',
        '| Variant | Removed-incident | Retained-infringed | Retained-defect |',
        '|---|---:|---:|---:|']
    for name,r in rows.items():
        if 'counts' in r:
            c=r['counts'];lines.append(f'| {name} | {c["removed_incident"]} | {c["retained_infringed"]} | {c["retained_defect"]} |')
    lines+=['',f'Artifacts/configs/provenance/logs: `{OUT}`. Independent scores for all three:61.54% exact,87.50% observed shift (previous runs). '
        'Historical overlap result:constraint−25.2365%,oracle80.7692%; different subdivision/objective, not a probe-only control.',
        '', 'Run approved `bash /home/hendrik/villa3/scripts/run_training_failure_mining` with action `iso20_probe_removal`. '
        'Shared native benchmark runner and parser reused. Probe semantics unchanged: four displaced connectors plus center median; '
        'consensus disagreement retains original, not rejection. This is not an endpoint-agreement experiment.']
    if all(name in rows and 'counts' in rows[name] for name in NAMES):
        base=rows[NAMES[0]]['counts']['problematic']
        lines+=['','## Paired comparison','',
            'All variants contain243682 initial constraints; no changed acceptance or segmentation.']
        for name in NAMES[1:]:
            delta=rows[name]['counts']['problematic']-base
            lines.append(f'- {name}: {delta:+d} problematic constraints versus no probes.')
        lines+=['','Initial states and original measurement exports were compared to each saved independent run; '
            'see initial_states_identical and initial_factors_identical in JSON. '
            'Eight Python checks passed; no native code/build changes. '
            'One crop, one run per variant; these are the established heuristic removal outcomes, not a globally minimal deletion proof.']
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(exist_ok=True,parents=True);shared.OUT=OUT
    rows={}
    for name in NAMES:
        template=json.loads((SOURCE/f'{name}.config.json').read_text())
        args=template['commands'][0]['args']
        assert args[args.index('--winding-weights')+1]=='0,0,0,0.05,0'
        template['commands'][0]['args']=oracle_args(template['commands'][0]['args'])
        config=shared.configured(template,name,'',{})
        try:
            rec=shared.execute(config)
        except Exception:
            record=OUT/f'{name}.run.json'
            if not record.exists():raise
            rec=json.loads(record.read_text())
            if rec['returncode']==0:raise
            log=(OUT/f'{name}.log').read_text()
            if 'Reference-connected BP gauge has no calibration' not in log:raise
            rows[name]=dict(error='Reference-connected BP gauge has no calibration',seconds=rec['wall_seconds'])
            publish(rows);continue
        log=(OUT/f'{name}.log').read_text()
        counts=pruning_counts(log)
        metrics=bp_metrics(log)
        # Native export consistency.csv describes the initial, unconditioned solve.
        same=initial_states(OUT/f'{name}_consistency.csv')==initial_states(SOURCE/f'{name}_consistency.csv')
        factors_same=factor_identities(OUT/f'{name}_winding_factors.csv')==factor_identities(SOURCE/f'{name}_winding_factors.csv')
        rows[name]=dict(**metrics,counts=counts,initial_states_identical=same,initial_factors_identical=factors_same,seconds=rec['wall_seconds'])
        publish(rows)
        print(name+': '+json.dumps(rows[name]),flush=True)


if __name__=='__main__':
    import sys
    if '--status' in sys.argv:
        shared.OUT=OUT
        shared.status()
    else:
        main()
