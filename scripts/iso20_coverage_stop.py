"""1024-only matched coverage-stop pilot through the established native runner."""
import json
import re
import subprocess
from pathlib import Path
import strict_reverse_crop as shared
from reviewed_bp_independent import readout_metrics
from iso20_probe_removal import oracle_args, pruning_counts
from vesuvius.neural_tracing.fiber_trace_3d.crop_benchmark import bp_metrics
from lasagna.inference_provenance import atomic_write

SOURCE=shared.OUT
WIN=SOURCE.parent/'20261003_iso20_iterative_v1'
OUT=SOURCE.parent/'20261003_iso20_coverage_stop_v1'
REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-03-coverage-stop'
VARIANTS=[('control',None),('stop40x80',(40,80)),('stop20x40',(20,40)),('stop10x20',(10,20)),('stop5x10',(5,10)),('stop2x4',(2,4))]


def configs(name,radii,trace_template,bp_template):
    trace=OUT/f'{name}.zarr'
    extra=() if radii is None else ('--stop-at-covered-traces','--covered-trace-normal',str(radii[0]),'--covered-trace-tangential',str(radii[1]))
    tc=shared.configured(trace_template,name,'.zarr',{},extra)
    bc=shared.configured(bp_template,name+'-ind','',{})
    args=bc['commands'][0]['args'];old=Path(args[1]);args[1]=str(trace)
    bc['input_files']=[str(trace/'.zattrs') if p==str(old/'.zattrs') else p for p in bc['input_files']]
    return tc,bc


def publish(rows):
    atomic_write(OUT/'results.json',rows,durable=False)
    atomic_write(REPORT.with_suffix('.json'),rows,durable=False)
    lines=['# Indexed coverage stopping:1024 pilot','',
        'Same raw40500,beam8/lookahead2,full-prefix512/iso20. Seed coverage remains40/80,angle45; '
        'only accepted-trace stopping radii vary. Same selected median64/minimum256 BP costs,500messages, '
        '128threads,RelWithDebInfo,one run per variant,cached predictions/anchors,no GPU or2048run.','',
        '| Variant | Trace pieces | Length vs control | Trace s | Commit s | Coverage stops | BP pieces | Constraints | Non-oracle ↑ | Observed shift ↑ | Removal ↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    control=rows.get('control',{}).get('summary',{}).get('retained_length_base',1.)
    for name,_ in VARIANTS:
        if name not in rows:continue
        r=rows[name]
        s=r['summary'];i=r.get('independent',{});o=r.get('oracle',{})
        def pct(v):return 'unavailable' if v is None else f'{v:.4f}%'
        commit=f"{s['integration_seconds']:.2f}" if 'integration_seconds' in s else 'not recorded'
        lines.append(f"| {name} | {r['traces']} | {100*s['retained_length_base']/control:.2f}% | {r['trace_seconds']:.2f} | {commit} | {s.get('covered_trace_stops',0)} | {r.get('pieces','—')} | {r.get('constraints','—')} | {pct(i.get('independent_percent'))} | {pct(i.get('observed_ladder_shift',{}).get('ladder_shift_percent'))} | {pct(o.get('constraint_score_percent'))} |")
    lines+=['','## Verdict and implementation','',
        'Keep disabled by default. All tested stops reduce geometry and wall time but lose non-oracle accuracy versus control. '
        'The best stopping accuracy is5/10; no2048trial was run. Smaller graphs alone are not a quality improvement. '
        'Stopping does not create a continuation/equality link to the contacted trace in BP; whether this or changed '
        'coverage/weighting explains the regression has not been isolated.','',
        'Opt in with `--stop-at-covered-traces --covered-trace-normal N --covered-trace-tangential T`. '
        'Spatial buckets hold accepted segments; direction and sphere rejection precede normal sampling. '
        'Workers use submission watermarks and ordered commit rechecks raw prefixes before terminal reverse verification. '
        'Both sides are checked before publishing either. Commit seconds include ordinary seed suppression and indexing, '
        'not just reverse rechecks. One run per setting: these are end-to-end timings, not latency distributions.',
        '', 'Build/tests: approved runner with control action `build`; experiment action `iso20_coverage_stop`, '
        'optional `report_only:true`. Command: `bash /home/hendrik/villa3/scripts/run_training_failure_mining`. '
        'Eight Python config/helper checks and native regression suites passed. No-flag OBJ geometry exactly matches historical1024. '
        'Final code also matches worker per-edge length arithmetic on arbitrary scales; the benchmark uses scale2 where this is unchanged.',
        '', 'Lengths are summed retained geometry, not reference coverage. Observed shift excludes missing references; '
        'non-oracle accuracy denominator26. Removal is separate reference-conditioned oracle score. '
        'Direct reference-seeded replay does not exercise accepted-trace coverage and has not been rerun/misrepresented as a new score. '
        'Artifact replay needs native collection export; not measured in this pilot. Full metrics/errors and counters in JSON.',
        '',f'Artifacts/configs/provenance:`{OUT}`. Approved runner action `iso20_coverage_stop`. Defaults unchanged.']
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main(report_only=False):
    OUT.mkdir(parents=True,exist_ok=True);shared.OUT=OUT
    rows=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else {}
    if report_only:publish(rows);return
    tt=json.loads((SOURCE/'beam-isotropic20.config.json').read_text())
    bt=json.loads((WIN/'c1681f71b4163-ind.config.json').read_text())
    for name,radii in VARIANTS:
        tc,bc=configs(name,radii,tt,bt)
        rec=shared.execute(tc)
        art=json.loads((OUT/f'{name}.zarr/.zattrs').read_text())['processing']['artifact']
        summary={k:v for k,v in art['trace_summary'].items() if k!='reverse_piece_provenance'}
        if name=='control':
            assert (OUT/'control.obj').read_bytes()==(SOURCE/'beam-isotropic20.obj').read_bytes(),'no-flag geometry changed'
        rows[name]=r=dict(traces=art['trace_count'],summary=summary,trace_seconds=rec['wall_seconds'],radii=radii)
        publish(rows)
        for stage in ('ind','oracle'):
            config=bc if stage=='ind' else shared.configured(bc,name+'-oracle','',{})
            if stage=='oracle':config['commands'][0]['args']=oracle_args(config['commands'][0]['args'])
            try:
                br=shared.execute(config)
                log=(OUT/f'{name}-{stage}.log').read_text()
                if stage=='ind':
                    match=re.search(r'BP-only cohort .*pieces=(\d+) selected_constraints=(\d+)',log)
                    r.update(pieces=int(match[1]),constraints=int(match[2]))
                    r['independent']=readout_metrics(log)
                else:r['oracle']=dict(bp_metrics(log),counts=pruning_counts(log))
                r[stage+'_seconds']=br['wall_seconds']
            except (subprocess.CalledProcessError,ValueError) as exc:
                r[stage+'_error']=str(exc)
            publish(rows)
        print('COMPLETE '+name+' '+json.dumps({k:v for k,v in r.items() if k not in ('summary','independent','oracle')}),flush=True)


if __name__=='__main__':
    import sys
    main('--report-only' in sys.argv)
