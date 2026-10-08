"""Matched 2048 context test of the selected raw40500/iso20 BP configuration."""
import json
from pathlib import Path
import strict_reverse_crop as shared
from reviewed_bp_prefix import prepare_coverage
from reviewed_bp_independent import readout_metrics
from lasagna.inference_provenance import atomic_write

OUT=shared.OUT.parent/'20261003_iso20_best2048_v1'
WIN=OUT.parent/'20261003_iso20_iterative_v1'
TRACE_TEMPLATE=shared.OUT/'beam-isotropic20.config.json'
LO=(9728,21504,5632)
HI=(11776,23552,7680)
REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-03-best2048'


def trace_config(template):
    config=shared.configured(template,'beam2048','.zarr',{})
    args=config['commands'][0]['args'];i=args.index('--bbox')
    args[i+1:i+7]=list(map(str,(*LO,*HI)))
    return config


def bp_config(template,trace):
    config=shared.configured(template,'independent2048','',{})
    args=config['commands'][0]['args']
    old=Path(args[1]);args[1]=str(trace)
    config['input_files']=[str(trace/'.zattrs') if p==str(old/'.zattrs') else p for p in config['input_files']]
    assert '--reference-independent' in args and '--reference-prune-offenders' not in args
    return config


def publish(data):
    atomic_write(REPORT.with_suffix('.json'),data,durable=False)
    small=data['small'];large=data['large']
    def pct(value):return 'unavailable' if value is None else f'{value:.4f}%'
    lines=['# Current best BP parameters: 1024 versus 2048 context','',data.get('protocol',
        'Same raw40500 snapshot, beam8 full-prefix512 isotropic20, reference set and selected BP parameters. '
        'Minimum256, median±64/blend1, boundary10, disabling100 per128, P1=.0375, '
        'other magnitude weights0, hard signs both, scale.822/phase.5,500 messages,128 native threads. '
        'Independent fixed-graph reference readout only; no oracle pruning or parameter retuning.'),'',
        '| Crop | Non-oracle accuracy ↑ | Observed shift ↑ | Removal ↑ | Oracle accuracy ↑ | Missing | Tied | Unbounded |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in [('1024',small),('2048',large)]:
        oracle=data.get('oracle',{}).get(name,{})
        lines.append(f"| {name} | {pct(r.get('independent_percent'))} | {pct(r.get('observed_ladder_shift',{}).get('ladder_shift_percent'))} | {pct(oracle.get('constraint_score_percent'))} | {pct(oracle.get('reference_score_percent'))} | {r.get('independent_missing','unavailable')} | {r.get('tied_rows','unavailable')} | {r.get('unbounded_rows','unavailable')} |")
    lines+=['','| Reference | 1024 assignment | 2048 assignment | 2048 hard-sign errors |','|---:|---:|---:|---:|']
    a={r['reference']:r for r in small.get('ladder',[])};b={r['reference']:r for r in large.get('ladder',[])}
    refs=set(a)|set(b)|set(small.get('missing_references',[]))|set(large.get('missing_references',[]))
    for ref in sorted(refs,key=float):
        lines.append(f"| {ref} | {a.get(ref,{}).get('winding','Missing')} | {b.get(ref,{}).get('winding','Unavailable' if large.get('independent_percent') is None else 'Missing')} | {b.get(ref,{}).get('hard_sign_errors','—')} |")
    lines+=['', 'Accuracy denominator26; observed shift excludes missing references, whose counts may differ across crops. '
        'Hard-sign errors belong to reference readout, not the graph-wide violation count. '
        'The larger crop changes the traced candidates and constraint graph; this is a context test, not just a BP rerun on identical candidates.',
        '',f'2048 bounds XYZ `{LO}`–`{HI}`. Trace/BP seconds: `{data["timings"]}`. '+
        data.get('execution_note','One run each, native RelWithDebInfo; existing inference/anchors reused and missing prediction coverage filled through the ordinary managed pipeline.'),
        '',f'Artifacts/configs/provenance: `{OUT}`. Approved runner action `'+data.get('action','iso20_best2048')+'`; '
        '`report_only:true` refreshes saved report without execution. No runtime-default changes.']
    lines+=data.get('extra_markdown',[])
    REPORT.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main(report_only=False):
    if report_only:
        publish(json.loads((OUT/'results.json').read_text()));return
    OUT.mkdir(parents=True,exist_ok=True);shared.OUT=OUT
    state=json.loads((WIN/'state.json').read_text());winner=state['incumbent']
    assert state['stage']=='complete'
    template=json.loads(TRACE_TEMPLATE.read_text())
    print('Preparing current-model 2048 coverage',flush=True)
    _,source,manifest,_=prepare_coverage(LO,HI,out=OUT)
    args=template['commands'][0]['args']
    assert args[args.index('--fiber-manifest')+1]==str(manifest)
    assert args[args.index('--normal-manifest')+1]==source['normal_manifest']
    trace=shared.execute(trace_config(template))
    template=json.loads((WIN/f'{winner}-ind.config.json').read_text())
    rec=shared.execute(bp_config(template,OUT/'beam2048.zarr'))
    for key in ('binary_sha256','shared_libraries','references'):
        assert rec[key]==state['provenance'][key],f'baseline provenance changed: {key}'
    data=dict(small=state['rows'][winner]['independent'],large=readout_metrics((OUT/'independent2048.log').read_text()),
        params=state['rows'][winner]['params'],winner=winner,timings=dict(trace=trace['wall_seconds'],bp=rec['wall_seconds']),
        bounds=dict(minimum=LO,maximum=HI),record=str(OUT/'independent2048.run.json'))
    atomic_write(OUT/'results.json',data,durable=False);publish(data)
    print(json.dumps(data['large']),flush=True)


if __name__=='__main__':
    import sys
    main('--report-only' in sys.argv)
