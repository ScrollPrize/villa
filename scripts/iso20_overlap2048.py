"""Apply promoted presence/f0 baseline to validated cached2048 tracing."""
import json
from pathlib import Path
import iso20_best2048 as context
import iso20_post_overlap as overlap
import iso20_iterative as solver
from lasagna.inference_provenance import atomic_write, sha256_file

shared=overlap.shared
SOURCE=context.OUT
WIN=SOURCE.parent/'20261003_iso20_overlap_costs_v1'
OUT=SOURCE.parent/'20261004_iso20_overlap2048_v1'
REPORT=shared.REPO/'volume-cartographer/docs/fiber_benchmark_runs/2026-10-04-overlap2048'
VARIANT='presence-10x20-f0'


def invariant_trace_args(args):
    result=[];i=0
    while i<len(args):
        if args[i]=='--bbox':i+=7
        elif args[i]=='--output':i+=2
        else:result.append(args[i]);i+=1
    return result


def validate_source(large_config,small_config,record):
    large=large_config['commands'][0]['args'];small=small_config['commands'][0]['args']
    assert invariant_trace_args(large)==invariant_trace_args(small),'2048 tracing settings differ'
    i=large.index('--bbox')
    assert list(map(int,large[i+1:i+7]))==[*context.LO,*context.HI]
    assert record['returncode']==0 and record['command'][1:]==large
    for path,digest in record['inputs'].items():
        assert sha256_file(path)==digest,'cached trace input changed: '+path


def publish(data):
    atomic_write(OUT/'results.json',data,durable=False)
    context.OUT=OUT;context.REPORT=REPORT
    context.publish(data)


def main(report_only=False):
    OUT.mkdir(parents=True,exist_ok=True)
    if report_only:publish(json.loads((OUT/'results.json').read_text()));return
    shared.OUT=OUT;overlap.OUT=OUT;overlap.LOW=True
    solver.OUT=OUT;solver.isolation.OUT=OUT
    state=json.loads((WIN/'state.json').read_text())
    assert state['stage']=='complete'
    winner=state['variants'][VARIANT]['winner'];small=state['rows'][winner]
    assert small['params']==dict(boundary=10,disable=100,far=.2,near=.2,p0=0,p1=.0375,sign='hard')
    template=json.loads((SOURCE/'beam2048.config.json').read_text())
    trace_record=json.loads((SOURCE/'beam2048.run.json').read_text())
    validate_source(template,json.loads(context.TRACE_TEMPLATE.read_text()),trace_record)
    trace=SOURCE/'beam2048.zarr'
    source_attrs=json.loads((trace/'.zattrs').read_text())['processing']['artifact']
    print('Validated cached2048 trace: '+str(source_attrs['trace_count'])+' pieces',flush=True)
    config=overlap.filter_config(template,'presence2048','presence',10,20,0,source_trace=trace)
    record=shared.execute(config)
    for key in ('binary_sha256','shared_libraries','references'):
        assert record[key]==small['provenance'][key],'1024/native provenance mismatch: '+key
    filtered=OUT/'presence2048.zarr'
    attrs=json.loads((filtered/'.zattrs').read_text())['processing']['artifact']
    print('Filtered2048: '+str(attrs['trace_count'])+' pieces',flush=True)
    bp_template=context.bp_config(json.loads((WIN/f'{winner}-ind.config.json').read_text()),filtered)
    r=solver.run_stage('bp2048',{},'ind',bp_template,[],parameter_updates={})
    if 'independent' in r:
        r.update(solver.run_stage('bp2048',{},'oracle',bp_template,[],r['provenance'],parameter_updates={}))
    data=dict(small=small['independent'],large=r.get('independent',{}),
        oracle={'1024':small['oracle'],'2048':r.get('oracle',{})},winner=winner,params=small['params'],
        source_trace_record=trace_record,source_trace_attrs_sha256=sha256_file(trace/'.zattrs'),
        source_trace_count=source_attrs['trace_count'],filtered_trace_count=attrs['trace_count'],
        filter_summary=attrs['trace_summary'],bp_result=r,
        bounds=dict(minimum=context.LO,maximum=context.HI),action='iso20_overlap2048',
        timings=dict(historical_trace=trace_record['wall_seconds'],filter=record['wall_seconds'],
            independent=r.get('independent',{}).get('seconds'),oracle=r.get('oracle',{}).get('seconds')),
        protocol='Promoted presence-ranked ellipse10/20,f0 whole-piece deduplication;45degree gate,<=4base samples. '
            'Same raw40500 beam8/iso20/full-prefix512 source traces. BP near/far.2/.2,P1.0375,others0, '
            'hard signs,disable100/128,boundary10,min256,median64/blend1,scale.822/phase.5,500messages. '
            'Independent readout and separate reference-conditioned removal; no retuning.',
        execution_note='Historical2048 tracing reused as frozen input after exact configuration comparison '
            '(only bbox/output differ) and recorded manifest hash checks. Its binary is recorded separately; '
            'not claimed regenerated with current binary. Native filter and BP match1024tuning binary/libraries/references. '
            'No new inference or tracing; one measurement, RelWithDebInfo,128threads.',
        extra_markdown=['','## Geometry and diagnostics','',
            f"2048 source pieces: {source_attrs['trace_count']}; retained: {attrs['trace_count']}; "
            f"BP pieces: {r.get('independent',{}).get('pieces','unavailable')}.",
            '', '2048 observed-shift denominator: '+str(r.get('independent',{}).get('observed_ladder_shift',{}).get('references','unavailable'))+'.',
            '', 'Errors/readout diagnostics: '+json.dumps({k:v for k,v in r.items() if 'error' in k})+'.',
            '', 'Expanded crop overlaps the tuning crop; not an independent held-out validation. '
            'All original solver resource limits retained. Removal =−100×problematic/fulfilled (higher better).'])
    publish(data);print(json.dumps(dict(large=data['large'],oracle=data['oracle']['2048'])),flush=True)


if __name__=='__main__':
    import sys
    main('--report-only' in sys.argv)
