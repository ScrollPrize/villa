"""Current EMA999: frozen shared inference, matched width/FW-BW experiments."""
import json
from pathlib import Path

from reviewed_normal_width_eval import (ROOT, REPO, RUNS, PREFIX, WIDTH_NAME,
    freeze_selection, evaluate_config, measured_result, without_reverse_width_config, width_metric_table)
from lasagna.inference_provenance import atomic_write
from vesuvius.neural_tracing.fiber_trace_3d.beam_parameter_sweep import apply_trace_args

NAME='normalwidth_122417_current_ema999_20261006_v1'


def validate_ema(saved):
    if saved.get('snapshot_kind')!='ema' or saved.get('ema_metadata',{}).get('decay')!=.999:
        raise ValueError('expected EMA0.999 snapshot')


def configs(snapshot,digest):
    base=json.loads((ROOT/WIDTH_NAME/'workload.json').read_text())
    base.update(checkpoint=str(snapshot),checkpoint_sha256=digest)
    base['evaluation']['inference_cache_namespace']=NAME
    base['evaluation']['inference']['emit_width']=True
    base['evaluation']['trace_args']=apply_trace_args(base['evaluation']['trace_args'],{
        'width-relative-tolerance':.4,'width-weighting':'cosine'})
    result={}
    for label in ('cosine40_no_reverse','cosine40_fw512','off_no_reverse'):
        name=NAME+'_'+label
        if label=='cosine40_fw512':
            cfg=json.loads(json.dumps(base))
            cfg['output_root']=str(ROOT/name)
            cfg['evaluation']['gpu_manager']['label']=name
        else:
            cfg=without_reverse_width_config(base,name,cosine=label.startswith('cosine'),tolerance=.4)
        result[label]=cfg
    return result


def document(results,metadata):
    target=REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks'/NAME
    atomic_write(target.with_suffix('.json'),dict(snapshot=metadata,results=results),durable=False)
    lines=['# Current EMA999 — cosine40 width and FW/BW','',
        'Same 91 reviewed fibers / 182 directions, beam8/lookahead2. Forward ellipse40/80, grace160. '
        'FW/BW uses full prefix every512 and endpoints, ellipse20/40, frozen forward width mean. '
        'Cosine40 multiplies presence by cos(pi*d/2) for d<1, otherwise0; d=abs(width/mean-1)/0.4. '
        'Cumulative committed arithmetic mean; reset on restart. No numerical/parallelism changes; GPUs0–5.', '',
        'Errors are80-base-voxel grouped episodes; filtered stops ungrouped. Mean=retained GT mm/(all actual stops+1). '
        'Median/P05/P95 include terminal and zero-length retained spans. GT controls failures/reseeding; terminal reverse conversions are oracle-timed.', '',
        'Snapshot: `'+json.dumps(metadata)+'`.', '']
    lines+=width_metric_table({k:v['combined'] for k,v in results.items()})
    if results:
        for scroll in next(iter(results.values()))['scrolls']:
            lines+=['',f'## {scroll}','']+width_metric_table({k:v['scrolls'][scroll] for k,v in results.items()})
    docs=target.parent
    old_no=json.loads((docs/'reviewed_u11000_width_no_reverse_20261006.json').read_text())
    old_fw=json.loads((docs/'reviewed_u11000_width_sweep_20261006.json').read_text())
    comparisons={}
    for label in results:
        old=old_fw['cosine40'] if label=='cosine40_fw512' else old_no['off' if label=='off_no_reverse' else 'cosine40']
        comparisons['raw u11000 / '+label]=old
        comparisons['EMA999 / '+label]=results[label]
    lines+=['','## Matched previous comparisons','']+width_metric_table({k:v['combined'] for k,v in comparisons.items()})
    lines+=['','## Timing','','| Variant | Wall s | Additional inference tiles |','|---|---:|---:|']
    lines += [f'| {k} | {v["wall_seconds"]:.2f} | {v.get("inferred_tiles", "pending")} |' for k,v in results.items()]
    lines+=['','## Reproduce','','Control action `reviewed_ema999_width`; same frozen selection reused on resume.', '',
        '```bash','bash /home/hendrik/villa3/scripts/run_training_failure_mining','```','',
        'Artifacts under `'+str(ROOT/NAME)+'*`; timings include startup, resource waits and missing-cache inference.']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main(control):
    import torch
    source=RUNS/(PREFIX+'122417')/'snapshots/current_ema_0p999.pt'
    snapshot,digest=freeze_selection(NAME,source)
    saved=torch.load(snapshot,map_location='cpu',weights_only=False,mmap=True)
    validate_ema(saved)
    metadata=dict(source=str(source),sha256=digest,path=str(snapshot),
        **{k:saved.get(k) for k in ('step','samples_seen','snapshot_kind','ema_metadata','metric_name','metric')})
    del saved
    atomic_write(ROOT/NAME/'snapshot_metadata.json',metadata,durable=False)
    print('FROZEN EMA999 '+json.dumps(metadata),flush=True)
    if control.get('freeze_only'): return
    results={}
    for label,cfg in configs(snapshot,digest).items():
        name=Path(cfg['output_root']).name
        if not control.get('report_only'):
            evaluate_config(cfg,name)
        folder=ROOT/name
        effective=json.loads((folder/'full.json').read_text())
        previous=json.loads((ROOT/WIDTH_NAME/'full.json').read_text())
        if effective['sources']!=previous['sources'] or effective['trace_args']!=cfg['evaluation']['trace_args']:
            raise ValueError('effective cohort/trace configuration mismatch')
        result=measured_result(folder)
        if label!='cosine40_fw512' and (result['combined']['reverse_checks'] or result['combined']['filtered_stops']):
            raise ValueError('unexpected reverse checks')
        result['wall_seconds']=json.loads((folder/'timing.json').read_text())['wall_seconds']
        inv=json.loads((folder/'full/inference_summary.json').read_text())
        result['inferred_tiles']=sum(s['inferred_tiles'] for s in inv['sources'].values())
        results[label]=result
        document(results,metadata)
        print('COMPLETED '+label+' '+json.dumps(result['combined']),flush=True)
    if control.get('report_only'): return
    import reviewed_width_paths as analysis
    analysis.BASE=ROOT/(NAME+'_off_no_reverse')
    analysis.OUT=ROOT/(NAME+'_width_paths')
    analysis.SHA=digest
    analysis.REPORT_NAME=NAME+'_width_paths'
    analysis.MODEL_LABEL=f'EMA999 u{metadata["step"]}'
    analysis.main()


if __name__=='__main__':
    main(json.loads((REPO/'scripts/training_failure_mining_control.json').read_text()))
