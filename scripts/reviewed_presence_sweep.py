"""Frozen two-model raw-presence/width/reverse-filter factorial benchmark."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from reviewed_normal_width_eval import (ROOT, REPO, WIDTH_NAME, evaluate_config,
    measured_result, without_reverse_width_config)
from reviewed_ema999_width_eval import NAME as EMA_NAME
from lasagna.inference_provenance import atomic_write, sha256_file
from vesuvius.neural_tracing.fiber_trace_3d.beam_parameter_sweep import apply_trace_args

NAME='reviewed_presence_width_reverse_matrix_20261006_v1'
POST_NAME='reviewed_postmod_presence_ema999_20261006_v1'
THRESHOLDS=(0.,.10,.25,.50)
POST_THRESHOLDS=(.5,.4,.3,.6,.7,.1,.2)
MODELS={
    'raw11000':(ROOT/WIDTH_NAME,'2408ece6ffa2251655709cf3987357f1892a573828929934266145f09f9d2eba'),
    'ema999_12100':(ROOT/(EMA_NAME+'_cosine40_fw512'),'fd9dc4bac7a04bbdbf98d773629b264b902c9c25228482a9adbdbc08e78ef05a'),
}


def variants(base,model,post=False):
    for cutoff in (POST_THRESHOLDS if post else THRESHOLDS):
        for width in (False,True):
            for reverse in (False,True):
                label=f'{model}_p{round(cutoff*100):02d}_{"cos40" if width else "off"}_{"fw512" if reverse else "norev"}'
                cfg=json.loads(json.dumps(base))
                if not reverse: cfg=without_reverse_width_config(cfg,label,cosine=width,tolerance=.4)
                cfg['output_root']=str(ROOT/(POST_NAME if post else NAME)/label)
                cfg['evaluation']['gpu_manager']['label']=label
                cfg['evaluation']['trace_args']=apply_trace_args(cfg['evaluation']['trace_args'],{
                    'width-relative-tolerance':.4 if width else 0,
                    'width-weighting':'cosine' if width else 'linear','replay-presence-cutoff':cutoff})
                if post:
                    cfg['evaluation']['trace_args']=apply_trace_args(cfg['evaluation']['trace_args'],{'replay-presence-mode':'modulated'})
                yield label,cfg,dict(model=model,cutoff=cutoff,width=width,reverse=reverse,mode='modulated' if post else 'raw')


def filter_counts(folder):
    def read(path):
        doc=json.loads(path.read_text()); source=doc['provenance']['source_id']
        segments=doc['replay']['segments']
        p=sum(s['termination_reason']=='presence_filtered' for s in segments)
        r=sum(s['termination_reason']=='reverse_consistency_filtered' for s in segments)
        if p+r!=len(doc['replay']['filtered_stops']): raise ValueError('unaccounted filtered reason')
        return source,p,r
    counts={}
    with ThreadPoolExecutor(16) as pool:
        for source,p,r in pool.map(read,(folder/'full/cases').glob('*/trace.json')):
            row=counts.setdefault(source,[0,0]);row[0]+=p;row[1]+=r
    return counts


def table(rows,scroll=None):
    lines=['| Model | Width | FW/BW | Cutoff | Errors | Presence stops | Reverse stops | Actual stops | Kept % | Mean mm | Median mm | P05 mm | P95 mm |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows.values():
        c=r['combined'] if scroll is None else r['scrolls'][scroll]
        lines.append(f'| {r["model"]} | {"cos40" if r["width"] else "off"} | {"512" if r["reverse"] else "off"} | {r["cutoff"]:.2f} | {c["ordinary_restarts"]} | {c["presence_filtered"]} | {c["reverse_filtered"]} | {c["actual_stops"]} | {c["retained_percent"]:.3f} | {c["mean_mm"]:.3f} | {c["median_mm"]:.3f} | {c["p05_mm"]:.3f} | {c["p95_mm"]:.3f} |')
    return lines


def document(rows,post=False):
    name=POST_NAME if post else NAME
    target=REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks'/name
    atomic_write(target.with_suffix('.json'),rows,durable=False)
    count=len(rows)
    if post:
        old=json.loads((ROOT/NAME/'results.json').read_text())
        compared={}
        for k,r in rows.items():
            if k in old:
                compared[k+'_raw']={**old[k],'model':r['model']+'/raw'}
            compared[k+'_modulated']={**r,'model':r['model']+'/modulated'}
        rows=compared
    lines=['# Presence cutoff × width × FW/BW — frozen two-model matrix','',
        f'Completed {len(rows)}/32 configurations. Same reviewed91fibers/182directions, beam8/lookahead2, forward40/80 ellipse and160 grace. '
        'Raw u11000 and pinned current EMA999 u12100 from122417. Same per-model width-enabled prediction cache; BF16/float16 accumulation, GPU0–5 manager, scheduling and worker settings unchanged.', '',
        'Cutoff checks raw (not width/direction-modulated) predicted presence at each newly committed forward point. '
        'Maximum finite presence among valid direction options; invalid/absent options count0; equality passes. '
        'Zero disables the feature exactly. GT-provided initial seed itself is not filtered; pre-commit invalid-start failures remain ordinary failures. '
        'Below cutoff filters the point and stops before it; no presence grace. With FW/BW, rollback to last reverse-verified prefix; otherwise keep all previous accepted points. '
        'Restart uses existing GT-guided10/20 ellipse/local maximum/aligned direction search, additionally requiring presence>=cutoff. '
        'FW/BW full-prefix512+endpoints uses20/40 ellipse and frozen forward-width mean. Width multiplier is cosine40 when enabled.', '',
        'Errors are80-vx grouped geometric episodes; filtered reason counts are ungrouped. '
        'Mean=retainedGTmm/(all individual error+filtered stops+1). Median/quantiles are retained segment spans including terminal/zero spans. '
        'Retained distance is coverage, not accuracy. Filtering can lower errors by dropping more data. '
        'GT still supplies failure/reseed locations; this is not a GT-free deployment score.', '']+table(rows)
    if post:
        lines=['# Post-modulation presence cutoff — EMA999 u12100','',
            f'Completed {count}/{4*len(POST_THRESHOLDS)} post-modulation configurations:cutoffs{sorted(POST_THRESHOLDS)}. Completed runs reused; only missing configurations executed. Available matched raw controls reused from prior matrix. Same reviewed91fibers/182directions, frozen checkpoint/cache, GPU0–5 and unchanged scheduling/precision.', '',
            'Accepts scored presence evidence >=cutoff. Modulated evidence is 1 - fiberLocalAlignmentLoss: raw presence × width scale × six clamped positive dot products between previous/candidate step and prediction directions. Captured from the winning candidate option at scoring time and carried with the committed path; separate smoothness/embedding penalties excluded. Width-off still includes direction modulation. Cosine40 uses the exact scored cumulative-width reference, not a later updated mean.', '',
            'Same committed-point stop/rollback as raw mode: FW retains last verified prefix; no FW retains previous accepted points. Initial seeds unchecked. Reseeding uses raw cutoff because a new segment resets width/history. Errors grouped80vx; filtered reasons ungrouped. Mean retainedGTmm/(individual stops+1), median/quantiles include terminal/zero spans. Retention is not accuracy; GT-aided reseeding remains a limitation. Defaults unchanged.', '']+table(rows)
    if rows:
        for scroll in next(iter(rows.values()))['scrolls']:
            lines+=['',f'## {scroll}','']+table(rows,scroll)
    lines+=['','## Timing/provenance','','| Variant | Wall s | Additional tiles | SHA256 |','|---|---:|---:|---|']
    lines += [f'| {k} | {r["wall_seconds"]:.2f} | {r["inferred_tiles"]} | {r["checkpoint_sha256"]} |' for k,r in rows.items()]
    lines+=['','Wall times include startup/manager waits/cache fill and are order dependent, not isolated trace speed.', '',
        'Control `{"action":"'+('reviewed_presence_postmod' if post else 'reviewed_presence_sweep')+'"}`; resume skips completed immutable runs.', '',
        '```bash','bash /home/hendrik/villa3/scripts/run_training_failure_mining','```','',f'Artifacts: `{ROOT/name}`.']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main(post=False):
    rows={}
    for model,(folder,digest) in MODELS.items():
        if post and model!='ema999_12100': continue
        base=json.loads((folder/'workload.json').read_text())
        if base['checkpoint_sha256']!=digest or sha256_file(base['checkpoint'])!=digest: raise ValueError('wrong frozen model')
        for label,cfg,params in variants(base,model,post):
            evaluate_config(cfg,label)
            output=Path(cfg['output_root'])
            effective=json.loads((output/'full.json').read_text())
            if effective['sources']!=json.loads((folder/'full.json').read_text())['sources'] or effective['trace_args']!=cfg['evaluation']['trace_args']:
                raise ValueError('effective inputs mismatch')
            r=measured_result(output)
            counts=filter_counts(output)
            for source,(p,b) in counts.items(): r['scrolls'][source].update(presence_filtered=p,reverse_filtered=b)
            r['combined'].update(presence_filtered=sum(v[0] for v in counts.values()),reverse_filtered=sum(v[1] for v in counts.values()))
            if r['combined']['filtered_stops']!=r['combined']['presence_filtered']+r['combined']['reverse_filtered']: raise ValueError('filter count mismatch')
            r.update(params,checkpoint_sha256=digest,native_sha256=sha256_file(cfg['native_binary']),
                wall_seconds=json.loads((output/'timing.json').read_text())['wall_seconds'],
                inferred_tiles=sum(v['inferred_tiles'] for v in json.loads((output/'full/inference_summary.json').read_text())['sources'].values()))
            rows[label]=r
            atomic_write(ROOT/(POST_NAME if post else NAME)/'results.json',rows,durable=False)
            document(rows,post)
            print(f'MATRIX {len(rows)}/{4*len(POST_THRESHOLDS) if post else 32} {label} '+json.dumps(r['combined']),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--post-modulated',action='store_true')
    main(parser.parse_args().post_modulated)
