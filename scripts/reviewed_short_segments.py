"""Cached, output-only length filtering; no inference/tracer execution."""
import json
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from reviewed_presence_sweep import ROOT, REPO, NAME, POST_NAME, POST_THRESHOLDS
from lasagna.inference_provenance import atomic_write
from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import filter_output_segments, filtered_output_statistics

OUT='reviewed_ema999_short512_20261006_v1'
DOCS=REPO/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks'


def inputs():
    post=json.loads((DOCS/(POST_NAME+'.json')).read_text())
    raw=json.loads((DOCS/(NAME+'.json')).read_text())
    result={k+'_post':v for k,v in post.items()}
    result.update({k+'_raw':v for k,v in raw.items() if v['model']=='ema999_12100' and
        (v['cutoff']==.5 or (v['cutoff']==0 and not v['width']))})
    if len(result)!=4*len(POST_THRESHOLDS)+6: raise ValueError('incomplete configurations')
    return result


def analyze(row):
    folder=Path(row['summary_path']).parent.parent
    paths=sorted((folder/'full/cases').glob('*/trace.json'))
    if len(paths)!=182: raise ValueError('missing cases')
    def read(path):
        doc=json.loads(path.read_text())
        return [dict(filter_output_segments(doc,minimum), trace_path=str(path)) for minimum in (0,512)]
    with ThreadPoolExecutor(16) as pool:
        data=list(pool.map(read,paths))
    results={}
    for i,minimum in enumerate((0,512)):
        cases=[pair[i] for pair in data]
        scrolls={s:filtered_output_statistics([c for c in cases if c['source']==s]) for s in row['scrolls']}
        results[str(minimum)]=dict(combined=filtered_output_statistics(cases),scrolls=scrolls)
    for source in ['combined',*row['scrolls']]:
        old=row['combined'] if source=='combined' else row['scrolls'][source]
        before=results['0']['combined'] if source=='combined' else results['0']['scrolls'][source]
        after=results['512']['combined'] if source=='combined' else results['512']['scrolls'][source]
        for k in ('ordinary_restarts','actual_stops','filtered_stops','retained_mm','retained_percent','mean_mm','median_mm','p05_mm','p95_mm'):
            if not math.isclose(before[k],old[k],abs_tol=1e-7,rel_tol=1e-8):
                raise ValueError(f'baseline mismatch:{source}/{k}:{before[k]}!={old[k]}')
        if not math.isclose(after['retained_mm']+after['discarded_mm'],before['retained_mm'],abs_tol=1e-7):
            raise ValueError('coverage accounting mismatch')
    return results,[pair[1] for pair in data]


def table(rows,scroll=None):
    out=['| Mode | Cutoff | Width | FW | Min vx | Errors | Pres | Rev | Stops | Surviving pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |',
         '|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def num(x): return '—' if x is None else f'{x:.3f}'
    for row in rows.values():
        for minimum,r in row['metrics'].items():
            c=r['combined'] if scroll is None else r['scrolls'][scroll]
            out.append(f'| {row["mode"]} | {row["cutoff"]} | {"cos40" if row["width"] else "off"} | {"on" if row["reverse"] else "off"} | {minimum} | {c["ordinary_restarts"]} | {c["presence_filtered"]} | {c["reverse_filtered"]} | {c["actual_stops"]} | {c["kept_segments"]} | {c["retained_percent"]:.3f} | {num(c["mean_mm"])} | {num(c["piece_mean_mm"])} | {num(c["median_mm"])} | {num(c["p05_mm"])} | {num(c["p95_mm"])} |')
    return out


def document(rows):
    target=DOCS/OUT
    atomic_write(target.with_suffix('.json'),rows,durable=False)
    lines=['# Minimum512base-voxel output-piece filter — EMA999u12100','',
        f'Completed {len(rows)}/{4*len(POST_THRESHOLDS)+6} configurations,91fibers/182directions each. Cached analysis only; no new inference or tracing.', '',
        '**This is GT-assisted retained-output analysis, not a new online tracing result.** Existing benchmark GT errors already terminate pieces. Discarding those pieces can remove their errors; it does not demonstrate a GT-free detection mechanism. Original unfiltered online errors are in min0 rows; source traces remain untouched.', '',
        'Drop saved output pieces with traced-polyline length<512base voxels, after FW rollback; equality kept. Selection never uses GT length or correctness. Initial/final pieces also filtered. Coverage/length statistics retain prior GT-span accounting. Pres/Rev/Stops in min512 rows count only surviving piece endpoints; all original stops and discarded-error counts remain in JSON. Surviving piece counts are shown; discarded pieces and lost coverage remain explicit in JSON. No new starts or merging across gaps.', '',
        'Mean=retainedGTmm/(surviving individual error+filtered stops+1), the existing virtual-concatenation convention. Piece mean=retainedGTmm/number of surviving pieces; median/P05/P95 use surviving GT spans.512trace-vx need not mean512GT-vx. Errors regrouped within each case using existing80base-vx rule. Empty output reports null lengths,0coverage, not infinite quality. All min0 rows audited against original metrics globally/per-scroll.', '',
        f'{4*len(POST_THRESHOLDS)}post-modulation variants,4raw.5controls,2no-cutoff/no-width references. Same frozen checkpoint, cache and native traces. Defaults unchanged.', '']+table(rows)
    if rows:
        for source in next(iter(rows.values()))['metrics']['0']['scrolls']:
            lines+=['',f'## {source}','']+table(rows,source)
    lines+=['','## Reproduce','',
        'Set scripts/training_failure_mining_control.json action to reviewed_short_segments; then:', '',
        '```bash','bash /home/hendrik/villa3/scripts/run_training_failure_mining','```','',
        f'Artifacts and per-case/segment decisions: `{ROOT/OUT}`. Source summary paths and checkpoint hashes are in JSON.']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main():
    rows={}
    for name,row in inputs().items():
        metrics,cases=analyze(row)
        rows[name]={k:row[k] for k in ('model','cutoff','width','reverse','summary_path','checkpoint_sha256')}
        rows[name].update(mode=row.get('mode','raw'),metrics=metrics)
        atomic_write(ROOT/OUT/(name+'_decisions.json'),cases,durable=False)
        atomic_write(ROOT/OUT/'results.json',rows,durable=False)
        document(rows)
        print(f'SHORT {len(rows)}/{4*len(POST_THRESHOLDS)+6} {name} '+json.dumps(metrics['512']['combined']),flush=True)


if __name__=='__main__': main()
