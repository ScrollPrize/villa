"""Compare native neural replay artifacts using shared retained-piece metrics."""
import json
import math
import numpy as np
from collections import Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from lasagna.inference_provenance import atomic_write
from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import filter_output_segments, filtered_output_statistics

DOCS=Path(__file__).resolve().parents[1]/'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks'


def report(root):
    baseline=json.loads((DOCS/'normalwidth_194246_tuned_20261007_v1.json').read_text())['Previous EMA999 u12100']
    rows={'EMA999 u12100 tuned':baseline}
    baseline_cases=Path(baseline['summary_path']).parent/'cases'
    historical=root.parent/'neural_follow_91_20261006_v2'
    selections=[('Neural memory u27000',historical/'memory-full'),('Neural Flow v4 u20000',historical/'flow-full'),
                ('Flow v5 u26000',root/'flow_v5-full'),('Unified u35000',root/'unified-full')]
    for name,folder in selections:
        if not (folder/'summary.json').exists(): continue
        summary=json.loads((folder/'summary.json').read_text())
        if not summary['completed'] or summary['cases']!=182: raise ValueError('incomplete neural run')
        paths=sorted((folder/'cases').glob('*/trace.json'))
        if len(paths)!=182: raise ValueError('missing artifacts')
        with ThreadPoolExecutor(16) as pool:
            docs=list(pool.map(lambda p:json.loads(p.read_text()),paths))
        scales={s['source_id']:s['summary']['base_voxel_size_um'] for s in summary['sources']}
        for doc in docs:
            doc['provenance'].setdefault('base_voxel_size_um',scales[doc['provenance']['source_id']])
            prior=json.loads((baseline_cases/str(doc['provenance']['case_index'])/'trace.json').read_text())
            for key in ('source_id','fiber_path','reverse'):
                if prior['provenance'][key]!=doc['provenance'][key]: raise ValueError('reference identity mismatch')
            if not np.array_equal(prior['reference_points_base_xyz'],doc['reference_points_base_xyz']):
                raise ValueError('reference geometry mismatch')
        reasons=Counter(s['termination_reason'] for d in docs for s in d['replay']['segments'])
        metrics={}
        for minimum in (0,512):
            cases=[filter_output_segments(d,minimum,confidence_reasons=('confidence',)) for d in docs]
            combined=filtered_output_statistics(cases)
            if abs(combined['reference_mm']-baseline['metrics'][str(minimum)]['combined']['reference_mm'])>1e-7:
                raise ValueError('reference distance mismatch')
            metrics[str(minimum)]=dict(combined=combined,scrolls={s:filtered_output_statistics([c for c in cases if c['source']==s]) for s in sorted({c['source'] for c in cases})})
        if not math.isclose(metrics['512']['combined']['retained_mm']+metrics['512']['combined']['discarded_mm'],metrics['0']['combined']['retained_mm'],abs_tol=1e-7):
            raise ValueError('retained/discarded accounting mismatch')
        progress=json.loads((folder/'progress.json').read_text())
        rows[name]=dict(metrics=metrics,reasons=dict(reasons),summary_path=str(folder/'summary.json'),identity=summary['neural_follow'],
            seconds_to_last_case=progress['seconds'],summary_wall_seconds=summary['wall_seconds'])
    atomic_write(DOCS/'neural_follow_20261007.json',rows,durable=False)
    lines=['# Neural followers: Flow v5 and unified scoring','',
        'Same reviewed 91 fibers / 182 directions, native reference acceptance40/80base vx, grace160 and restart grouping80. Confidence terminations are counted separately; all other nonterminal reasons remain errors. GT supplies restart locations and traversal sign, not the inferred frame or history. No FW/BW or width modulation added to neural models.', '',
        'Flow v5 u26000: confidence0.5, n_commit8, gatefull. Unified supplied checkpoint is u35000 (S3 has no u33000): confidence0.7, n_commit8, gatefull. Both CTlevel2/grid4, tracegrid8, maxlen100000 trace voxels, BF16, seed0. Unified maximum ready-forward batch32, Flow16. Upstream853dcf536 with native-reader injection and commit observer. Noise keys use per-trace seed position/decision, not scheduling; BF16 batch-dependent roundoff remains possible. Historical memory u27000 and Flow v4 u20000 are recomputed from saved traces, not rerun; their original policies/revisions remain in the companion JSON.', '',
        'Collection-wide seed dedup16 and CP export spacing800 are CLI collection/export operations, not applied across independent benchmark references. Full min0 results respect min-length0; min512 is additional output-only analysis to match the tuned baseline. This filtering is GT-assisted because geometric errors already split outputs. Report confidence stops and coverage, not errors alone.', '',
        'Neural confidence restarts use the existing native GT restart advancement; the tuned baseline uses its presence-aligned restart policy. This is a comparison of complete policies, not an isolated model substitution.', '',
        'Mean = retained GT distance / (all surviving individual stops +1). Piece mean = retained GT distance / kept pieces; median is retained GT span per piece. P05/P95 and complete machine-readable metrics are in companion JSON.', '']
    def table(source=None):
        out=['| Model | Min vx | Errors | Confidence/presence stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label,row in rows.items():
            for minimum,metric in row['metrics'].items():
                c=metric['combined'] if source is None else metric['scrolls'][source]
                lengths=['—' if c[k] is None else f'{c[k]:.3f}' for k in ('mean_mm','piece_mean_mm','median_mm')]
                out.append(f'| {label} | {minimum} | {c["ordinary_restarts"]} | {c["filtered_stops"]} | {c["kept_segments"]} | {c["retained_percent"]:.3f} | '+' | '.join(lengths)+' |')
        return out
    lines+=table()
    for source in baseline['metrics']['0']['scrolls']:
        lines+=['','## '+source,'']+table(source)
    lines+=['','## Runtime','', '| Model | Seconds to last completed case |','|---|---:|']
    lines += [f'| {k} | {v["seconds_to_last_case"]:.2f} |' for k,v in rows.items() if 'seconds_to_last_case' in v]
    lines+=['','Times include worker/model initialization and CT reads, exclude manager wait, reference/hash preflight and shutdown. Progress timestamps preserve original measurements even if cached summary analysis is rerun.','',
        '## Reproduce','', 'Set `scripts/training_failure_mining_control.json` to `{"action":"neural_follow","pilot":false}` then run:', '', '```bash','bash /home/hendrik/villa3/scripts/run_training_failure_mining','```','',f'Artifacts/configs/checkpoint hashes: `{root}`. GPUs0–5 via manager. Native RelWithDebInfo build. One run per model; timing depends on CT cache state, not an isolated speed comparison.']
    (DOCS/'neural_follow_20261007.md').write_text('\n'.join(lines)+'\n')
    for name,row in rows.items(): print('RESULT',name,json.dumps(row['metrics']['512']['combined']),flush=True)
