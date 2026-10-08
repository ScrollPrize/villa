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


def report(root, *, reverse=False):
    baseline=json.loads((DOCS/'normalwidth_194246_tuned_20261007_v1.json').read_text())['Previous EMA999 u12100']
    rows={'EMA999 u12100 tuned (FW/BW off)':baseline}
    baseline_cases=Path(baseline['summary_path']).parent/'cases'
    historical=root.parent/'neural_follow_91_20261006_v2'
    original=root.parent/'neural_follow_91_20261007_v1' if reverse else root
    selections=[('Neural memory u27000',historical/'memory-full'),('Neural Flow v4 u20000',historical/'flow-full'),
                ('Flow v5 u26000',original/'flow_v5-full'),('Unified u35000',original/'unified-full')]
    if reverse:
        selections += [('Flow v5 + FW/BW',root/'flow_v5-full'),('Unified + FW/BW',root/'unified-full')]
        selections += [('Memory matched control',root/'memory-control-full'),
                       ('Flow v4 matched control',root/'flow-control-full'),
                       ('Memory + FW/BW',root/'memory-full'),('Flow v4 + FW/BW',root/'flow-full')]
    for name,folder in selections:
        if not (folder/'summary.json').exists(): continue
        summary=json.loads((folder/'summary.json').read_text())
        if not summary['completed'] or summary['cases']!=182: raise ValueError('incomplete neural run')
        if reverse and name.endswith('+ FW/BW'):
            control_name={'Flow v5 + FW/BW':'Flow v5 u26000','Unified + FW/BW':'Unified u35000',
                'Memory + FW/BW':'Memory matched control','Flow v4 + FW/BW':'Flow v4 matched control'}[name]
            old=rows[control_name]['identity']
            new=summary['neural_follow']
            for key in ('checkpoint_sha256','heading_sha256','backend','sampling_seed',
                        'policy_overrides','trace_grid_base','ct_level','ct_grid_base','precision'):
                if old[key]!=new[key]: raise ValueError('FW/BW control mismatch: '+key)
            for key,value in old['replay'].items():
                if new['replay'][key]!=value: raise ValueError('reference replay policy mismatch: '+key)
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
            reverse_checks=sum(d['replay'].get('reverse_checks',0) for d in docs),
            seconds_to_last_case=progress['seconds'],summary_wall_seconds=summary['wall_seconds'])
    stem='neural_follow_reverse_20261007' if reverse else 'neural_follow_20261007'
    atomic_write(DOCS/(stem+'.json'),rows,durable=False)
    lines=['# Neural followers: Flow v5 and unified scoring','',
        'Same reviewed 91 fibers / 182 directions, native reference acceptance40/80base vx, grace160 and restart grouping80. Confidence terminations are counted separately; all other nonterminal reasons remain errors. GT supplies restart locations and traversal sign, not the inferred frame or history. No FW/BW or width modulation added to neural models.', '',
        'Flow v5 u26000: confidence0.5, n_commit8, gatefull. Unified supplied checkpoint is u35000 (S3 has no u33000): confidence0.7, n_commit8, gatefull. Both CTlevel2/grid4, tracegrid8, maxlen100000 trace voxels, BF16, seed0. Unified maximum ready-forward batch32, Flow16. Upstream853dcf536 with native-reader injection and commit observer. Noise keys use per-trace seed position/decision, not scheduling; BF16 batch-dependent roundoff remains possible. Historical memory u27000 and Flow v4 u20000 are recomputed from saved traces, not rerun; their original policies/revisions remain in the companion JSON.', '',
        'Collection-wide seed dedup16 and CP export spacing800 are CLI collection/export operations, not applied across independent benchmark references. Full min0 results respect min-length0; min512 is additional output-only analysis to match the tuned baseline. This filtering is GT-assisted because geometric errors already split outputs. Report confidence stops and coverage, not errors alone.', '',
        'Neural confidence restarts use the existing native GT restart advancement; the tuned baseline uses its presence-aligned restart policy. This is a comparison of complete policies, not an isolated model substitution.', '',
        'Mean = retained GT distance / (all surviving individual stops +1). Piece mean = retained GT distance / kept pieces; median is retained GT span per piece. P05/P95 and complete machine-readable metrics are in companion JSON.', '']
    def table(source=None):
        out=['| Model | Min vx | Errors | Confidence stops | Presence stops | FW/BW stops | Pieces | Retained % | Mean mm | Piece mean mm | Median mm |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label,row in rows.items():
            for minimum,metric in row['metrics'].items():
                c=metric['combined'] if source is None else metric['scrolls'][source]
                lengths=['—' if c[k] is None else f'{c[k]:.3f}' for k in ('mean_mm','piece_mean_mm','median_mm')]
                out.append(f'| {label} | {minimum} | {c["ordinary_restarts"]} | {c.get("confidence_filtered",0)} | {c["presence_filtered"]} | {c["reverse_filtered"]} | {c["kept_segments"]} | {c["retained_percent"]:.3f} | '+' | '.join(lengths)+' |')
        return out
    lines+=table()
    for source in baseline['metrics']['0']['scrolls']:
        lines+=['','## '+source,'']+table(source)
    lines+=['','## Runtime','', '| Model | Seconds to last completed case |','|---|---:|']
    lines += [f'| {k} | {v["seconds_to_last_case"]:.2f} |' for k,v in rows.items() if 'seconds_to_last_case' in v]
    lines+=['','Times include worker/model initialization and CT reads, exclude manager wait, reference/hash preflight and shutdown. Progress timestamps preserve original measurements even if cached summary analysis is rerun.','',
        '## Reproduce','', 'Set `scripts/training_failure_mining_control.json` to `{"action":"neural_follow","pilot":false}` then run:', '', '```bash','bash /home/hendrik/villa3/scripts/run_training_failure_mining','```','',f'Artifacts/configs/checkpoint hashes: `{root}`. GPUs0–5 via manager. Native RelWithDebInfo build. One run per model; timing depends on CT cache state, not an isolated speed comparison.']
    if reverse:
        lines[0]='# Neural followers: full-prefix FW/BW consistency'
        lines[2]=lines[2].replace('No FW/BW or width modulation added to neural models.',
            'FW/BW rows enable full-prefix checks every512 base voxels plus all terminal endpoints, ellipse20/40, no reverse history. Controls are unfiltered runs. Neural models have no width modulation. The regular EMA999 u12100 control keeps cosine width40% modulation, post-modulated presence cutoff0.4, step6 trace voxels (12base), beam8/lookahead2, with FW/BW disabled.')
        lines[2]=lines[2].replace('Confidence terminations are counted separately; all other nonterminal reasons remain errors.',
            'Confidence and reverse-consistency stops are counted separately from ordinary errors.')
        lines[8]+=' Neural FW/BW filtered stops also restart on the GT arc, not a dense-presence maximum. Reverse confidence exhaustion before reaching the prefix origin fails verification; terminal confidence may therefore become a consistency stop. Reverse seeding uses the actual forward endpoint and tangent only for traversal sign, with a fresh CT heading/frame. No reference path is used in reverse verification.'
        lines=[line.replace('{"action":"neural_follow","pilot":false}',
            '{"action":"neural_follow","pilot":false,"reverse":true}') for line in lines]
        lines += ['', '## Validation', '',
            '149 native regression cases and38 Python tests passed through the same approved runner (control: action neural_follow, tests true). Native verifier extraction preserves existing history0/history4 and frozen-width behavior. Callback tests cover disabled parity, correct/wrong-way/lateral/duplicate/confidence reverse paths, interval/terminal checks, rollback, reseed advancement and expired callbacks. Two-direction GPU pilots validate nested inference before full runs; configuration tests preserve historical policies and batching.', '',
            'This is an online filter: rejected prefixes change subsequent restart locations and later traces, so confidence counts need not remain constant. Min512 analysis discards saved pieces only; no tracing changes. Older memory/Flowv4 results are historical unfiltered controls, not new FW/BW trials. No default promotion.', '',
            '| FW/BW model | Reverse checks |', '|---|---:|']
        lines += [f'| {name} | {row["reverse_checks"]} |' for name,row in rows.items() if name.endswith('+ FW/BW')]
        if 'Memory matched control' in rows or 'Flow v4 matched control' in rows:
            lines=[line.replace('Older memory/Flowv4 results are historical unfiltered controls, not new FW/BW trials.',
                'Historical memory/Flowv4 rows are retained; matched-control and +FW/BW rows are new full evaluations.') for line in lines]
            lines += ['', '## Memory and Flow v4 follow-up', '',
                'The current upstream loader rejects the old memory/identity architecture. These two models therefore use the frozen working runtime from ba86eea12231ff770324d2adb7fceb709e93a3ea (upstream9200d51a9), checked against saved implementation hashes. The shared current native replay/filter and GPU queue are reused; models/rollout are not reimplemented or simplified.', '',
                'Original settings retained: memory confidence.4/full gate/commit8 with3refinement steps; Flow v4 checkpoint-recorded confidence.5/prefix gate/commit16 with0refinement steps. BF16, batch8,16case workers/GPU,24input/I/O workers. Original forward length bound is3×remaining reference length. Fresh matched unfiltered controls distinguish FW/BW changes from integration/batching variation. Our own tuned EMA999u12100 row is explicitly WITHOUT FW/BW.', '',
                'To reproduce old-model runs use models ["memory","flow"], pilot false, reverse false for matched controls or true for FW/BW, through the same approved runner. Run operation inspect_older first to materialize and verify the frozen runtime.']
    (DOCS/(stem+'.md')).write_text('\n'.join(lines)+'\n')
    for name,row in rows.items(): print('RESULT',name,json.dumps(row['metrics']['512']['combined']),flush=True)
