"""Targeted cached tracing sweeps; reuse evaluation and retained-output metrics."""
import copy
import json
from pathlib import Path

from lasagna.inference_provenance import atomic_write, sha256_file
from reviewed_presence_sweep import ROOT, REPO, filter_counts, MODELS
from reviewed_normal_width_eval import evaluate_config, measured_result
from reviewed_short_segments import analyze, DOCS
from vesuvius.neural_tracing.fiber_trace_3d.beam_parameter_sweep import apply_trace_args

NAME = 'reviewed_ema999_trace_tuning_20261006_v1'
PLAN = REPO/'scripts/reviewed_trace_tuning_plan.json'
CHANGES = {
    'step2': {'step-voxels': 2},
    'step6': {'step-voxels': 6},
    'step5': {'step-voxels': 5},
    'step8': {'step-voxels': 8},
    'tangent5': {'smoothness-tangent-weight': 5},
    'tangent20': {'smoothness-tangent-weight': 20},
    'normal005': {'smoothness-normal-weight': .05},
    'normal02': {'smoothness-normal-weight': .2},
    'cumulative0': {'cumulative-smoothness-tangent-weight': 0},
    'cumulative4': {'cumulative-smoothness-tangent-weight': 4},
}


def variant(base, label, changes):
    unknown = set(changes)-{k for v in CHANGES.values() for k in v}
    if unknown:
        raise ValueError(f'unsupported tuning parameters: {unknown}')
    cfg = copy.deepcopy(base)
    cfg['output_root'] = str(ROOT/NAME/label)
    cfg['evaluation']['gpu_manager']['label'] = label
    cfg['evaluation']['trace_args'] = apply_trace_args(cfg['evaluation']['trace_args'], changes)
    return cfg


def nondominated(rows):
    def values(row):
        c = row['metrics']['512']['combined']
        return c['ordinary_restarts'], c['kept_segments'], -c['retained_percent']
    return [name for name, row in rows.items() if not any(
        all(a <= b for a, b in zip(values(other), values(row))) and
        any(a < b for a, b in zip(values(other), values(row)))
        for other in rows.values())]


def document(rows, name=NAME, title='Targeted tracing tuning — EMA999 u12100', notes=(), action='reviewed_trace_tuning'):
    atomic_write(DOCS/(name+'.json'), rows, durable=False)
    lines = ['# '+title, '', *notes, '',
        ('Same frozen model, reviewed91fibers/182directions and inference cache. ' if name==NAME else 'Each named model is frozen independently; identical reviewed91fibers/182directions. ')+
        'No inference precision or resource changes. Beam8/lookahead2 and FW protocol fixed; changed parameters are explicit below. Defaults unchanged. The tracing policy was tuned on this benchmark, not independently validated; single-error differences may not generalize.', '',
        'Step is in trace voxels; trace-to-base scale is2. Baseline4 means8base voxels. Step2/5/6/8 means4/10/12/16base voxels, nominal two-step lookahead8/20/24/32base voxels (baseline16). Changing step also changes the physical smoothing scale of cumulative history (history parameter4, normalized recursive mean, not a hard four-step window) and spacing of arithmetic-width-mean samples; these are not isolated weight-only experiments. Reverse-check settings are recorded in trace_args; disabled checks remain disabled.', '',
        'Min512 is output-only filtering of actual traced-polyline lengths after rollback. Existing GT failures split pieces: this is GT-assisted retained-output analysis, not a GT-free error detector. Min0 online results remain available. Mean=retainedGTmm/(surviving individual stops+1); piece mean=retainedGTmm/surviving pieces. Coverage is always reported; fewer pieces alone is not a winner.', '',
        'Non-dominated over min512 errors, surviving pieces and retained coverage: '+', '.join(nondominated(rows)), '']
    lines += ['Complete cutoff/control comparison: [34 configurations, before/after512 filter](reviewed_ema999_short512_20261006_v1.md). '
              'Original online cutoff results: [post-modulation sweep](reviewed_postmod_presence_ema999_20261006_v1.md).', '']
    def table(source=None):
        result = ['| Variant | Changed args | Min vx | Errors | Pieces | Kept % | Mean mm | Piece mean mm | Median mm | P05 mm | P95 mm |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for label, row in rows.items():
            for minimum, metric in row['metrics'].items():
                c = metric['combined'] if source is None else metric['scrolls'][source]
                nums = ['—' if c[k] is None else f'{c[k]:.3f}' for k in
                        ('mean_mm','piece_mean_mm','median_mm','p05_mm','p95_mm')]
                result.append(f'| {label} | {json.dumps(row["changes"],sort_keys=True)} | {minimum} | {c["ordinary_restarts"]} | {c["kept_segments"]} | {c["retained_percent"]:.3f} | '+ ' | '.join(nums)+' |')
        return result
    lines += table()
    if rows:
        for source in next(iter(rows.values()))['metrics']['0']['scrolls']:
            lines += ['', '## '+source, '']+table(source)
    lines += ['', '## Reproduce', '', 'Set control action `'+action+'`. '+('Candidate plan: `scripts/reviewed_trace_tuning_plan.json`.' if name==NAME else 'Frozen snapshot selections are retained on resume.'), '',
              '```bash', 'bash /home/hendrik/villa3/scripts/run_training_failure_mining', '```', '',
              f'Immutable workloads, native effective config, timing, outputs and per-piece decisions: `{ROOT/name}`. JSON includes source/control paths, checkpoint and inspected native executable hash; the latter describes the binary present at analysis, not independent proof of historical cached-run provenance. Timings include startup and any on-demand cache fill, not isolated tracing speeds.']
    (DOCS/(name+'.md')).write_text('\n'.join(lines)+'\n')


def collect_filtered_result(folder):
    """Shared native result collection and min0/min512 accounting audit."""
    row = measured_result(folder)
    counts = filter_counts(folder)
    for source, (p, r) in counts.items():
        row['scrolls'][source].update(presence_filtered=p, reverse_filtered=r)
    row['combined'].update(presence_filtered=sum(p for p,r in counts.values()), reverse_filtered=sum(r for p,r in counts.values()))
    metrics, cases = analyze(row)
    return row, metrics, cases


def main():
    plan = json.loads(PLAN.read_text())
    if not plan['seeds']:
        raise ValueError('select seeds from completed cutoff sweep first')
    rows = {}
    for seed, item in plan['seeds'].items():
        base = json.loads((Path(item['folder'])/'workload.json').read_text())
        if base['checkpoint_sha256'] != MODELS['ema999_12100'][1] or sha256_file(base['checkpoint']) != base['checkpoint_sha256']:
            raise ValueError('wrong frozen checkpoint')
        trials = [('baseline', {})]+[(k, CHANGES[k]) for k in item['changes']]+list(item.get('combinations', {}).items())
        for name, changes in trials:
            label = seed+'_'+name
            cfg = base if name == 'baseline' else variant(base, label, changes)
            evaluate_config(cfg, label)
            folder = Path(cfg['output_root'])
            effective = json.loads((folder/'full.json').read_text())
            if effective['trace_args'] != cfg['evaluation']['trace_args']:
                raise ValueError('effective trace args mismatch')
            row, metrics, cases = collect_filtered_result(folder)
            native = json.loads((folder/'full/cases/0/trace.json').read_text())['provenance']['effective_trace']
            for key, value in changes.items():
                if native[key.replace('-', '_')] != value:
                    raise ValueError('native tuning mismatch')
            rows[label] = dict(seed=seed, changes=changes, metrics=metrics, summary_path=row['summary_path'],
                checkpoint_sha256=base['checkpoint_sha256'], native_sha256_at_analysis=sha256_file(cfg['native_binary']),
                effective_trace=native, trace_args=cfg['evaluation']['trace_args'],
                wall_seconds=json.loads((folder/'timing.json').read_text())['wall_seconds'])
            atomic_write(ROOT/NAME/(label+'_decisions.json'), cases, durable=False)
            atomic_write(ROOT/NAME/'results.json', rows, durable=False)
            document(rows)
            print('TUNING '+label+' '+json.dumps(metrics['512']['combined']), flush=True)


if __name__ == '__main__':
    main()
