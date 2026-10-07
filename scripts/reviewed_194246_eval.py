"""Frozen raw and EMA selections, exact previously winning tracing policy."""
import copy
import json
from pathlib import Path

from reviewed_normal_width_eval import ROOT, REPO, RUNS, PREFIX, freeze_selection, validate_snapshot, evaluate_config
from reviewed_trace_tuning import NAME as TUNING_NAME, DOCS, collect_filtered_result, document
from lasagna.inference_provenance import atomic_write, sha256_file

NAME = 'normalwidth_194246_tuned_20261007_v1'
BASE = ROOT/TUNING_NAME/'p40_cos40_norev_step6'
SELECTIONS = (
    ('current', 'current.pt', None),
    ('best_loss', 'best.pt', 'test/loss_total'),
    ('best_restarts', 'best_reviewed_trace.pt', 'reviewed_trace/restarts_combined'),
    ('current_ema99', 'current_ema_0p99.pt', 'test_ema_0p99/loss_total'),
    ('best_loss_ema99', 'best_ema_0p99.pt', 'test_ema_0p99/loss_total'),
    ('current_ema999', 'current_ema_0p999.pt', 'test_ema_0p999/loss_total'),
    ('best_loss_ema999', 'best_ema_0p999.pt', 'test_ema_0p999/loss_total'),
)


def validate_selection(saved, filename, metric):
    if '_ema_' not in filename:
        validate_snapshot(saved, metric)
        return 'raw'
    decay = 0.999 if '0p999' in filename else 0.99
    if (saved.get('snapshot_kind') != 'ema' or
            saved.get('ema_metadata', {}).get('decay') != decay or
            saved.get('metric_name') != metric):
        raise ValueError('EMA selection kind, decay or metric mismatch')
    return 'EMA999' if decay == 0.999 else 'EMA99'


def configuration(base, label, snapshot, digest):
    cfg = copy.deepcopy(base)
    cfg.update(checkpoint=str(snapshot), checkpoint_sha256=digest, output_root=str(ROOT/NAME/(label+'_eval')))
    cfg['evaluation']['inference_cache_namespace'] = 'nw194246_'+digest[:32]
    cfg['evaluation']['gpu_manager']['label'] = NAME+'_'+label
    return cfg


def main(control):
    import torch
    base = json.loads((BASE/'workload.json').read_text())
    pending = []
    # Freeze every mutable selection before the first evaluation starts.
    for label, filename, metric in SELECTIONS:
        source = RUNS/(PREFIX+'194246')/'snapshots'/filename
        selection_name = NAME+'/'+label
        snapshot, digest = freeze_selection(selection_name, source)
        saved = torch.load(snapshot, map_location='cpu', weights_only=False, mmap=True)
        kind = validate_selection(saved, filename, metric)
        meta = dict(source=str(source), path=str(snapshot), sha256=digest,
            **{k:saved.get(k) for k in ('step','samples_seen','metric','metric_name','snapshot_kind','reviewed_trace_selection','reviewed_trace_metrics')})
        meta['kind'] = kind
        meta['ema_metadata'] = saved.get('ema_metadata')
        meta['captured_at'] = json.loads((ROOT/selection_name/'snapshot_selection.json').read_text())['captured_at']
        del saved
        atomic_write(ROOT/selection_name/'snapshot_metadata.json', meta, durable=False)
        print('SNAPSHOT '+label+' '+json.dumps(meta), flush=True)
        pending.append((label, configuration(base,label,snapshot,digest),meta))
    if control.get('freeze_only'):
        return
    prior = json.loads((DOCS/(TUNING_NAME+'.json')).read_text())['p40_cos40_norev_step6']
    rows = {'Previous EMA999 u12100': prior}
    def cohort(cfg):
        return [(s['id'],s['coordinate_space'],s['base_voxel_size_um'],s['fibers']) for s in cfg['sources']]
    baseline_effective = json.loads((BASE/'full.json').read_text())
    for label,cfg,meta in pending:
        evaluate_config(cfg,NAME+'_'+label)
        folder = Path(cfg['output_root'])
        effective = json.loads((folder/'full.json').read_text())
        if effective['trace_args'] != base['evaluation']['trace_args'] or cohort(effective)!=cohort(baseline_effective):
            raise ValueError('cohort or winning trace args changed')
        row, metrics, cases = collect_filtered_result(folder)
        native = json.loads((folder/'full/cases/0/trace.json').read_text())['provenance']['effective_trace']
        if native != prior['effective_trace']:
            raise ValueError('effective native policy differs from winning control')
        key = label+' '+meta['kind']+' u'+str(meta['step'])
        rows[key] = dict(metrics=metrics, changes={}, snapshot=meta, summary_path=row['summary_path'],
            checkpoint_sha256=meta['sha256'], native_sha256_at_analysis=sha256_file(cfg['native_binary']),
            effective_trace=native, trace_args=effective['trace_args'],
            wall_seconds=json.loads((folder/'timing.json').read_text())['wall_seconds'])
        atomic_write(ROOT/NAME/(label+'_decisions.json'), cases, durable=False)
        atomic_write(ROOT/NAME/'results.json', rows, durable=False)
        notes = ['194246 raw and EMA checkpoints, no additional tuning. All use postcutoff.4,cos40,noFW,step6trace=12base and original weights. Previous EMA999u12100 is the matched historical control. GPUs0–5; inference settings unchanged.',
                 'Best-restarts refers to the training selection protocol. Only one best_reviewed_trace selection exists; no separate EMA best-restarts files are available. Raw selections were frozen in the initial run; the four EMA selections were frozen before this continuation. Identical hashes mean identical selections, not independent checkpoints.', '',
                 '| Selection | Step | Samples | Selection metric | SHA256 | Captured UTC |', '|---|---:|---:|---|---|---|']
        notes += [f'| {l} {m["kind"]} | {m["step"]} | {m["samples_seen"]} | {m["metric_name"]} | {m["sha256"]} | {m["captured_at"]} |' for l,c,m in pending]
        document(rows,name=NAME,title='194246 raw and EMA selections — winning tracing policy',notes=notes,action='reviewed_194246')
        print('COMPLETED '+key+' '+json.dumps(metrics['512']['combined']),flush=True)


if __name__ == '__main__':
    main(json.loads((REPO/'scripts/training_failure_mining_control.json').read_text()))
