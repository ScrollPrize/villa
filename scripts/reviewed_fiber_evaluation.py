"""Frozen reviewed cohort; delegate all inference/tracing to the existing runner."""
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'vesuvius/src'), str(REPO), str(REPO / 'gpu_manager/src')]
CONFIGS = REPO / 'vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs'
ROOT = Path('/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v3')
REFERENCES = Path('/home/hendrik/data/2026-09-30_fibers_approved_refined')
GAUSSIAN = Path('/ephemeral/hendrik/fiber3d_train2/evaluations/reviewed_fibers_20260930_v2/gaussian/frozen/live_inference/_models/274acb0c686bfb030646d039f2740f7ad6349cd507a4de0843cf26e4f638ccec/snapshot.pt')
HALFSIGMA_RAW = Path('/ephemeral/hendrik/fiber3d_train2/runs/5scroll_json_eligible_fullres_refinement_gaussian_halfsigma_128_sd1_16x8_ema_20260930_155818/snapshots/best.pt')
SIDEBAND_MODEL = 'halfsigma_current_raw_sideband_reviewed32_v1'
SIDEBAND_CURRENT = Path('/ephemeral/hendrik/fiber3d_train2/runs/5scroll_json_eligible_fullres_refinement_gaussian_halfsigma_sideband_reviewed32_128_sd1_16x8_ema_20261001_203634/snapshots/current.pt')
BEST_MODELS = {
    'halfsigma_sideband_best_raw_20261002': ('best.pt', None),
    'halfsigma_sideband_best_ema99_20261002': ('best_ema_0p99.pt', .99),
    'halfsigma_sideband_best_ema999_20261002': ('best_ema_0p999.pt', .999),
}


def freeze_selection(model, source):
    from lasagna.inference_provenance import atomic_write, sha256_file
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import freeze_checkpoint
    selection = ROOT/model/'snapshot_selection.json'
    if selection.exists():
        pinned = json.loads(selection.read_text())
        result = Path(pinned['path']), pinned['sha256']
        if pinned.get('source', str(source)) != str(source) or sha256_file(result[0]) != result[1]:
            raise ValueError('corrupt or mismatched selected snapshot')
    else:
        result = freeze_checkpoint(source, ROOT/model/'frozen')
        atomic_write(selection, dict(path=str(result[0]), sha256=result[1], source=str(source),
                     captured_at=datetime.now(timezone.utc).isoformat()), durable=False)
    print(f'Frozen {model}: {result}', flush=True)
    return result


def validate_best_model(model, saved, evaluation):
    decay = BEST_MODELS[model][1]
    if decay is None:
        validate_raw_unfiltered(saved, evaluation)
    else:
        metric = f'test_ema_{str(decay).replace(".", "p")}/loss_total'
        if (saved.get('snapshot_kind') != 'ema' or saved.get('metric_name') != metric
                or saved.get('ema_metadata', {}).get('decay') != decay):
            raise ValueError('best EMA checkpoint kind/metric/decay mismatch')


def current_raw_source(model):
    return SIDEBAND_CURRENT if model == SIDEBAND_MODEL else HALFSIGMA_RAW.with_name('current.pt')


def halfsigma_namespace(model):
    if model in BEST_MODELS:
        return model
    if model == SIDEBAND_MODEL:
        return 'sideband_reviewed32_203634_current_raw_v1'
    suffix = model.removeprefix('halfsigma_')
    return (f'halfsigma_155818_current_raw_reviewed_{suffix.rsplit("_",1)[1]}' if suffix.startswith('current_raw_')
            else f'fullres_halfsigma_155818_best_{suffix}_reviewed_20260930_v1')


def validate_raw_unfiltered(saved, evaluation):
    if saved.get('snapshot_kind') == 'ema' or saved.get('metric_name') != 'test/loss_total':
        raise ValueError('expected raw best selected by test/loss_total')
    if any('reverse' in arg or 'consistency' in arg for arg in evaluation['trace_args']):
        raise ValueError('unfiltered evaluation cannot contain reverse/consistency options')


def validate_ema99(saved):
    if saved.get('snapshot_kind') != 'ema' or saved.get('metric_name') != 'test_ema_0p99/loss_total':
        raise ValueError('expected best EMA99 selected by test_ema_0p99/loss_total')
    if saved.get('ema_metadata', {}).get('decay', .99) != .99:
        raise ValueError('EMA decay does not match0.99')


def capture_cohort(source, destination):
    from lasagna.inference_provenance import atomic_write, sha256_file
    files = sorted(source.glob('*/*.json'))
    if not files:
        raise ValueError('empty reviewed cohort')
    # Validate the complete input before freezing anything: partial transfers
    # must not silently become a benchmark cohort.
    for p in files:
        try:
            document = json.loads(p.read_text())
        except (ValueError, UnicodeError) as exc:
            raise ValueError(f'invalid reviewed annotation: {p}: {exc}') from exc
        if not isinstance(document, dict):
            raise ValueError(f'reviewed annotation must be an object: {p}')
    manifest = {'source': str(source), 'files': [dict(
        source=p.parent.name, split='test', path=str(p.relative_to(source)),
        sha256=sha256_file(p)) for p in files]}
    path = destination / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('reviewed cohort changed; use a new benchmark version')
    for p, row in zip(files, manifest['files']):
        target = destination / 'test' / row['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(p, target)
        if sha256_file(target) != row['sha256']:
            raise ValueError(f'changed frozen annotation: {target}')
    if not path.exists():
        atomic_write(path, manifest, durable=False)
    return manifest


def validate_result(result):
    if (not result.get('completed') or result.get('route_state_limit_cases', 0)
            or result.get('infrastructure_failures', 0)
            or result.get('evaluated_cases') != result.get('cases') or not result.get('cases')):
        raise RuntimeError('evaluation incomplete or infrastructure-limited')


def validate_workload(config, cohort, output, manifest):
    counts = {}
    for row in manifest['files']:
        counts[row['source']] = counts.get(row['source'], 0) + 1
    configured = {s['id']: s['expected_test_count'] for s in config['sources']}
    if (config['split_root'] != str(cohort) or config['output_root'] != str(output)
            or config.get('split') != 'test' or counts != configured
            or len(config['sources']) != len(configured)):
        raise ValueError('saved workload does not match reviewed cohort/output')


def run(model):
    if model not in ('lowres', 'fullres', 'gaussian', 'halfsigma_raw', 'halfsigma_ema99', 'halfsigma_current_raw_v1', 'halfsigma_current_raw_v2', 'halfsigma_current_raw_v3', SIDEBAND_MODEL) and model not in BEST_MODELS:
        raise ValueError('unknown reviewed evaluation model')
    from lasagna.inference_provenance import atomic_write, sha256_file
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import freeze_checkpoint
    early_snapshot = None
    if (model.startswith('halfsigma_current_raw_') or model in BEST_MODELS) and not (ROOT/model/'workload.json').exists():
        source = SIDEBAND_CURRENT.with_name(BEST_MODELS[model][0]) if model in BEST_MODELS else current_raw_source(model)
        early_snapshot = freeze_selection(model, source)
    cohort = ROOT / 'cohort'
    manifest = capture_cohort(REFERENCES, cohort)
    output = ROOT / model
    config_path = output / 'workload.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        base = 'lowres' if model == 'lowres' else 'fullres'
        config = json.loads((CONFIGS / f'eval_train_mining_{base}_20260927.json').read_text())
        counts = {}
        for row in manifest['files']:
            counts[row['source']] = counts.get(row['source'], 0) + 1
        bindings = {s['sample_id']: s for s in config['sources']}
        config['sources'] = []
        for name, count in sorted(counts.items()):
            binding = dict(bindings[name], id=name, expected_test_count=count)
            binding.pop('expected_train_count', None)
            config['sources'].append(binding)
        config.update(split_root=str(cohort), split='test', output_root=str(output),
                      devices='0,1,2,3,4,5', pilot_devices='0,1,2,3,4,5',
                      pilot_source_id=config['sources'][0]['id'],
                      native_binary=str(REPO / 'volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_metric'))
        config.pop('fiber_order_seed', None)
        evaluation = config['evaluation']
        evaluation['workers'] = 192
        evaluation['gpu_manager'].update(cpu_cores=192, devices=config['devices'], label=f'reviewed-{model}')
        evaluation['render'] = False
        if model == 'gaussian' or model.startswith('halfsigma_'):
            import torch
            selected = GAUSSIAN if model == 'gaussian' else HALFSIGMA_RAW
            if model == 'halfsigma_ema99':
                selected = HALFSIGMA_RAW.with_name('best_ema_0p99.pt')
            elif model.startswith('halfsigma_current_raw_'):
                selected = current_raw_source(model)
            elif model in BEST_MODELS:
                selected = SIDEBAND_CURRENT.with_name(BEST_MODELS[model][0])
            snapshot, digest = early_snapshot or freeze_checkpoint(selected, output / 'frozen')
            saved = torch.load(snapshot, map_location='cpu', weights_only=False)
            if model in BEST_MODELS:
                validate_best_model(model, saved, evaluation)
            elif model == 'halfsigma_raw' or model.startswith('halfsigma_current_raw_'):
                validate_raw_unfiltered(saved, evaluation)
            elif model == 'halfsigma_ema99':
                validate_ema99(saved)
            elif saved.get('snapshot_kind') != 'ema' or saved.get('metric_name') != 'test_ema_0p999/loss_total':
                raise ValueError('expected best EMA999 checkpoint')
            atomic_write(output / 'snapshot_metadata.json', {
                'source': str(selected), 'sha256': digest,
                **{k: saved.get(k) for k in ('step', 'samples_seen', 'metric', 'metric_name', 'snapshot_kind', 'ema_metadata')}}, durable=False)
            config.update(checkpoint=str(snapshot), checkpoint_sha256=digest)
            evaluation['inference_cache_namespace'] = ('fullres_gaussian_092844_best_ema999_reviewed_20260930_v1'
                if model == 'gaussian' else halfsigma_namespace(model))
            del saved
        atomic_write(config_path, config, durable=False)
    if model == 'halfsigma_current_raw_v1':
        # Keep below the cache API's64-character component limit, including
        # when resuming the initial pre-inference namespace-validation failure.
        config['evaluation']['inference_cache_namespace'] = halfsigma_namespace(model)
        prepared = output/'full.json'
        if prepared.exists() and len(json.loads(prepared.read_text()).get('inference_cache_namespace','')) > 64:
            if (output/'full').exists():
                raise ValueError('cannot migrate a namespace after evaluation started')
            archive = output/'rejected_namespace_preflight'
            archive.mkdir(exist_ok=True)
            for name in ('full.json','full_bindings.json'):
                if (archive/name).exists():
                    raise ValueError('refusing to replace archived preflight evidence')
                (output/name).rename(archive/name)
        atomic_write(config_path, config, durable=False)
    if sha256_file(config['checkpoint']) != config['checkpoint_sha256']:
        raise ValueError('checkpoint hash mismatch')
    validate_workload(config, cohort, output, manifest)
    execute_evaluation(config_path, config, model)
    if model.startswith('halfsigma_'):
        report_halfsigma(model)


def report_halfsigma(selected_model='halfsigma_raw'):
    from lasagna.inference_provenance import atomic_write
    from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import complete_report
    models = ('lowres','fullres','gaussian','halfsigma_raw')
    if selected_model.startswith('halfsigma_current_raw_'):
        models += ('halfsigma_ema99',)
    if selected_model in ('halfsigma_current_raw_v2','halfsigma_current_raw_v3', SIDEBAND_MODEL):
        models += ('halfsigma_current_raw_v1',)
    if selected_model in ('halfsigma_current_raw_v3', SIDEBAND_MODEL):
        models += ('halfsigma_current_raw_v2',)
    if selected_model == SIDEBAND_MODEL:
        models += ('halfsigma_current_raw_v3',)
    if selected_model in BEST_MODELS:
        models += ('halfsigma_current_raw_v3', SIDEBAND_MODEL)
        for earlier in BEST_MODELS:
            if earlier == selected_model:
                break
            if (ROOT/earlier/'full/summary.json').exists():
                models += (earlier,)
    if selected_model != 'halfsigma_raw':
        models += (selected_model,)
    results = {}
    expected_references = None
    for model in models:
        folder = ROOT/model
        summary = json.loads((folder/'full/summary.json').read_text())
        validate_result(summary)
        assert summary['cases'] == 182 and summary['eligible_runs'] == 91
        references = []
        for source in summary['sources']:
            effective = source['backend_config']['effective_trace']
            assert effective['embedding_weight'] == 0
            assert effective.get('replay_reverse_interval_base',0) == 0
            assert all(not c.get('reverse_checks',0) and not c.get('filtered_stops',[]) for c in source['cases'])
            references.extend((source['source_id'],c['source_index'],c['direction'],c['reference_run']) for c in source['cases'])
        if expected_references is None:
            expected_references = references
        assert references == expected_references, 'reference geometry/selection changed'
        stats = complete_report(summary)
        workload = json.loads((folder/'workload.json').read_text())
        inference = json.loads((folder/'full/inference_summary.json').read_text())
        results[model] = dict(summary_path=str(folder/'full/summary.json'),
            checkpoint_sha256=workload['checkpoint_sha256'],
            coordinator_seconds=inference['wall_seconds'], **stats)
    date = '20261001' if selected_model in ('halfsigma_current_raw_v2','halfsigma_current_raw_v3', SIDEBAND_MODEL) else '20260930'
    if selected_model in BEST_MODELS:
        date = '20261002'
    target = REPO/f'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/benchmarks/reviewed_{selected_model}_{date}'
    metadata = json.loads((ROOT/selected_model/'snapshot_metadata.json').read_text())
    atomic_write(target.with_suffix('.json'),dict(snapshot=metadata,results=results),durable=False)
    lines = [f'# Reviewed cohort: {selected_model}', '',
        f'Snapshot: `{metadata["source"]}`, frozen update{metadata["step"]}, `{metadata["metric_name"]}`={metadata["metric"]:.8f}. SHA256 `{metadata["sha256"]}`.', '',
        'Same corrected reviewedv3 cohort:91fibers,182directions, five scrolls. All directions completed with zero infrastructure failures. No embedding, no fw/bw filtering. Native beam8/lookahead2, step4prediction voxels,40/80base ellipse,160base grace,80base restart grouping. Fullres prediction lattice2base voxels,64³output chunks.192workers, GPUs0–5 demand-managed.', '',
        'Mean uses the established combined directed reference distance/(restarts+1); quantiles use reference partitions including successful terminal spans. This selected reviewed hard-region cohort is not proven to be an independent training holdout.', '',
        '| Model | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def row(label,s):
        return f'| {label} | {s["failures"]} | {s["failures"]/s["directed_reference_mm"]:.5f} | '+ ' | '.join(f'{s[k]:.3f}' for k in ('mean_segment_length_mm','median_segment_length_mm','p05_segment_length_mm','p95_segment_length_mm'))+' |'
    for model in models:
        lines.append(row(model,results[model]['global_result']))
    lines += ['', '## Per scroll', '', '| Scroll / model | Restarts | Failures/mm | Mean mm | Median mm | P5 mm | P95 mm |','|---|---:|---:|---:|---:|---:|---:|']
    for scroll in results['halfsigma_raw']['scrolls']:
        for model in models:
            lines.append(row(f'{scroll} / {model}',results[model]['scrolls'][scroll]))
    lines += ['', '## Reproduction', '',
        'Set scripts/training_failure_mining_control.json to `'+json.dumps(dict(action='reviewed_eval',reviewed_model=selected_model))+'`, then:', '',
        '```bash','bash /home/hendrik/villa3/scripts/run_training_failure_mining','```','',
        f'Output: `{ROOT}/{selected_model}/`. Frozen snapshot, selection metadata, workload, preparation/binding validation and complete per-case artifacts retained there.', '',
        'Coordinator seconds (cache/inference work differs; not a model-speed comparison): '+', '.join(f'{m}={results[m]["coordinator_seconds"]:.2f}' for m in models)+'.', '',
        'Validation:focused Python tests; independent plan review; identical per-case reference selection/lengths; all182cases complete; effective embedding/reverse settings and saved per-case check counters verified disabled. No native or scheduling changes.']
    target.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:13]),flush=True)


def execute_evaluation(config_path, config, model):
    output = Path(config['output_root'])
    summary = output / 'full/summary.json'
    if summary.exists() and json.loads(summary.read_text()).get('completed'):
        validate_result(json.loads(summary.read_text()))
        print(f'Already completed: {summary}', flush=True)
        return
    if config['devices'] != '0,1,2,3,4,5' or config['evaluation']['gpu_manager']['devices'] != '0,1,2,3,4,5':
        raise ValueError('evaluation must request only GPUs0–5')
    runner = REPO / 'vesuvius/scripts/run_classic_fiber_evaluation.py'
    for mode in ('prepare', 'validate-full', 'full'):
        print(f'Reviewed {model}: {mode}', flush=True)
        subprocess.run([sys.executable, str(runner), mode, '--config', str(config_path)], check=True, cwd=REPO)
    result = json.loads(summary.read_text())
    validate_result(result)
    print(f'COMPLETED {model}: {summary}', flush=True)


def run_reverse(model):
    if model not in ('gaussian', 'lowres', 'halfsigma_sideband_best_raw_20261002'):
        raise ValueError('unsupported reverse experiment model')
    from lasagna.inference_provenance import atomic_write, sha256_file
    original = json.loads((ROOT/model/'workload.json').read_text())
    output = ROOT/'reverse256_v1'/model
    config = json.loads(json.dumps(original))
    config['output_root'] = str(output)
    if any('reverse' in arg or 'consistency' in arg for arg in config['evaluation']['trace_args']):
        raise ValueError('reverse experiment requires an unfiltered baseline')
    config['evaluation']['trace_args'] += ['--replay-reverse-interval-base', '256']
    config['evaluation']['gpu_manager']['label'] = f'reviewed-reverse-{model}'
    path = output/'workload.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('reverse experiment config changed; use new version')
    if sha256_file(config['checkpoint']) != config['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    atomic_write(path, config, durable=False)
    execute_evaluation(path, config, model)


if __name__ == '__main__':
    control = json.loads((REPO / 'scripts/training_failure_mining_control.json').read_text())
    if control['action'] == 'reviewed_reverse':
        run_reverse(control['reviewed_model'])
        if control.get('reference_reverse_after'):
            from reviewed_reference_reverse import run as run_reference_reverse
            run_reference_reverse(control['reviewed_model'])
    else:
        models = control.get('reviewed_models', [control.get('reviewed_model')])
        for model in models:
            if model in BEST_MODELS:
                freeze_selection(model, SIDEBAND_CURRENT.with_name(BEST_MODELS[model][0]))
        for model in models:
            run(model)
