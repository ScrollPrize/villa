"""Reproduction experiment entrypoint; CPU inspection does not initialize CUDA."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import threading
import re
import statistics
import glob
import hashlib
import pickle
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path('/ephemeral/hendrik/fiber3d_train2/reproduction')
HISTORICAL = Path('/ephemeral/hendrik/fiber3d_train/runs')
CONTROL = Path(__file__).with_name('fiber3d_experiment.json')


def input_manifest(raw):
    result = {}
    for key in ('datasets', 'test_datasets', 'historical_test_datasets'):
        rows = []
        for dataset in raw.get(key, []):
            for filename in sorted(glob.glob(dataset['fiber_glob'])):
                path = Path(filename)
                rows.append({'path':str(path), 'bytes':path.stat().st_size,
                             'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        result[key] = rows
    return result


def remote_auth_probe():
    """Verify a public remote volume opens without invoking an AWS helper."""
    settings = json.loads(CONTROL.read_text())['remote_auth_probe']
    with tempfile.TemporaryDirectory(prefix='vc-public-auth-probe-') as temporary:
        root = Path(temporary)
        fake_bin = root / 'bin'
        fake_bin.mkdir()
        marker = root / 'aws-called'
        helper = fake_bin / 'aws'
        helper.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 97\n')
        helper.chmod(0o755)
        code = (
            'from vc.volume import Volume\n'
            'import sys\n'
            'volume = Volume.open_url(sys.argv[1], sys.argv[2])\n'
            'print({"shape": volume.shape, "levels": volume.present_scale_levels()})\n'
        )
        env = dict(os.environ)
        env['PATH'] = str(fake_bin) + os.pathsep + env.get('PATH', '')
        subprocess.run(
            [sys.executable, '-c', code, settings['url'], str(root / 'cache')],
            cwd=ROOT,
            env=env,
            check=True,
            timeout=float(settings.get('timeout_seconds', 60)),
        )
        if marker.exists():
            raise RuntimeError('public remote open invoked the AWS credential helper')
        print('public remote open completed without AWS credential discovery', flush=True)


def trace_evaluate(settings=None):
    """Freeze selected raw/EMA weights and reuse the locked managed evaluator."""
    import math
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import freeze_checkpoint
    if settings is None:
        settings = json.loads(CONTROL.read_text())['trace_evaluation']
    run = Path(settings['run'])
    runs = [run, *map(Path, settings.get('additional_runs', []))]
    paths = [path for source in runs for path in (
        [source / 'snapshots/best.pt'] if settings['variant'] == 'raw'
        else sorted((source / 'snapshots').glob('best_ema_*.pt')))]
    if settings['variant'] not in ('raw', 'ema') or not paths:
        raise ValueError('need raw or ema selection and corresponding best snapshots')
    candidates = []
    for path in paths:
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if int(payload['samples_seen']) > int(settings.get('max_samples', 2**63 - 1)):
            raise ValueError(f'checkpoint is beyond comparison cutoff: {path}')
        metric = float(payload['metric'])
        if not math.isfinite(metric):
            raise ValueError(f'nonfinite checkpoint metric: {path}')
        expected_metric = 'test/loss_total' if settings['variant'] == 'raw' else f'test_{path.stem[5:]}/loss_total'
        if payload['metric_name'] != expected_metric:
            raise ValueError(f'checkpoint selection metric mismatch: {path}')
        candidates.append(dict(path=str(path), metric=metric, step=payload['step'],
                               samples=payload['samples_seen'], metric_name=payload['metric_name']))
    selected = min(candidates, key=lambda row:(row['metric'], row['path']))
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=False)
    frozen, digest = freeze_checkpoint(Path(selected['path']), output / 'frozen')
    selection = dict(run=str(run), candidates=candidates, selected=selected,
                     frozen=str(frozen), sha256=digest)
    return evaluate_frozen(selection, settings['template'], output)


def evaluate_frozen(selection, template, output):
    """Single shared invocation path, retaining managed evaluator admission."""
    workload = json.loads((ROOT / template).read_text())
    workload.update(checkpoint=selection['frozen'], checkpoint_sha256=selection['sha256'], output_root=str(output))
    workload['evaluation']['inference_cache_namespace'] = 'eval-' + hashlib.sha256(
        str(output.resolve()).encode()).hexdigest()[:48]
    config = output / 'workload.json'
    config.write_text(json.dumps(workload, indent=2))
    (output / 'selection.json').write_text(json.dumps(selection, indent=2))
    print(f"trace selection {selection['selected']} sha256={selection['sha256']} output={output}", flush=True)
    script = ROOT / 'vesuvius/scripts/run_classic_fiber_evaluation.py'
    for mode in ('prepare', 'validate-full', 'full'):
        print(f'trace {mode}: log={output / (mode + ".log")}', flush=True)
        with (output / (mode + '.log')).open('w') as log:
            subprocess.run([sys.executable, str(script), mode, '--config', str(config)],
                           cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    return output


EMA_SERIES = [('current999', 'current_ema_0p999.pt', 0.999),
              ('best999', 'best_ema_0p999.pt', 0.999),
              ('best99', 'best_ema_0p99.pt', 0.99)]
SNAPSHOT_VARIANTS = [*EMA_SERIES, ('current99', 'current_ema_0p99.pt', 0.99),
                     ('current', 'current.pt', None), ('best', 'best.pt', None),
                     ('best_trace', 'best_reviewed_trace.pt', None)]


def check_series_training_state(settings, identity):
    from villa_gpu_manager.processes import process_state
    expected = settings.get('expected_training_state', 'alive')
    if expected not in ('alive', 'dead'):
        raise ValueError('expected_training_state must be alive or dead')
    observed = process_state(identity, os.getuid())
    if observed != expected:
        raise RuntimeError(f'expected training root {expected}, observed {observed}; no automatic restart')
    return observed


def freeze_snapshot_selection(run, filename, decay, cache):
    import math
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import freeze_checkpoint
    path = run / 'snapshots' / filename
    frozen, digest = freeze_checkpoint(path, cache)
    payload = torch.load(frozen, map_location='cpu', weights_only=True, mmap=True)
    expected_metric = ('reviewed_trace/restarts_combined'
                       if filename == 'best_reviewed_trace.pt' else
                       'test/loss_total' if decay is None else
                       'test_' + filename.split('_', 1)[1][:-3] + '/loss_total')
    kind_matches = ((payload.get('snapshot_kind') in (None, 'raw') and
                     'ema_metadata' not in payload) if decay is None else
                    (payload.get('snapshot_kind') == 'ema' and
                     payload.get('ema_metadata', {}).get('decay') == decay))
    if (not kind_matches or payload.get('metric_name') != expected_metric or
            not math.isfinite(float(payload['metric']))):
        raise ValueError(f'snapshot metadata mismatch: {path}')
    selected = dict(path=str(path), metric=float(payload['metric']), metric_name=payload['metric_name'],
                    step=int(payload['step']), samples=int(payload['samples_seen']), decay=decay)
    return dict(run=str(run), selected=selected, candidates=[selected],
                frozen=str(frozen), sha256=digest)


def trace_series_prepare():
    settings = json.loads(CONTROL.read_text())['trace_series']
    available = {name: (name, filename, decay) for name, filename, decay in SNAPSHOT_VARIANTS}
    variants = settings.get('variants', [name for name, _, _ in EMA_SERIES])
    if (not isinstance(variants, list) or not variants or
            any(not isinstance(name, str) or name not in available for name in variants) or
            len(set(variants)) != len(variants)):
        raise ValueError('variants must be a nonempty unique list of supported snapshot names')
    identity = json.loads((Path(settings['launch']) / 'controller.json').read_text())
    check_series_training_state(settings, identity)
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    for name, filename, decay in (available[name] for name in variants):
        selection = freeze_snapshot_selection(Path(settings['run']), filename, decay, root / 'frozen')
        entries.append(dict(name=name, status='pending', selection=selection))
        print(f'frozen {name}: {selection}', flush=True)
    state = dict(settings=settings, training_identity=identity, entries=entries,
                 captured_at=datetime.now(timezone.utc).isoformat())
    (root / 'series.json').write_text(json.dumps(state, indent=2))


def trace_series_next(*, retry_failed=False):
    from fiber3d_comparison_report import series_report
    from vesuvius.neural_tracing.fiber_trace_3d.replay_metrics import complete_report
    root = Path(json.loads(CONTROL.read_text())['trace_series']['output'])
    path = root / 'series.json'
    state = json.loads(path.read_text())
    entry = next((e for e in state['entries'] if e['status'] != 'done'), None)
    if entry is None:
        print('all configured evaluations complete', flush=True)
        return
    if retry_failed and entry['status'] != 'failed':
        raise RuntimeError('explicit retry requires a failed evaluation')
    if entry['status'] != 'pending' and not retry_failed:
        raise RuntimeError('previous evaluation interrupted/failed; inspect before retrying')
    check_series_training_state(state['settings'], state['training_identity'])
    output = root / entry['name']
    completed_output = (Path(entry.get('attempt_output', output))
                        if retry_failed else None)
    reuse_completed = bool(completed_output and
        (completed_output / 'full/summary.json').is_file() and
        json.loads((completed_output / 'full/summary.json').read_text()).get('completed'))
    if retry_failed and not reuse_completed:
        attempts = entry.setdefault('failed_attempts', [])
        attempts.append(dict(output=entry.get('attempt_output', str(output)), error=entry.get('error')))
        output = root / f"{entry['name']}-retry{len(attempts)}"
    elif reuse_completed:
        output = completed_output
    if not reuse_completed:
        output.mkdir(exist_ok=False)
    entry['status'] = 'running'
    entry['attempt_output'] = str(output)
    entry.pop('error', None)
    path.write_text(json.dumps(state, indent=2))
    try:
        if not reuse_completed:
            evaluate_frozen(entry['selection'], state['settings']['template'], output)
        audit = json.dumps(complete_report(json.loads(
            (output / 'full/summary.json').read_text())), indent=2)
        (output / 'trace_audit.json').write_text(audit)
        entry.update(status='done', metrics=json.loads(audit), output=str(output))
        if state['settings'].get('report_baselines', True):
            report = series_report(state)
            report_path = root / f"comparison_after_{entry['name']}.md"
        else:
            report = json.dumps({
                'completed': [
                    {key: row[key] for key in ('name', 'selection', 'metrics', 'output')}
                    for row in state['entries'] if row['status'] == 'done'
                ]
            }, indent=2)
            report_path = root / f"results_after_{entry['name']}.json"
        report_path.write_text(report)
        entry['report'] = str(report_path)
        entry['training_root_state'] = check_series_training_state(state['settings'], state['training_identity'])
        path.write_text(json.dumps(state, indent=2))
        print(report, flush=True)
        print(f"REPORT={report_path}; original training root remains {entry['training_root_state']}", flush=True)
    except BaseException as exc:
        entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        path.write_text(json.dumps(state, indent=2))
        raise


def comparison_followup():
    """Evaluate a completed sample-boundary stop, then continue the same run."""
    settings = json.loads(CONTROL.read_text())['comparison_followup']
    launch = Path(settings['launch'])
    stopped = json.loads((launch / 'sample_stop_result.json').read_text())
    if stopped['snapshot_samples'] != settings['max_samples']:
        raise ValueError('follow-up stop does not match requested sample boundary')
    from villa_gpu_manager.processes import process_state
    if process_state(stopped['identity'], os.getuid()) != 'dead':
        raise RuntimeError('follow-up training root has not stopped')
    root = Path(settings['output'])
    root.mkdir(parents=True, exist_ok=False)
    report = {'stop': stopped, 'evaluations': []}
    for variant in ('raw', 'ema'):
        output = trace_evaluate(dict(run=stopped['run'], additional_runs=[settings['parent_run']],
            variant=variant, max_samples=settings['max_samples'], template=settings['template'],
            output=str(root / variant)))
        script = ROOT / 'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/trace_restart_analysis_20260911.py'
        audit = subprocess.check_output([sys.executable, str(script), str(output / 'full')], cwd=ROOT, text=True)
        (output / 'trace_audit.json').write_text(audit)
        report['evaluations'].append(dict(variant=variant, output=str(output),
            selection=json.loads((output / 'selection.json').read_text()), metrics=json.loads(audit)))
        (root / 'report.json').write_text(json.dumps(report, indent=2))
    folder, process = production_start(settings['continuation_config'],
        resume=Path(stopped['run']) / 'snapshots/current.pt')
    report['continuation'] = {'launch': str(folder), 'pid': process.pid}
    (root / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def production_start(config_path=None, resume=None):
    """Start the ordinary adaptive trainer, with no outer exclusive reservation."""
    from villa_gpu_manager.processes import process_identity, process_state
    production = None
    if config_path is None:
        production = json.loads(CONTROL.read_text())['production']
        config_path = production['config']
    config_path = (ROOT / config_path).resolve()
    config = json.loads(config_path.read_text())
    root = Path(config['training']['run_path']) / '.launches'
    root.mkdir(parents=True, exist_ok=True)
    latest = root / 'latest'
    if latest.exists():
        previous = Path(latest.read_text())
        identity = json.loads((previous / 'controller.json').read_text())
        if process_state(identity, os.getuid()) != 'dead':
            raise RuntimeError(f'previous production launch is still alive: {previous}')
    folder = root / (config['training']['run_name'] + '-' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    folder.mkdir()
    (folder / 'config.json').write_text(json.dumps(config, indent=2))
    (folder / 'inputs.json').write_text(json.dumps(input_manifest(config), indent=2))
    command = [sys.executable, '-m', 'vesuvius.neural_tracing.fiber_trace_3d.train', str(config_path)]
    if resume is not None:
        command += ['--resume', str(resume)]
    (folder / 'command.json').write_text(json.dumps(command))
    (folder / 'source.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True)
        + subprocess.check_output(['git','diff'], cwd=ROOT,text=True))
    environment = dict(os.environ, PYTHONUNBUFFERED='1')
    if production is not None and production.get('cuda_visible_devices') is not None:
        environment['CUDA_VISIBLE_DEVICES'] = str(production['cuda_visible_devices'])
    with (folder / 'train.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(process.pid)
    (folder / 'controller.json').write_text(json.dumps(identity))
    latest.write_text(str(folder))
    print(f'managed training root={process.pid} log={folder / "train.log"}', flush=True)
    return folder, process


def comparison():
    """Serial managed experiment campaign; abort rather than skip failed stages."""
    import torch
    settings = json.loads(CONTROL.read_text())['comparison']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=False)
    report = {'settings': settings, 'stages': [], 'evaluations': []}
    report_path = output / 'comparison.json'
    active_training = None

    def save():
        report_path.write_text(json.dumps(report, indent=2))

    def audit(evaluation_root):
        script = ROOT / 'vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/trace_restart_analysis_20260911.py'
        result = subprocess.check_output([sys.executable, str(script), str(evaluation_root / 'full')], cwd=ROOT, text=True)
        metrics = json.loads(result)
        (evaluation_root / 'trace_audit.json').write_text(result)
        selection = json.loads((evaluation_root / 'selection.json').read_text())
        report['evaluations'].append(dict(output=str(evaluation_root), selection=selection, metrics=metrics))
        save()
        print(f"AUDIT {evaluation_root.name}: {metrics['global_result']}", flush=True)

    save()
    try:
        # First raw evaluation was launched interactively. Require its complete
        # result; never race unfinished predictions or silently reuse a failure.
        audit(Path(settings['s1a_raw_evaluation']))
        if report['evaluations'][0]['selection']['run'] != settings['stages'][0]['existing_run']:
            raise ValueError('initial raw evaluation belongs to a different run')
        for stage in settings['stages']:
            if stage.get('existing_run'):
                run = Path(stage['existing_run'])
                report['stages'].append(dict(name=stage['name'], run=str(run), config=stage['config']))
                save()
            else:
                folder, process = production_start(stage['config'])
                active_training = json.loads((folder / 'controller.json').read_text())
                save()
                deadline = time.monotonic() + 8 * 3600
                while process.poll() is None:
                    if time.monotonic() > deadline:
                        identity = json.loads((folder / 'controller.json').read_text())
                        stop_production_identity(identity)
                        raise TimeoutError(f'training exceeded8h safety budget: {folder}')
                    time.sleep(5)
                if process.returncode != 0:
                    raise RuntimeError(f'training exit={process.returncode}; inspect {folder / "train.log"}')
                active_training = None
                text = (folder / 'train.log').read_text()
                run = Path(re.findall(r'run_dir=(\S+)', text)[-1])
                payload = torch.load(run / 'snapshots/current.pt', map_location='cpu', weights_only=True, mmap=True)
                expected = json.loads((ROOT / stage['config']).read_text())['training']['max_steps']
                if payload['step'] != expected:
                    raise RuntimeError(f'training did not publish expected final step {expected}')
                report['stages'].append(dict(name=stage['name'], run=str(run), config=stage['config'], launch=str(folder),
                    samples=payload['samples_seen'], step=payload['step']))
                del payload
                save()
            for variant in stage['variants']:
                evaluation_root = trace_evaluate(dict(run=str(run), variant=variant,
                    output=str(output / f"{stage['name']}_{variant}"), template=settings['template']))
                audit(evaluation_root)
        if len(report['evaluations']) != 6:
            raise RuntimeError('comparison requires all six successful evaluations')
        winner = max(report['evaluations'], key=lambda row:row['metrics']['global_result']['mean_segment_length_mm'])
        run = Path(winner['selection']['run'])
        checkpoint = run / 'snapshots/current.pt'
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        source_stage = next(stage for stage in report['stages'] if stage['run'] == str(run))
        config = json.loads((ROOT / source_stage['config']).read_text())
        config['training']['max_steps'] = 2000000
        # Resume defaults can otherwise preserve a short experiment cap.
        resume_config = output / 'continuation_config.json'
        resume_config.write_text(json.dumps(config, indent=2))
        report['winner'] = dict(evaluation=winner['output'], run=str(run), resume=str(checkpoint),
            traced_checkpoint=winner['selection']['selected'], continuation_config=str(resume_config))
        save()
        del payload
        folder, process = production_start(resume_config, resume=checkpoint)
        report['continuation_launch'] = str(folder)
        report['continuation_pid'] = process.pid
        save()
        print(f'COMPARISON COMPLETE; continuation launched: {report["winner"]}', flush=True)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        if active_training is not None:
            try:
                stop_production_identity(active_training)
            except Exception as cleanup:
                report['cleanup_error'] = f'{type(cleanup).__name__}: {cleanup}'
        save()
        raise


def stop_production_identity(identity):
    """Reuse bounded trainer shutdown, never signal a recycled PID."""
    from villa_gpu_manager.processes import process_state
    state = process_state(identity, os.getuid())
    if state == 'dead':
        return
    if state != 'alive':
        raise RuntimeError(f'cannot verify training identity: {state}')
    os.kill(identity['pid'], signal.SIGTERM)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and process_state(identity, os.getuid()) != 'dead':
        time.sleep(.1)
    if process_state(identity, os.getuid()) != 'dead':
        raise RuntimeError('managed training root has not exited; inspect its bounded stop guard, do not start a replacement')


def completed_test_samples(text):
    return max((int(value) for value in re.findall(
        r'^test_timing samples=(\d+) update=\d+ total_seconds=', text, re.MULTILINE)), default=0)


def production_watch():
    """CPU-only monitor bound to an explicit launch; no mutable latest lookup."""
    import torch
    from villa_gpu_manager.processes import process_state
    settings = json.loads(CONTROL.read_text())['production_stop']
    folder = Path(settings['launch']).resolve()
    target = int(settings['samples'])
    if target <= 0:
        raise ValueError('sample stop target must be positive')
    identity = json.loads((folder / 'controller.json').read_text())
    print(f'stop monitor launch={folder} identity={identity} target_samples={target}', flush=True)
    while process_state(identity, os.getuid()) == 'alive':
        text = (folder / 'train.log').read_text()
        if completed_test_samples(text) >= target:
            run = Path(re.findall(r'run_dir=(\S+)', text)[-1])
            result = None
            try:
                # test_timing precedes saves. Wait for all current and best
                # files to finish serialization, including the final EMA best.
                raw = torch.load(run / 'snapshots/current.pt', map_location='cpu', weights_only=True, mmap=True)
                ready = int(raw['samples_seen']) >= target
                best = torch.load(run / 'snapshots/best.pt', map_location='cpu', weights_only=True, mmap=True)
                ready = ready and float(best['metric']) <= float(raw['metric'])
                for name, track in raw.get('ema', {}).get('tracks', {}).items():
                    current = torch.load(run / 'snapshots' / f'current_{name}.pt', map_location='cpu', weights_only=True, mmap=True)
                    best = torch.load(run / 'snapshots' / f'best_{name}.pt', map_location='cpu', weights_only=True, mmap=True)
                    ready = ready and current['step'] == raw['step'] and best['metric'] == track['best_metric']
                if ready:
                    result = dict(launch=str(folder), run=str(run), identity=identity,
                                  target_samples=target, snapshot_samples=raw['samples_seen'], step=raw['step'])
            except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as exc:
                print(f'waiting for snapshot serialization: {exc}', flush=True)
            if result is not None:
                stop_production_identity(identity)
                (folder / 'sample_stop_result.json').write_text(json.dumps(result, indent=2))
                print(f'stopped after complete snapshots: {result}', flush=True)
                return
        time.sleep(2)
    raise RuntimeError('training exited or identity became unverifiable before sample stop completed')


def production_status(stop=False):
    from villa_gpu_manager.processes import process_state
    control = json.loads(CONTROL.read_text())['production']
    config = json.loads((ROOT / control['config']).read_text())
    root = Path(config['training']['run_path']) / '.launches'
    folder = Path((root / 'latest').read_text())
    identity = json.loads((folder / 'controller.json').read_text())
    if stop:
        stop_production_identity(identity)
    print('production:', folder, 'state:', process_state(identity, os.getuid()))
    text = (folder / 'train.log').read_text(errors='replace')
    markers = [line for line in text.splitlines() if (
        line.startswith('fiber cooperative:')
        or line.startswith('fiber_trace_3d train:')
        or line.startswith('fiber_trace_3d resume:')
        or line.startswith('test_timing ')
        or line.startswith('samples=')
        or 'Traceback (most recent call last)' in line
        or line.startswith('RuntimeError:')
    )]
    print('\n'.join(markers[-80:]))


def audit_latest():
    """CPU-only verification of written snapshot variants and TensorBoard tags."""
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    folder = Path((OUTPUT / 'latest').read_text())
    text = (folder / 'train.log').read_text()
    run = Path(re.findall(r'run_dir=(\S+)', text)[-1])
    raw = torch.load(run / 'snapshots/current.pt', map_location='cpu', weights_only=True, mmap=True)
    variants = {}
    for name, track in raw['ema']['tracks'].items():
        export = torch.load(run / 'snapshots' / f'current_{name}.pt',
                            map_location='cpu', weights_only=True, mmap=True)
        assert export['snapshot_kind'] == 'ema' and 'optimizer' not in export
        assert export['step'] == raw['step']
        for key, tensor in track['model'].items():
            assert torch.equal(tensor, export['model'][key]), (name, key)
        variants[name] = {key: track[key] for key in ('decay','updates','start_step','best_metric')}
    events = EventAccumulator(str(run), size_guidance={'scalars':0})
    events.Reload()
    tags = events.Tags()['scalars']
    for prefix in ('test','test_historical',*[f'test_{name}' for name in variants],
                   *[f'test_historical_{name}' for name in variants]):
        for metric in ('loss_total','angle_mean_deg'):
            assert prefix + '/' + metric in tags
    report = {'run':str(run),'step':raw['step'],'variants':variants,
              'test_tags':[tag for tag in tags if tag.startswith('test')]}
    (folder / 'snapshot_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def start():
    from villa_gpu_manager.bootstrap import ensure_manager
    from villa_gpu_manager.exclusive import ExclusiveTask
    from villa_gpu_manager.exclusive_command import ExclusiveCommand
    from villa_gpu_manager.stop_guard import StopGuard
    from villa_gpu_manager.processes import process_identity
    control = json.loads(CONTROL.read_text())
    config_path = ROOT / control['config']
    raw = json.loads(config_path.read_text())
    devices = control['devices']
    if raw['batch_size'] * len(devices) != raw['training']['optimizer_global_batch_size']:
        raise ValueError('Reproduction requires a physical effective batch, no accumulation')
    if (raw['training'].get('resume') or raw.get('resume') or
            raw['training'].get('initialize_from')):
        raise ValueError('Reproduction trials start from scratch')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    folder = OUTPUT / (control['stage'] + '-' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    folder.mkdir()
    (folder / 'config.json').write_text(json.dumps(raw, indent=2))
    (folder / 'inputs.json').write_text(json.dumps(input_manifest(raw), indent=2))
    (folder / 'control.json').write_text(json.dumps(control, indent=2))
    (folder / 'source.txt').write_text(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True)
                                    + subprocess.check_output(['git', 'diff'], cwd=ROOT, text=True))
    identity = process_identity(os.getpid())
    (folder / 'controller.json').write_text(json.dumps(identity))
    (OUTPUT / 'latest').write_text(str(folder))
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    socket, inventory = ensure_manager()
    uuids = subprocess.check_output(['nvidia-smi', '--id=' + ','.join(map(str, devices)),
        '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).splitlines()
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=' + str(len(devices)), '-m',
        'vesuvius.neural_tracing.fiber_trace_3d.train', str(folder / 'config.json'), '--unmanaged']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(uuids), PYTHONUNBUFFERED='1')
    (folder / 'command.json').write_text(json.dumps(command))
    print('experiment:', folder, flush=True)
    # Full inventory exclusivity: no other cooperative GPU task runs during trials.
    with ExclusiveTask(socket, inventory, label='reproduce-' + control['stage'],
                       priority=100, stop_event=stop) as lease, (folder / 'train.log').open('w') as log:
        child = ExclusiveCommand(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        lease.cohort = child
        lease.authorize()
        guard = StopGuard(grace_seconds=15)
        (folder / 'guard.json').write_text(json.dumps(process_identity(guard.process.pid)))
        registered = set()
        def observe():
            child.observe()
            for pid in child.identities:
                try:
                    if os.getsid(pid) == pid and pid not in registered:
                        guard.register([pid])
                        registered.add(pid)
                except ProcessLookupError:
                    pass
        def request_stop(*_):
            stop.set()
            guard.stop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, request_stop)
        observe()
        began = time.monotonic()
        try:
            while child.process.poll() is None:
                guard.check()
                observe()
                if stop.is_set():
                    raise InterruptedError('experiment stopped')
                if time.monotonic() - began > control.get('timeout_seconds', 14400):
                    raise TimeoutError('experiment time limit')
                lease.check()
                time.sleep(1)
            result = {'returncode': child.process.returncode, 'seconds': time.monotonic() - began}
            (folder / 'result.json').write_text(json.dumps(result, indent=2))
        finally:
            guard.stop()
            child.abort()
            guard.close()
    print('experiment complete:', result, flush=True)
    if result['returncode']:
        raise SystemExit(result['returncode'])


def status(stop=False):
    from villa_gpu_manager.processes import process_state
    folder = Path((OUTPUT / 'latest').read_text())
    identity = json.loads((folder / 'controller.json').read_text())
    state = process_state(identity, os.getuid())
    print(folder, state, flush=True)
    if stop and state == 'alive':
        os.kill(identity['pid'], signal.SIGTERM)
    if stop:
        deadline = time.monotonic() + 45
        while True:
            identities = [identity]
            if (folder / 'guard.json').exists():
                identities.append(json.loads((folder / 'guard.json').read_text()))
            survivors = [row for row in identities if process_state(row, os.getuid()) != 'dead']
            if not survivors:
                print('controller and cleanup guard exited', flush=True)
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f'cleanup not confirmed; ownership must be retained: {survivors}')
            time.sleep(.2)
    log = folder / 'train.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 6000))
            print(stream.read().decode(errors='replace'))


def summarize():
    plot_rows = []
    for folder in sorted(OUTPUT.glob('repro*')):
        log = folder / 'train.log'
        if not log.exists():
            continue
        rows = []
        for match in re.finditer(r'^test samples=(\d+) update=(\d+).*?angle_mean_deg=([\d.]+)',
                                 log.read_text(), re.MULTILINE):
            rows.append({'samples':int(match[1]), 'update':int(match[2]), 'angle':float(match[3])})
        window = [r for r in rows if 200000 <= r['samples'] <= 300000]
        report = {'experiment':folder.name, 'latest':rows[-1:] or None, 'window':window,
                  'window_mean':statistics.mean(r['angle'] for r in window) if window else None,
                  'window_median':statistics.median(r['angle'] for r in window) if window else None,
                  'first_under20':next((r for r in rows if r['angle'] < 20), None),
                  'result':json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None}
        (folder / 'summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        plot_rows.append((folder.name, rows))
    if plot_rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        historical = json.loads((OUTPUT / 's1a_128_1_single_8x8_20260729_075013.scalars.json').read_text())
        old = [r for r in historical['test/angle_mean_deg'] if r['step'] * 64 <= 300800]
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot([r['step']*64 for r in old], [r['value'] for r in old], label='Historical Jul29: CP test')
        for name, rows in plot_rows:
            ax.plot([r['samples'] for r in rows], [r['angle'] for r in rows], label=name.split('-2026')[0])
        ax.axhline(20, color='gray', linestyle='--')
        ax.axvspan(200000,300800,color='gray',alpha=.1)
        ax.set(xlabel='Global training samples', ylabel='Historical test mean angle (degrees)', xlim=(0,300800))
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT / 'convergence.svg')
        plt.close(fig)


def evaluate():
    """Use the existing managed benchmark, with immutable per-ablation settings."""
    from vesuvius.neural_tracing.fiber_trace_3d.evaluation_benchmark import main as benchmark
    options = json.loads(CONTROL.read_text())['evaluation']
    raw = json.loads((ROOT / options['config']).read_text())
    def merge(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                merge(target[key], value)
            else:
                target[key] = value
    merge(raw, options.get('overrides', {}))
    folder = OUTPUT / ('eval-' + options['name'] + '-' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(parents=True)
    config = folder / 'config.json'
    config.write_text(json.dumps(raw, indent=2))
    (folder / 'control.json').write_text(json.dumps(options, indent=2))
    (folder / 'source.txt').write_text(subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True))
    argv = ['--reference',str(config),'--checkpoint',options['checkpoint'],
            '--output',str(folder),'--backend','pipeline','--devices','all',
            '--samples',str(options.get('samples',128)),
            '--batch-size',str(options.get('batch_size',8)),
            '--passes','1','--cpu-workers','32','--timeout','300']
    (folder / 'command.json').write_text(json.dumps(argv))
    print('evaluation:',folder, flush=True)
    result = benchmark(argv)
    for report in folder.glob('evaluation-*.json'):
        data = json.loads(report.read_text())
        rows = [row for rank in data['ranks'] for pass_record in rank for row in pass_record['metrics']]
        metrics = {key:statistics.mean(row[1][key] for row in rows) for key in rows[0][1]}
        (folder / 'summary.json').write_text(json.dumps(metrics, indent=2))
        print('metrics:',json.dumps(metrics), flush=True)
    return result


def cross_section_series():
    """Evaluate configured raw/EMA snapshots under one managed GPU grant."""
    from vesuvius.neural_tracing.fiber_trace_3d.evaluation_benchmark import main as benchmark

    settings = json.loads(CONTROL.read_text())["cross_section_series"]
    run = Path(settings["run"])
    variants = settings.get(
        "variants",
        ["best.pt", "current.pt", "best_ema_0p99.pt", "current_ema_0p99.pt",
         "best_ema_0p999.pt", "current_ema_0p999.pt"],
    )
    source_checkpoints = [run / "snapshots" / name for name in variants]
    missing = [str(path) for path in source_checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing configured cross-section snapshots: {missing}")
    output = Path(settings["output"])
    if output.exists():
        raise FileExistsError(f"cross-section output already exists: {output}")
    output.mkdir(parents=True)
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import freeze_checkpoint
    frozen_root = output / "frozen"
    frozen_root.mkdir()
    selections = []
    checkpoints = []
    import torch
    for source in source_checkpoints:
        frozen, digest = freeze_checkpoint(source, frozen_root)
        target = frozen_root / source.name
        os.link(frozen, target)
        payload = torch.load(target, map_location="cpu", weights_only=False, mmap=True)
        refinement_state = payload.get("refinement_state")
        if refinement_state:
            archive_source = source.parent / refinement_state["file"]
            archive_target = frozen_root / refinement_state["file"]
            if not archive_target.exists():
                os.link(archive_source, archive_target)
        checkpoints.append(target)
        selections.append({
            "source": str(source),
            "frozen": str(target),
            "sha256": digest,
            "step": int(payload.get("step", 0)),
            "samples_seen": payload.get("samples_seen"),
            "metric": payload.get("metric"),
            "metric_name": payload.get("metric_name"),
            "refinement_state": refinement_state,
        })
    (output / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    argv = [
        "--evaluation-only",
        "--reference", str((ROOT / settings["config"]).resolve()),
        "--output", str(output),
        "--backend", "pipeline",
        "--devices", str(settings.get("devices", "all")),
        "--samples", str(settings.get("samples", 0)),
        "--batch-size", str(settings.get("batch_size", 8)),
        "--passes", "1",
        "--cpu-workers", str(settings.get("cpu_workers", 128)),
        "--timeout", str(settings.get("timeout_seconds", 3600)),
        "--cross-section",
        "--cross-section-width-base", str(settings.get("width_base", 160)),
        "--cross-section-height-base", str(settings.get("height_base", 80)),
        "--cross-section-histogram-bins", str(settings.get("histogram_bins", 1024)),
        "--checkpoint-series", *map(str, checkpoints),
    ]
    return benchmark(argv)


def cross_section_rerender():
    """Re-render configured cross sections from completed histogram shards."""
    from dataclasses import replace
    from vesuvius.neural_tracing.fiber_trace_3d.cross_section import (
        merge_shards,
        write_cross_section_artifacts,
    )

    settings = json.loads(CONTROL.read_text())["cross_section_series"]
    output = Path(settings["output"])
    variants = settings.get(
        "variants",
        ["best.pt", "current.pt", "best_ema_0p99.pt", "current_ema_0p99.pt",
         "best_ema_0p999.pt", "current_ema_0p999.pt"],
    )
    for variant in variants:
        label = Path(variant).stem
        shards = sorted((output / "shards").glob(f"{label}-rank*.npz"))
        if not shards:
            raise FileNotFoundError(f"no completed cross-section shards for {label}")
        directory = output / label
        old_metadata = json.loads((directory / "cross_section.json").read_text())
        metadata = {
            key: old_metadata[key]
            for key in ("checkpoint", "base_volume_scale", "samples_requested")
            if key in old_metadata
        }
        accumulator = merge_shards(shards)
        accumulator.config = replace(
            accumulator.config,
            quantiles=tuple(settings.get("exceedance_fractions", (0.001, 0.01, 0.1, 0.2, 0.5))),
        )
        paths = write_cross_section_artifacts(accumulator, directory, metadata=metadata)
        print(f"cross-section rerendered label={label} files={paths}", flush=True)


def inspect():
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = [HISTORICAL / name for name in ['s1a_128_1_single_8x8_20260729_075013',
                 's1a_128_1_single_8x8_20260730_063937']]
    runs.append(Path('/ephemeral/hendrik/fiber3d_train2/runs/s1a_nml_128_sd1_16x8_20260912_062932'))
    for run in runs:
        name = run.name
        if (OUTPUT / f'{name}.scalars.json').exists():
            continue
        checkpoint = torch.load(run / 'snapshots/current.pt', map_location='cpu',
                                weights_only=False, mmap=True)
        metadata = {k: v for k, v in checkpoint.items()
                    if k not in ('model', 'model_state_dict', 'optimizer', 'optimizer_state_dict',
                                 'grad_scaler', 'rng_state')}
        # Tensor-valued training state is not copied into the inspection report.
        (OUTPUT / f'{name}.checkpoint.json').write_text(json.dumps(metadata, indent=2, default=str))
        event = EventAccumulator(str(run), size_guidance={'scalars': 0})
        event.Reload()
        scalars = {tag: [dict(step=x.step, value=x.value, wall_time=x.wall_time)
                         for x in event.Scalars(tag)] for tag in event.Tags()['scalars']}
        (OUTPUT / f'{name}.scalars.json').write_text(json.dumps(scalars))
        print(name, 'checkpoint keys=', list(checkpoint), 'tags=', list(scalars), flush=True)
    print('inspection saved:', OUTPUT)


def main():
    if sys.argv[1:] == ['annotation-report']:
        # CPU-only inspection of the last short annotation training. No GPU use.
        import math
        import torch
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
        torch.set_num_threads(1)
        config = json.loads((ROOT / json.loads(CONTROL.read_text())['production']['config']).read_text())
        training = config['training']
        run = sorted(Path(training['run_path']).glob(training['run_name'] + '_*'))[-1]
        checkpoint = torch.load(run / 'snapshots/current.pt', map_location='cpu', mmap=True, weights_only=False)
        model = build_fiber_trace_3d_model(checkpoint['config'])
        model.load_state_dict(checkpoint['model'], strict=True)
        assert model.annotations.channels == 8
        events = EventAccumulator(str(run), size_guidance={'scalars': 0}).Reload()
        report = {'run': str(run), 'checkpoint_step': checkpoint['step'], 'metrics': {}}
        assert not checkpoint['config']['refinement']['enabled']
        for prefix in ('train', 'test', 'test_ema_0p99', 'test_ema_0p999'):
            for name in ('normal', 'width', 'hv'):
                tag = prefix + '/loss_' + name
                values = events.Scalars(tag)
                assert values and all(math.isfinite(x.value) for x in values), tag
                report['metrics'][tag] = [{'samples': x.step, 'value': x.value} for x in values]
        print(json.dumps(report, indent=2), flush=True)
        return
    if sys.argv[1:] == ['embedding-watch-start']:
        from villa_gpu_manager.processes import process_identity
        control = json.loads(CONTROL.read_text())
        root = Path(control['embedding_milestone']['output'])
        root.mkdir(parents=True, exist_ok=False)
        frozen_control = root / 'control.json'
        frozen_control.write_text(json.dumps(control, indent=2))
        with (root / 'watch.log').open('w') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m',
                'vesuvius.neural_tracing.fiber_trace_3d.embedding_milestone', str(frozen_control)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (root / 'watcher.json').write_text(json.dumps(process_identity(process.pid)))
        print(f'watcher pid={process.pid} log={root / "watch.log"}', flush=True)
        return
    if sys.argv[1:] in (['embedding-stop-pilot'], ['embedding-stop-sweep']):
        from villa_gpu_manager.processes import process_identity
        module = ('embedding_trace_sweep' if sys.argv[1] == 'embedding-stop-sweep'
                  else 'embedding_trace_pilot')
        matches = []
        for proc in Path('/proc').glob('[0-9]*'):
            try:
                args = (proc / 'cmdline').read_bytes().split(b'\0')
                if (proc.stat().st_uid == os.getuid() and b'-m' in args and
                    ('vesuvius.neural_tracing.fiber_trace_3d.' + module).encode() in args and
                    str(CONTROL).encode() in args and b'--unmanaged' not in args):
                    matches.append(int(proc.name))
            except (FileNotFoundError, ProcessLookupError):
                continue
        for pid in matches:
            stop_production_identity(process_identity(pid))
        print(f'stopped managed {module} roots: {matches}', flush=True)
        return
    if sys.argv[1:] == ['embedding-freeze-current']:
        from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import freeze_checkpoint
        import torch
        control = json.loads(CONTROL.read_text())
        settings = control['embedding_trace_pilot']
        frozen, digest = freeze_checkpoint(Path(settings['checkpoint']), Path(settings['output'] + '-frozen'))
        payload = torch.load(frozen, map_location='cpu', weights_only=True, mmap=True)
        settings['checkpoint'] = str(frozen)
        CONTROL.write_text(json.dumps(control, indent=2) + '\n')
        print(json.dumps(dict(checkpoint=str(frozen), sha256=digest, step=payload['step'], samples_seen=payload.get('samples_seen'))), flush=True)
        return
    if sys.argv[1:] == ['embedding-trace-sweep']:
        raise SystemExit(subprocess.call([sys.executable, '-m',
            'vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_sweep', str(CONTROL)]))
    if sys.argv[1:] == ['embedding-route-analysis']:
        raise SystemExit(subprocess.call([sys.executable, '-m',
            'vesuvius.neural_tracing.fiber_trace_3d.embedding_route_analysis', str(CONTROL)]))
    if sys.argv[1:] == ['embedding-trace-pilot']:
        raise SystemExit(subprocess.call([sys.executable, '-m',
            'vesuvius.neural_tracing.fiber_trace_3d.embedding_trace_pilot', str(CONTROL)]))
    if sys.argv[1:] == ['embedding-histograms']:
        return subprocess.call([sys.executable, '-m',
            'vesuvius.neural_tracing.fiber_trace_3d.embedding_histograms', str(CONTROL)])
    if sys.argv[1:] == ['production-embedding-metrics']:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        config = json.loads((ROOT / json.loads(CONTROL.read_text())['production']['config']).read_text())
        launch = Path((Path(config['training']['run_path']) / '.launches/latest').read_text())
        run = Path(re.findall(r'run_dir=(\S+)', (launch/'train.log').read_text())[-1])
        events = EventAccumulator(str(run), size_guidance={'scalars':0})
        events.Reload()
        values = {}
        for prefix in ('test', 'test_ema_0p99', 'test_ema_0p999'):
            for metric in ('positive_mean', 'positive_patch_p05', 'negative_mean', 'negative_patch_p95',
                           'negative_patch_p99', 'negative_above_0p9', 'hardest_negative_mean', 'positive_hardest_gap'):
                key = f'{prefix}/embedding_similarity/{metric}'
                row = events.Scalars(key)[-1]
                values[key] = dict(step=row.step, value=row.value)
        print(json.dumps(dict(run=str(run), metrics=values), indent=2))
        return
    if sys.argv[1:] == ['embedding-consistency-report']:
        return subprocess.call([sys.executable, '-m',
            'vesuvius.neural_tracing.fiber_trace_3d.embedding_consistency', str(CONTROL), '--report'])
    if sys.argv[1:] == ['embedding-consistency']:
        return subprocess.call([sys.executable, '-m',
            'vesuvius.neural_tracing.fiber_trace_3d.embedding_consistency', str(CONTROL)])
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['inspect', 'start', 'status', 'stop', 'summarize', 'inputs', 'evaluate', 'cross-section-series', 'cross-section-rerender', 'audit-latest', 'production-start', 'production-status', 'production-stop', 'production-watch', 'trace-evaluate', 'comparison', 'comparison-followup', 'trace-series-prepare', 'trace-series-next', 'trace-series-retry', 'remote-auth-probe'])
    parser.add_argument('--resume', type=Path,
                        help='checkpoint passed to production-start')
    args = parser.parse_args()
    if args.command == 'inspect':
        inspect()
    elif args.command == 'production-watch':
        production_watch()
    elif args.command == 'trace-evaluate':
        trace_evaluate()
    elif args.command == 'comparison':
        comparison()
    elif args.command == 'comparison-followup':
        comparison_followup()
    elif args.command == 'trace-series-prepare':
        trace_series_prepare()
    elif args.command == 'trace-series-next':
        trace_series_next()
    elif args.command == 'trace-series-retry':
        trace_series_next(retry_failed=True)
    elif args.command == 'production-start':
        production_start(resume=args.resume)
    elif args.command == 'production-status':
        production_status()
    elif args.command == 'production-stop':
        production_status(stop=True)
    elif args.command == 'remote-auth-probe':
        remote_auth_probe()
    elif args.command == 'audit-latest':
        audit_latest()
    elif args.command == 'start':
        start()
    elif args.command == 'summarize':
        summarize()
    elif args.command == 'inputs':
        folder = Path((OUTPUT / 'latest').read_text())
        result = input_manifest(json.loads((folder / 'config.json').read_text()))
        (folder / 'inputs.json').write_text(json.dumps(result, indent=2))
        print({key: len(rows) for key, rows in result.items()})
    elif args.command == 'evaluate':
        evaluate()
    elif args.command == 'cross-section-series':
        cross_section_series()
    elif args.command == 'cross-section-rerender':
        cross_section_rerender()
    else:
        status(stop=args.command == 'stop')


if __name__ == '__main__':
    main()
