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


def trace_evaluate(settings=None):
    """Freeze selected raw/EMA weights and reuse the locked managed evaluator."""
    import math
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import freeze_checkpoint
    if settings is None:
        settings = json.loads(CONTROL.read_text())['trace_evaluation']
    run = Path(settings['run'])
    paths = [run / 'snapshots/best.pt'] if settings['variant'] == 'raw' else sorted((run / 'snapshots').glob('best_ema_*.pt'))
    if settings['variant'] not in ('raw', 'ema') or not paths:
        raise ValueError('need raw or ema selection and corresponding best snapshots')
    candidates = []
    for path in paths:
        payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
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
    workload = json.loads((ROOT / settings['template']).read_text())
    workload.update(checkpoint=str(frozen), checkpoint_sha256=digest, output_root=str(output))
    workload['evaluation']['inference_cache_namespace'] = output.name
    config = output / 'workload.json'
    config.write_text(json.dumps(workload, indent=2))
    (output / 'selection.json').write_text(json.dumps(dict(run=str(run), candidates=candidates,
        selected=selected, frozen=str(frozen), sha256=digest), indent=2))
    print(f'trace selection {selected} sha256={digest} output={output}', flush=True)
    script = ROOT / 'vesuvius/scripts/run_classic_fiber_evaluation.py'
    for mode in ('prepare', 'validate-full', 'full'):
        print(f'trace {mode}: log={output / (mode + ".log")}', flush=True)
        with (output / (mode + '.log')).open('w') as log:
            subprocess.run([sys.executable, str(script), mode, '--config', str(config)],
                           cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    return output


def production_start(config_path=None, resume=None):
    """Start the ordinary adaptive trainer, with no outer exclusive reservation."""
    from villa_gpu_manager.processes import process_identity, process_state
    if config_path is None:
        config_path = json.loads(CONTROL.read_text())['production']['config']
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
    with (folder / 'train.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED='1'),
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
    with (folder / 'train.log').open('rb') as stream:
        stream.seek(max(0, os.fstat(stream.fileno()).st_size - 10000))
        print(stream.read().decode(errors='replace'))


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
    if raw['training'].get('resume') or raw.get('resume'):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['inspect', 'start', 'status', 'stop', 'summarize', 'inputs', 'evaluate', 'audit-latest', 'production-start', 'production-status', 'production-stop', 'production-watch', 'trace-evaluate', 'comparison'])
    args = parser.parse_args()
    if args.command == 'inspect':
        inspect()
    elif args.command == 'production-watch':
        production_watch()
    elif args.command == 'trace-evaluate':
        trace_evaluate()
    elif args.command == 'comparison':
        comparison()
    elif args.command == 'production-start':
        production_start()
    elif args.command == 'production-status':
        production_status()
    elif args.command == 'production-stop':
        production_status(stop=True)
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
    else:
        status(stop=args.command == 'stop')


if __name__ == '__main__':
    main()
