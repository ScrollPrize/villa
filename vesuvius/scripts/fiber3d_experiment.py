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
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path('/ephemeral/hendrik/fiber3d_train2/reproduction')
HISTORICAL = Path('/ephemeral/hendrik/fiber3d_train/runs')
CONTROL = Path(__file__).with_name('fiber3d_experiment.json')


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
    log = folder / 'train.log'
    if log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size - 6000))
            print(stream.read().decode(errors='replace'))


def summarize():
    for folder in sorted(OUTPUT.glob('repro*')):
        log = folder / 'train.log'
        if not log.exists():
            continue
        rows = []
        for match in re.finditer(r'^test samples=(\d+) update=(\d+).*?angle_mean_deg=([\d.]+)',
                                 log.read_text(), re.MULTILINE):
            rows.append({'samples':int(match[1]), 'update':int(match[2]), 'angle':float(match[3])})
        window = [r for r in rows if 200000 <= r['samples'] <= 300800]
        report = {'experiment':folder.name, 'latest':rows[-1:] or None, 'window':window,
                  'window_mean':statistics.mean(r['angle'] for r in window) if window else None,
                  'window_median':statistics.median(r['angle'] for r in window) if window else None,
                  'first_under20':next((r for r in rows if r['angle'] < 20), None),
                  'result':json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None}
        (folder / 'summary.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)


def inspect():
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ['s1a_128_1_single_8x8_20260729_075013',
                 's1a_128_1_single_8x8_20260730_063937']:
        run = HISTORICAL / name
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
    parser.add_argument('command', choices=['inspect', 'start', 'status', 'stop', 'summarize'])
    args = parser.parse_args()
    if args.command == 'inspect':
        inspect()
    elif args.command == 'start':
        start()
    elif args.command == 'summarize':
        summarize()
    else:
        status(stop=args.command == 'stop')


if __name__ == '__main__':
    main()
