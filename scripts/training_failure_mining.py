"""Persistent launcher for the configured training extraction; no scheduler changes."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import time
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor

REPO = Path(__file__).resolve().parents[1]
ROOT = Path('/ephemeral/hendrik/fiber3d_train2/evaluations/train_failure_union_20260927_v1')
CONFIGS = REPO / 'vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs'
RUNNER = REPO / 'vesuvius/scripts/run_classic_fiber_evaluation.py'
sys.path[:0] = [str(REPO / 'vesuvius/src'), str(REPO), str(REPO / 'gpu_manager/src')]


def status(stage, **extra):
    from lasagna.inference_provenance import atomic_write
    atomic_write(ROOT / 'workflow_status.json', dict(stage=stage, pid=os.getpid(),
        updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), durable=False)


def archive_incomplete_output(output):
    """Preserve interrupted traces; published inference lives outside this path."""
    output = Path(output)
    if not output.exists():
        return None
    if output.is_symlink() or not output.is_dir():
        raise RuntimeError(f'refusing unexpected evaluation output: {output}')
    summary = output / 'summary.json'
    if summary.exists() and json.loads(summary.read_text()).get('completed'):
        raise RuntimeError(f'refusing to archive completed evaluation: {output}')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = output.with_name(f'{output.name}.interrupted.{stamp}')
    if destination.exists():
        raise RuntimeError(f'archive already exists: {destination}')
    output.rename(destination)
    return destination


def discard_invalid_reproduction(root):
    """Invalidate only protocol-v1 trial artifacts; never traverse cache roots."""
    root = Path(root)
    source = root/'reproduction_search'
    discarded = root/'discarded_reproduction_local_arcs_v1'
    if source.is_symlink() or not source.is_dir() or discarded.exists():
        raise RuntimeError('unexpected reproduction directory; refusing cleanup')
    protocol = json.loads((source/'protocol.json').read_text())
    if protocol['protocol']['version'] != 1:
        raise RuntimeError('refusing to discard corrected reproduction protocol')
    from lasagna.inference_provenance import atomic_write
    atomic_write(root/'discarded_reproduction_report.json', dict(
        reason='segment-local arcs incorrectly applied to full original fiber',
        discarded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_protocol=protocol, inference_caches_preserved=True), durable=False)
    source.rename(discarded)
    return discarded


def cleanup_discarded_reproduction():
    from vesuvius.neural_tracing.fiber_trace_3d.failure_trial_audit import bounded_map
    discarded = ROOT/'discarded_reproduction_local_arcs_v1'
    if discarded.is_symlink() or not discarded.is_dir():
        raise RuntimeError('unexpected discarded reproduction path')
    # Parallelize independent case directories, not the cache or original GT.
    with ThreadPoolExecutor(max_workers=128) as pool:
        for cases in discarded.glob('batches/*/attempt*/full/cases'):
            if (not cases.resolve().is_relative_to(discarded.resolve()) or
                    any(p.is_symlink() for p in (cases, *cases.parents) if p != discarded.parent and discarded in (p, *p.parents))):
                raise RuntimeError(f'refusing linked directory in discarded results: {cases}')
            for _ in bounded_map(pool, shutil.rmtree, (p for p in cases.iterdir() if p.is_dir() and not p.is_symlink())):
                pass
    shutil.rmtree(discarded)
    print('Deleted invalid reproduction trials/audits; inference caches untouched', flush=True)


def worker():
    control = json.loads((REPO / 'scripts/training_failure_mining_control.json').read_text())
    devices = control['devices']
    lock = (ROOT / 'workflow.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Training mining is already running')
    try:
        names = control.get('evaluation_names', ['classic', 'lowres', 'fullres'])
        if not names or any(name not in ('classic', 'lowres', 'fullres') for name in names):
            raise ValueError('invalid evaluation_names')
        for name in names:
            config = CONFIGS / f'eval_train_mining_{name}_20260927.json'
            summary = ROOT / 'evaluations' / name / 'full/summary.json'
            if summary.exists() and json.loads(summary.read_text()).get('completed'):
                print(f'Reusing completed {name}: {summary}', flush=True)
                continue
            if name in control.get('resume_incomplete_evaluations', []):
                archived = archive_incomplete_output(summary.parent)
                if archived:
                    print(f'Preserved interrupted {name}: {archived}; reusing shared inference cache', flush=True)
            # Preserve the first failed preflight audit, which included three
            # single-CP fibers with no eligible reference span. No tracing ran.
            audit = summary.parent.parent / 'full_bindings.json'
            if name == 'classic' and control.get('archive_undersized_accumulator_attempt') and summary.parent.exists() and not summary.exists():
                archive = audit.parent / 'stopped_undersized_accumulator'
                archive.mkdir(exist_ok=False)
                for filename in ('full_bindings.json', 'full.json', 'full'):
                    (audit.parent / filename).rename(archive / filename)
                print(f'Preserved stopped undersized-accumulator attempt: {archive}', flush=True)
            if name == 'classic' and (summary.parent / 'native.log').exists() and not summary.exists():
                text = (summary.parent / 'native.log').read_text()
                if 'live case count exceeds explicit stack-preserving thread budget' in text:
                    archive = audit.parent / 'failed_thread_per_case_startup'
                    archive.mkdir(exist_ok=False)
                    for filename in ('full_bindings.json', 'full.json', 'full'):
                        (audit.parent / filename).rename(archive / filename)
                elif 'reference replay result interval does not match its case' in text:
                    # Preserve the failed attempt; published inference is in
                    # the shared cache, outside this evaluation output.
                    archive = audit.parent / 'failed_interval_roundoff'
                    archive.mkdir(exist_ok=False)
                    for filename in ('full_bindings.json', 'full.json', 'full'):
                        (audit.parent / filename).rename(archive / filename)
                    print(f'Archived failed interval-check attempt: {archive}', flush=True)
            if name == 'classic' and audit.exists() and not summary.exists():
                old = json.loads(audit.read_text())
                if any('excluded_fibers' not in s for s in old['sources']):
                    for filename in ('full_bindings.json', 'full.json'):
                        source = audit.parent / filename
                        destination = source.with_suffix('.pre_eligibility.json')
                        if destination.exists():
                            raise RuntimeError(f'archive already exists: {destination}')
                        source.rename(destination)
            for mode in ('prepare', 'validate-full', 'full'):
                status(f'{name}:{mode}')
                print(f'[{datetime.datetime.now(datetime.timezone.utc).isoformat()}] {name}:{mode}', flush=True)
                subprocess.run([sys.executable, str(RUNNER), mode, '--config', str(config),
                                '--devices', devices],
                               cwd=REPO, check=True)
        status('failure-union')
        # Union invokes the existing reproduction scheduler directly. Give the
        # entire stage one normal manager grant, just like short-failures CLI.
        from vesuvius.neural_tracing.fiber_trace_3d.managed_evaluation import launch_managed
        base = json.loads((CONFIGS / 'eval_train_mining_lowres_20260927.json').read_text())
        base['evaluation']['gpu_manager']['cpu_cores'] = control.get(
            'cpu_cores', base['evaluation']['gpu_manager']['cpu_cores'])
        result = launch_managed([sys.executable, str(RUNNER), 'failure-union', '--config',
            str(CONFIGS / 'failure_union_training_20260927.json')], devices=devices,
            settings=base['evaluation']['gpu_manager'])
        if result:
            raise RuntimeError(f'failure-union exited with {result}')
        status('completed')
    except BaseException as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ['--worker']:
        worker()
    elif sys.argv[1:] == ['--cleanup-invalid-reproduction']:
        cleanup_discarded_reproduction()
    elif not sys.argv[1:]:
        control = REPO / 'scripts/training_failure_mining_control.json'
        action = json.loads(control.read_text()).get('action', 'start') if control.exists() else 'start'
        if action == 'discard_invalid_reproduction':
            with (ROOT/'workflow.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                discarded = discard_invalid_reproduction(ROOT)
            with (ROOT/'discarded_reproduction_cleanup.log').open('a') as log:
                process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                    '--cleanup-invalid-reproduction'], cwd=REPO, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            print(f'Invalid results removed from active dataset: {discarded}; deletion PID {process.pid}')
            raise SystemExit(0)
        if action == 'check_ephemeral_health':
            subprocess.run([sys.executable, '-u', str(REPO / 'scripts/check_ephemeral_health.py')], check=True)
            raise SystemExit(0)
        if action == 'reap_stopped_workflow':
            session = int(json.loads(control.read_text())['stopped_session'])
            if session <= 1 or Path(f'/proc/{session}').exists():
                raise RuntimeError('stopped workflow leader must be gone')
            targets = []
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                    if int(stat[3]) != session or stat[0] == 'Z':
                        continue
                    command = (proc / 'cmdline').read_bytes()
                    if (proc.stat().st_uid != os.getuid() or int(stat[1]) != 1
                            or b'multiprocessing.' not in command):
                        raise RuntimeError(f'unexpected surviving workflow process {proc.name}')
                    targets.append((int(proc.name), stat[19]))
                except FileNotFoundError:
                    continue
            for pid, start in targets:
                try:
                    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                    if stat[19] != start or int(stat[3]) != session:
                        raise RuntimeError('process identity changed during orphan cleanup')
                    os.kill(pid, signal.SIGTERM)
                except FileNotFoundError:
                    pass
            print(f'Terminated {len(targets)} verified orphan workers from stopped session {session}')
            raise SystemExit(0)
        if action == 'reap_previous_gpu_workers':
            # Exact orphaned children observed from the stopped Classic run.
            # Never touch another job or a recycled process ID.
            for pid, handle in ((643356, 1068), (643357, 1072), (643358, 1076), (643359, 1080)):
                proc = Path(f'/proc/{pid}')
                if not proc.exists():
                    continue
                command = (proc / 'cmdline').read_bytes()
                stat = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
                if (int(stat[1]) != 1 or int(stat[2]) != 633076
                        or b'multiprocessing.spawn' not in command
                        or f'pipe_handle={handle}'.encode() not in command):
                    raise RuntimeError(f'refusing to signal unexpected process {pid}')
                os.kill(pid, signal.SIGTERM)
                print(f'Terminated verified old extraction worker {pid}', flush=True)
            raise SystemExit(0)
        if action == 'build':
            subprocess.run(['/usr/bin/g++-13', '-std=c++20', '-pthread', '-O2',
                str(REPO / 'volume-cartographer/apps/src/test_fiber_case_queue.cpp'),
                '-o', '/tmp/villa3_test_fiber_case_queue'], check=True)
            subprocess.run(['/tmp/villa3_test_fiber_case_queue'], check=True)
            subprocess.run(['bash', str(REPO / 'vesuvius/scripts/run_fiber3d_native_build')], check=True)
            for test in ('test_fiber_replay', 'test_fiber_trace3d', 'test_fiber_reference_replay_benchmark'):
                subprocess.run([str(REPO / 'volume-cartographer/build/fiber-evaluation/bin' / test)], check=True)
            subprocess.run([sys.executable, '-m', 'pytest', '-o', f'pythonpath={REPO}/vesuvius/src', '-q',
                'vesuvius/tests/neural_tracing/test_native_replay_evaluation.py',
                '-k', 'global_failure_trial'], cwd=REPO, check=True,
                env={**os.environ, 'VC_REPLAY_TEST_BINARY': str(REPO/'volume-cartographer/build/fiber-evaluation/bin/vc_fiber_trace_metric'),
                     'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'})
            raise SystemExit(0)
        if action == 'benchmark_export':
            subprocess.run([sys.executable, str(REPO/'scripts/benchmark_failure_export.py')],
                cwd=REPO, env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}, check=True)
            raise SystemExit(0)
        if action == 'validate_trial_geometry':
            subprocess.run([sys.executable, str(REPO/'scripts/validate_failure_trial_geometry.py')],
                cwd=REPO, env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}, check=True)
            raise SystemExit(0)
        if action == 'stop':
            state = json.loads((ROOT / 'workflow_status.json').read_text())
            pid = state['pid']
            proc = Path(f'/proc/{pid}/cmdline')
            if proc.exists():
                command = proc.read_bytes().split(b'\0')
                if str(Path(__file__).resolve()).encode() not in command or b'--worker' not in command:
                    raise RuntimeError('refusing to stop unexpected PID')
                if os.getpgid(pid) != pid:
                    raise RuntimeError('workflow process group identity mismatch')
                os.killpg(pid, signal.SIGINT)
                print(f'Sent SIGINT to workflow group {pid}', flush=True)
                for _ in range(30):
                    if not proc.exists():
                        break
                    time.sleep(1)
            raise SystemExit(0)
        if action != 'start':
            raise SystemExit('action must be start, stop or build')
        with (ROOT / 'workflow.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--worker'],
                cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Started workflow PID {process.pid}; log: {ROOT / "workflow.log"}')
    else:
        raise SystemExit('No arguments accepted')
