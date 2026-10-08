"""Stop a named trainer, its descendants, and its orphaned DAgger collectors."""
import json
from pathlib import Path
import sys

import psutil

PREFIX = 'vesuvius.neural_tracing.fiber_follow.'
TRAINERS = (PREFIX+'train.train',)
TRAINER_SCRIPT = 'fiber_follow/train/train.py'
COLLECTORS = (PREFIX+'tracing.collect',)


def option(args, key):
    try:
        return args[args.index(key)+1]
    except (ValueError, IndexError):
        return None


def trainer_matches(args, name, cwd='.'):
    """A trainer (``train/train.py`` or ``-m ...train.train``) whose run configuration is named ``name``."""
    if option(args, '-m') not in TRAINERS and not any(str(a).endswith(TRAINER_SCRIPT) for a in args):
        return False
    config = option(args, '--config')
    if config is None:
        return False
    try:
        return json.loads((Path(cwd)/config).read_text()).get('name') == name
    except (OSError, ValueError):
        return False


def collector_matches(args, run_dir):
    if option(args, '-m') not in COLLECTORS:
        return False
    root = (Path(run_dir)/'dagger').resolve()
    for key in ('--checkpoint', '--out'):
        value = option(args, key)
        if value is None or not Path(value).is_absolute() or not Path(value).resolve().is_relative_to(root):
            return False
    return True


def matching_collectors(run_dir):
    result = []
    for process in psutil.process_iter(['cmdline']):
        try:
            if collector_matches(process.info['cmdline'] or [], run_dir):
                result.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def stop_run(pid_file, name, run_dir=None):
    """``pid_file``: the launcher's ``output/logs/NAME.pid`` or the trainer's ``output/NAME/trainer.pid``."""
    if name in ('', '.', '..') or Path(name).name != name:
        raise ValueError('Run name must be a single directory name')
    pid_file = Path(pid_file)
    if run_dir is None:
        run_dir = pid_file.parent if pid_file.name == 'trainer.pid' else pid_file.parent.parent/name
    roots = []
    try:
        parent = psutil.Process(int(pid_file.read_text().strip()))
        args, cwd = parent.cmdline(), parent.cwd()
    except (FileNotFoundError, psutil.NoSuchProcess):
        parent = None
    if parent is not None:
        if not trainer_matches(args, name, cwd):
            raise RuntimeError(f'Refusing to stop PID {parent.pid}: it does not match run {name}')
        parent.suspend()  # No new collectors while we scan for orphans.
        roots.append(parent)
    owned = {}
    try:
        roots.extend(matching_collectors(run_dir))
        pending = list(roots)
        while pending:
            process = pending.pop()
            if process.pid in owned:
                continue
            try:
                process.suspend()  # Freeze each generation before enumerating children.
                owned[process.pid] = process
                pending.extend(process.children())
            except psutil.NoSuchProcess:
                continue
        for process in owned.values():
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                pass
    finally:
        for process in {p.pid:p for p in [*roots, *owned.values()]}.values():
            try:
                process.resume()
            except psutil.NoSuchProcess:
                pass
    _, remaining = psutil.wait_procs(list(owned.values()), timeout=5)
    for process in remaining:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(remaining, timeout=5)
    remaining = [p.pid for p in remaining if alive(p)]
    leftovers = [p.pid for p in matching_collectors(run_dir) if alive(p)]
    if remaining or leftovers:
        raise RuntimeError(f'Run processes still alive: {sorted(set(remaining+leftovers))}')
    print(f'Stopped {name}: {len(owned)} processes; verified no matching DAgger collectors remain', flush=True)
    return list(owned)


if __name__ == '__main__':
    stop_run(*sys.argv[1:])
