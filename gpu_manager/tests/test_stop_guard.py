"""CPU-only fault injection for client-owned termination, no manager/GPU."""
import json
import subprocess
import sys

import psutil


def test_guard_kills_stuck_root_and_worker_not_unregistered_service():
    code = r'''
import json, subprocess, sys, time, signal
from villa_gpu_manager.stop_guard import StopGuard
guard = StopGuard(grace_seconds=.1)
worker = subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print("ready",flush=True); time.sleep(60)'], stdout=subprocess.PIPE, text=True, start_new_session=True)
worker.stdout.readline()
service = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
guard.register([worker.pid])
print(json.dumps([worker.pid, service.pid, guard.process.pid]), flush=True)
guard.stop()
# Simulate a supervisor stuck in an uninterruptible Python/C cleanup wait.
time.sleep(60)
'''
    root = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    worker = service = guard = None
    try:
        worker, service, guard = json.loads(root.stdout.readline())
        root.wait(timeout=13)
        assert root.returncode < 0
        assert not psutil.pid_exists(worker) or psutil.Process(worker).status() == psutil.STATUS_ZOMBIE
        assert psutil.Process(service).is_running()
        assert psutil.Process(service).status() != psutil.STATUS_ZOMBIE
    finally:
        for pid in (root.pid, worker, service, guard):
            if pid is not None:
                try:
                    psutil.Process(pid).kill()
                except psutil.NoSuchProcess:
                    pass
        root.wait(timeout=5)


def test_normal_close_does_not_signal_parent():
    from villa_gpu_manager.stop_guard import StopGuard
    guard = StopGuard()
    guard.register([])
    guard.close()
    assert guard.process.returncode == 0


def test_guard_catches_orphan_created_between_scans():
    code = r'''
import json, subprocess, sys, time
from villa_gpu_manager.stop_guard import StopGuard
guard = StopGuard(grace_seconds=.1)
worker = subprocess.Popen([sys.executable, '-c',
    'import subprocess,sys; sys.stdin.readline(); p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); print(p.pid,flush=True)'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True)
guard.register([worker.pid])
worker.stdin.write('go\n'); worker.stdin.flush()
orphan = int(worker.stdout.readline())
worker.wait()
print(json.dumps([orphan, guard.process.pid]), flush=True)
guard.stop()
time.sleep(60)
'''
    root = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    orphan = guard = None
    try:
        orphan, guard = json.loads(root.stdout.readline())
        root.wait(timeout=8)
        assert root.returncode < 0
        assert not psutil.pid_exists(orphan) or psutil.Process(orphan).status() == psutil.STATUS_ZOMBIE
    finally:
        for pid in (root.pid, orphan, guard):
            if pid is not None:
                try:
                    psutil.Process(pid).kill()
                except psutil.NoSuchProcess:
                    pass
        root.wait(timeout=5)
