"""Fixed, non-pausable exclusive commands using the locked public protocol.

For controlled commands whose workers remain in the owned process group.
Commands must not daemonize/detach children. No Torch imports or GPU probing.
"""
from copy import deepcopy
from functools import wraps
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import psutil

from .client import Client
from .policy import empty
from .processes import process_identity, process_state
from .protocol import ProtocolError


_GATE = """import os, sys
fd = int(sys.argv[1])
permission = os.read(fd, 1)
os.close(fd)
if permission != b'1':
    raise SystemExit(125)
os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
"""


class OwnedProcessGroup:
    """Identity-checked cleanup shared by the launcher and its watchdog."""
    def __init__(self, identity):
        self.identity = identity
        self.known = {identity["pid"]: identity}

    def members(self):
        try:
            if process_identity(self.identity["pid"]) != self.identity:
                raise RuntimeError("owned command PID reused; refusing group cleanup")
        except psutil.NoSuchProcess:
            pass
        for p in psutil.process_iter():
            try:
                if os.getpgid(p.pid) == self.identity["pid"]:
                    if p.uids().real != os.getuid():
                        raise RuntimeError("unverifiable member in owned process group")
                    self.known[p.pid] = process_identity(p.pid)
            except (ProcessLookupError, psutil.NoSuchProcess):
                continue
        live = []
        for identity in self.known.values():
            state = process_state(identity, os.getuid())
            if state == "unknown":
                raise RuntimeError("cannot verify command cleanup; reservation retained")
            if state == "alive":
                live.append(identity)
        return live

    def close(self, reap=lambda: None):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if not self.members():
                break
            try:
                os.killpg(self.identity["pid"], sig)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                reap()
                if not self.members():
                    break
                time.sleep(.02)
        if self.members():
            raise RuntimeError("owned workers survived cleanup; reservation retained")


def _watch_group(fd, identity):
    """Reported independently; survive launcher/controller loss until cleanup."""
    os.read(fd, 1)  # EOF when the launcher exits or finishes group cleanup.
    os.close(fd)
    group = OwnedProcessGroup(identity)
    while True:
        try:
            group.close()
            return
        except Exception as error:
            # Remaining alive retains the manager reservation on uncertainty.
            print(f"exclusive watchdog retaining reservation: {error}", file=sys.stderr, flush=True)
            time.sleep(1)


class GatedCommand:
    """Child PID is verifiable before exec; group cleanup precedes release."""
    def __init__(self, command, environment, *, stdout=None, stderr=None):
        self.closed = False
        read_fd, self.gate = os.pipe()
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-c", _GATE, str(read_fd), *command],
                pass_fds=(read_fd,), start_new_session=True, env=environment,
                stdout=stdout, stderr=stderr)
            self.identity = process_identity(self.process.pid)
            self.group = OwnedProcessGroup(self.identity)
            watch_read, self.heartbeat = os.pipe()
            try:
                watch_code = ("import sys,json; sys.path.insert(0,sys.argv[1]); "
                              "from villa_gpu_manager.exclusive import _watch_group; "
                              "_watch_group(int(sys.argv[2]),json.loads(sys.argv[3]))")
                self.watchdog = subprocess.Popen([sys.executable, "-c", watch_code,
                    str(Path(__file__).resolve().parents[1]), str(watch_read), json.dumps(self.identity)],
                    pass_fds=(watch_read,), start_new_session=True, env=environment)
                self.watchdog_identity = process_identity(self.watchdog.pid)
            finally:
                os.close(watch_read)
        except BaseException:
            os.close(self.gate)
            self.gate = None
            if hasattr(self, "process"):
                self.process.terminate()
                self.process.wait()
            if hasattr(self, "heartbeat"):
                os.close(self.heartbeat)
            if hasattr(self, "watchdog"):
                self.watchdog.wait(timeout=10)
            raise
        finally:
            os.close(read_fd)

    @property
    def processes(self):
        return self.identities

    def abort(self):
        self.close()

    @property
    def identities(self):
        return [self.identity, self.watchdog_identity]

    def authorize(self):
        if self.gate is None:
            raise RuntimeError("command already authorized/closed")
        os.write(self.gate, b"1")
        os.close(self.gate)
        self.gate = None

    def members(self):
        if self.watchdog.poll() is not None:
            raise RuntimeError("exclusive watchdog exited unexpectedly")
        return self.group.members()

    def close(self):
        if self.gate is not None:
            os.close(self.gate)
            self.gate = None
        self.group.close(reap=self.process.poll)
        self.process.wait(timeout=3)
        if self.heartbeat is not None:
            os.close(self.heartbeat)
            self.heartbeat = None
        if self.watchdog.wait(timeout=10) != 0:
            raise RuntimeError("exclusive watchdog failed; reservation retained")
        self.closed = True


def _termination_signals(function):
    @wraps(function)
    def call(*args, **kwargs):
        stop = kwargs.get("stop_event")
        if stop is None:
            stop = threading.Event()
            kwargs["stop_event"] = stop
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGHUP):
                handlers[sig] = signal.signal(sig, lambda *_: stop.set())
        try:
            return function(*args, **kwargs)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    return call



class AdmissionCancelled(Exception):
    """Stopped before workers were authorized."""


class ExclusiveTask:
    """Shared exact-grant lease for gated calibration cohorts and commands.

    The cohort adapter supplies processes, authorize(), abort(), and closed.
    Cleanup must verifiably join workers before the lease releases ownership.
    """
    def __init__(self, path, grant, *, label, priority=0, poll_seconds=.1,
                 client_factory=Client, stop_event=None, cleanup_timeout=30.0, exclusive=True,
                 wait_for_exact=False, exact_retry_seconds=5.0):
        if type(exclusive) is not bool:
            raise ValueError("exclusive must be boolean")
        if type(priority) is not int or poll_seconds <= 0 or cleanup_timeout <= 0:
            raise ValueError("integer priority and positive intervals required")
        if (type(grant["cpu_cores"]) is not int or grant["cpu_cores"] < 1 or
            not grant["gpus"] or any(type(v) is not int or v < 1 for v in grant["gpus"].values())):
            raise ValueError("exclusive task requires positive GPU budgets and CPUs")
        sizes = list(grant["gpus"].values())
        self.grant = deepcopy(grant)
        # The locked scheduler scales CPU promises with the sum of GPU bytes.
        # Compensate for heterogeneous full-device capacities so the resulting
        # CPU grant is exact too (ceil in cores_for cancels this floor).
        max_cpu = grant["cpu_cores"] * len(sizes) * max(sizes) // sum(sizes)
        self.request = dict(max_cpu_cores=max_cpu, max_gpus=len(sizes),
            min_vram_bytes_per_gpu=min(sizes), max_vram_bytes_per_gpu=max(sizes),
            allowed_gpu_uuids=sorted(grant["gpus"]), priority=priority, exclusive=exclusive)
        self.client = client_factory(path)
        self.client_factory, self.path = client_factory, path
        self.label, self.poll_seconds = label, poll_seconds
        self.stop_event, self.cleanup_timeout = stop_event, cleanup_timeout
        if type(wait_for_exact) is not bool or not math.isfinite(exact_retry_seconds) or exact_retry_seconds <= 0:
            raise ValueError("boolean exact waiting and positive retry interval required")
        self.wait_for_exact, self.exact_retry_seconds = wait_for_exact, exact_retry_seconds
        self.cohort = None
        self.epoch = self.generation = None
        self.closed = False

    def _reconnect(self):
        self.client.connect()
        return self.client.register(self.label, self.request, pausable=False, adaptable=False)

    def __enter__(self):
        try:
            inventory = self.client.connect()["inventory"]
            if (self.grant["cpu_cores"] > inventory["cpu_cores"] or
                any(size > inventory["gpus"].get(gpu, 0) for gpu, size in self.grant["gpus"].items())):
                raise ValueError("exclusive budget exceeds managed inventory")
            task = self.client.register(self.label, self.request, pausable=False, adaptable=False)
            last = None
            while True:
                if self.stop_event is not None and self.stop_event.is_set():
                    raise AdmissionCancelled()
                if task["phase"] == "initialize":
                    if task["granted"] == self.grant or not self.wait_for_exact:
                        break
                    # No cohort exists yet. Return this unusable partial promise;
                    # a non-pausable initialized grant otherwise cannot grow.
                    # Non-adaptable tasks latch their first grant permanently;
                    # finish the unused registration before retrying admission.
                    print(f"gpu-manager task={self.client.task_id} waiting for exact grant; returning partial {task['granted']}", flush=True)
                    task = self.client.pause(True)
                    if task["granted"] != empty():
                        self.client.report("released", empty(), detail="partial admission; no workers started")
                    deadline = time.monotonic() + self.exact_retry_seconds
                    while time.monotonic() < deadline:
                        if self.stop_event is not None and self.stop_event.is_set():
                            raise AdmissionCancelled()
                        time.sleep(max(0., min(self.poll_seconds, deadline - time.monotonic())))
                    self.client.finish("partial admission; no workers started")
                    self.client.close()
                    self.client = self.client_factory(self.path)
                    task = self._reconnect()
                    continue
                if task["phase"] != "waiting":
                    raise RuntimeError(f"unexpected admission phase: {task['phase']}")
                status = (task["phase"], task.get("reason"))
                if status != last:
                    print(f"gpu-manager task={self.client.task_id} waiting: {status[1]}", flush=True)
                    last = status
                time.sleep(self.poll_seconds)
                task = self.client.poll()
            if task["granted"] != self.grant:
                raise RuntimeError("manager granted only part of the requested CPU/devices/VRAM; not launching")
            self.epoch, self.generation = self.client.epoch, task["generation"]
            print(f"gpu-manager task={self.client.task_id} exclusive={self.request['exclusive']} grant={self.grant}", flush=True)
            return self
        except BaseException:
            self.close("failed before worker startup")
            raise

    def _check(self, task, phase):
        if (self.client.epoch != self.epoch or task["generation"] != self.generation or
            task["phase"] != phase or task["granted"] != self.grant):
            raise RuntimeError("manager allocation changed or was lost")

    def authorize(self):
        if self.cohort is None:
            raise RuntimeError("no gated cohort")
        self._check(self.client.poll(), "initialize")
        task = self.client.report("applied", self.grant, processes=self.cohort.processes)
        self._check(task, "applied")
        self.cohort.authorize()

    def check(self):
        try:
            if self.client.socket is None:
                task = self._reconnect()
                # Retained-token reconnect may observe a new server epoch;
                # the rigid applied allocation must still be exactly ours.
                if task["phase"] != "applied" or task["granted"] != self.grant:
                    raise RuntimeError("manager allocation changed during reconnect")
                self.epoch, self.generation = self.client.epoch, task["generation"]
            else:
                task = self.client.poll()
            self._check(task, "applied")
        except (OSError, ProtocolError) as error:
            if isinstance(error, ProtocolError) and error.code not in ("stale_epoch", "unavailable"):
                raise
            self.client.close()
            print("gpu-manager disconnected; retaining fixed grant while reconnecting", flush=True)

    def close(self, outcome="completed"):
        if self.closed:
            return
        try:
            if self.cohort is not None and not self.cohort.closed:
                self.cohort.abort()
            if self.cohort is not None and not self.cohort.closed:
                raise RuntimeError("refusing release before workers join")
            if self.client.task_id is not None:
                deadline = time.monotonic() + self.cleanup_timeout
                while True:
                    try:
                        task = self.client.poll() if self.client.socket is not None else self._reconnect()
                        if task["terminal"]:
                            break
                        task = self.client.pause(True)
                        if task["granted"] != empty():
                            self.client.report("released", empty(), detail="owned workers stopped and joined")
                        self.client.finish(outcome)
                        break
                    except (OSError, ProtocolError) as error:
                        self.client.close()
                        if isinstance(error, ProtocolError) and error.code == "terminal":
                            # Authenticated re-registration after a lost finish
                            # reply confirms the journal already retired us.
                            # Workers have been joined before entering this loop.
                            break
                        if isinstance(error, ProtocolError) and error.code not in ("stale_epoch", "unavailable"):
                            raise
                        if time.monotonic() >= deadline:
                            raise RuntimeError("workers stopped but manager release unconfirmed; reservation retained") from error
                        time.sleep(min(self.poll_seconds, .2))
            self.closed = True
        finally:
            self.client.close()

    def __exit__(self, exc_type, exc, tb):
        self.close("completed" if exc is None else f"failed: {type(exc).__name__}")


@_termination_signals
def run_exclusive(command, *, socket, gpu_uuids, priority=100, label="fiber-evaluation",
                  environment=None, poll_interval=1.0, stop_event=None,
                  client_factory=Client, cleanup_timeout=30.0, before_authorize=None,
                  exclusive=True, cpu_cores=None, vram_bytes_per_gpu=None,
                  wait_for_exact=False, exact_retry_seconds=5.0):
    """Gated command adapter around ExclusiveTask; no unmanaged fallback.

    The caller may bootstrap the shared manager before calling. An independent
    watchdog retains ownership and cleans up if this launcher is killed.
    """
    if not command or not gpu_uuids or len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError("command and distinct GPU UUIDs required")
    client = client_factory(socket)
    try:
        inventory = client.connect()["inventory"]
    finally:
        client.close()
    if not set(gpu_uuids) <= inventory["gpus"].keys():
        raise ValueError("requested GPUs are not in the manager inventory")
    if not exclusive and (cpu_cores is None or vram_bytes_per_gpu is None):
        raise ValueError("non-exclusive command requires explicit CPU and VRAM budgets")
    grant = dict(cpu_cores=inventory["cpu_cores"] if cpu_cores is None else cpu_cores,
                 gpus={gpu: inventory["gpus"][gpu] if vram_bytes_per_gpu is None else vram_bytes_per_gpu for gpu in gpu_uuids})
    lease = ExclusiveTask(socket, grant, label=label, priority=priority,
        poll_seconds=poll_interval, stop_event=stop_event, client_factory=client_factory,
        cleanup_timeout=cleanup_timeout, exclusive=exclusive,
        wait_for_exact=wait_for_exact, exact_retry_seconds=exact_retry_seconds)
    outcome = "failed"
    try:
        lease.__enter__()
        if before_authorize is not None:
            before_authorize()
        env = dict(os.environ if environment is None else environment)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_uuids)
        lease.cohort = child = GatedCommand(command, env)
        lease.authorize()
        while child.process.poll() is None:
            child.members()
            if stop_event.is_set():
                outcome = "cancelled"
                return 130
            time.sleep(poll_interval)
            lease.check()
        code = child.process.returncode
        outcome = "completed" if code == 0 else f"command_exit:{code}"
        return code
    except AdmissionCancelled:
        outcome = "cancelled"
        return 130
    finally:
        lease.close(outcome)
