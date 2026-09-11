"""Fixed, non-pausable exclusive commands using the locked public protocol.

For controlled commands whose workers remain in the owned process group.
Commands must not daemonize/detach children. No Torch imports or GPU probing.
"""
from copy import deepcopy
from functools import wraps
import json
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
    def __init__(self, command, environment):
        read_fd, self.gate = os.pipe()
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-c", _GATE, str(read_fd), *command],
                pass_fds=(read_fd,), start_new_session=True, env=environment)
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


@_termination_signals
def run_exclusive(command, *, socket, gpu_uuids, priority=100, label="fiber-evaluation",
                  environment=None, poll_interval=1.0, stop_event=None,
                  client_factory=Client, cleanup_timeout=30.0):
    """Run once under an exact full-VRAM grant for the selected managed GPUs.

    No service startup or unmanaged fallback. Manager loss during a live rigid
    grant permits continued work/reconnect, as specified by protocol v1. Cleanup
    failure NEVER produces a zero-resource report. Returns the command exit code.
    """
    if not command or not gpu_uuids or len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError("command and distinct GPU UUIDs required")
    if type(priority) is not int or poll_interval <= 0 or cleanup_timeout <= 0:
        raise ValueError("integer priority and positive intervals required")
    client = client_factory(socket)
    child, request, grant = None, None, None
    outcome = "failed"

    def reconnect():
        client.connect()
        return client.register(label, request, pausable=False, adaptable=False)

    def stopped():
        return stop_event is not None and stop_event.is_set()

    try:
        inventory = client.connect()["inventory"]
        if not set(gpu_uuids) <= inventory["gpus"].keys():
            raise ValueError("requested GPUs are not in the manager inventory")
        expected = {g: inventory["gpus"][g] for g in gpu_uuids}
        request = dict(max_cpu_cores=inventory["cpu_cores"], max_gpus=len(expected),
                       min_vram_bytes_per_gpu=min(expected.values()),
                       max_vram_bytes_per_gpu=max(expected.values()),
                       allowed_gpu_uuids=list(gpu_uuids), priority=priority, exclusive=True)
        task = client.register(label, request, pausable=False, adaptable=False)
        while True:
            if stopped():
                outcome = "cancelled"
                return 130
            if task["phase"] == "initialize":
                grant = deepcopy(task["granted"])
                if grant["gpus"] != expected:
                    raise RuntimeError("manager granted only part of the requested devices/VRAM; not launching")
                env = dict(os.environ if environment is None else environment)
                env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_uuids)
                child = GatedCommand(command, env)
                task = client.report("applied", grant, processes=child.identities)
                if task["phase"] != "applied" or task["granted"] != grant:
                    raise RuntimeError("exclusive grant changed before command authorization")
                child.authorize()
                print(f"gpu-manager running task={client.task_id} priority={priority} GPUs={len(expected)} exclusive", flush=True)
                break
            if task["phase"] != "waiting":
                raise RuntimeError(f"unexpected admission phase: {task['phase']}")
            print(f"gpu-manager waiting: {task['reason']}", flush=True)
            time.sleep(poll_interval)
            task = client.poll()
        while child.process.poll() is None:
            child.members()
            if stopped():
                outcome = "cancelled"
                return 130
            time.sleep(poll_interval)
            try:
                task = client.poll() if client.socket is not None else reconnect()
                if task["phase"] != "applied" or task["granted"] != grant:
                    raise RuntimeError("manager changed the fixed exclusive grant")
            except OSError:
                client.close()
                print("gpu-manager disconnected; retaining fixed grant while reconnecting", flush=True)
            except ProtocolError as error:
                if error.code not in ("stale_epoch", "unavailable"):
                    raise
                client.close()
        code = child.process.returncode
        outcome = "completed" if code == 0 else f"command_exit:{code}"
        return code
    finally:
        try:
            # No release until all owned CPU/GPU processes are verifiably gone.
            # Closing the child gate also handles an uncertain applied response.
            if child is not None:
                child.close()
            if client.task_id is not None and request is not None:
                deadline = time.monotonic() + cleanup_timeout
                while True:
                    try:
                        task = client.poll() if client.socket is not None else reconnect()
                        if task["terminal"]:
                            break
                        task = client.pause(True)
                        if task["granted"] != empty():
                            client.report("released", empty(), detail="command group stopped and joined")
                        client.finish(outcome)
                        break
                    except (OSError, ProtocolError) as error:
                        client.close()
                        if isinstance(error, ProtocolError) and error.code not in ("stale_epoch", "unavailable", "terminal"):
                            raise
                        if time.monotonic() >= deadline:
                            raise RuntimeError("command stopped but manager release unconfirmed; reservation retained") from error
                        time.sleep(min(poll_interval, .2))
        finally:
            client.close()
