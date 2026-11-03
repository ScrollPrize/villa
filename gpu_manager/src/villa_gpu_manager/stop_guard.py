"""Independent client-owned stop escalation; no manager protocol/enforcement.

Register compute roots which create their own session before spawning children.
Descendants must stay in that session, as required by ProcessCohort. Never
register the supervisor: its descendants include the detached shared manager.
"""
import json
import os
import select
import signal
import subprocess
import sys
import time

import psutil

from .processes import live_worker_sessions, process_identity, process_state


class StopGuard:
    def __init__(self, *, grace_seconds=15):
        read, self.write = os.pipe()
        ready, ready_child = os.pipe()
        self.process = subprocess.Popen([sys.executable, "-m", __name__, str(read),
            json.dumps(process_identity()), str(grace_seconds), str(ready_child)],
            pass_fds=(read, ready_child), start_new_session=True)
        os.close(read)
        os.close(ready_child)
        try:
            if not select.select([ready], [], [], 10)[0] or os.read(ready, 1) != b"1":
                self.process.kill()
                self.process.wait(timeout=5)
                os.close(self.write)
                raise RuntimeError("training stop guard did not initialize")
        finally:
            os.close(ready)
        self.stopping = False

    def check(self):
        if self.process.poll() is not None:
            raise RuntimeError("training stop guard exited unexpectedly")

    def register(self, pids):
        self.check()
        for pid in pids:
            try:
                self._send({"register": process_identity(pid)})
            except psutil.NoSuchProcess:
                pass
        if self.stopping:
            raise KeyboardInterrupt("training stopped during worker startup")

    def _send(self, message):
        data = (json.dumps(message) + "\n").encode()
        if len(data) > os.fpathconf(self.write, "PC_PIPE_BUF"):
            raise ValueError("stop guard control record exceeds atomic pipe limit")
        os.write(self.write, data)

    def stop(self):
        # Signal handlers never write into the registration pipe or wait on a
        # Python lock. The guard is ready before handlers are installed.
        try:
            os.kill(self.process.pid, signal.SIGUSR2 if self.stopping else signal.SIGUSR1)
        except ProcessLookupError:
            pass  # check() reports failure; do not prevent remaining cleanup.
        self.stopping = True

    def close(self):
        self._send({"done": True})
        os.close(self.write)
        self.process.wait(timeout=5)
        if self.process.returncode:
            raise RuntimeError("training stop guard failed")


def watch(fd, root, grace, ready):
    owned = {}
    uid = os.getuid()
    deadline = None
    stage = 0
    buffer = b""
    next_scan = 0

    def stop(signum=None, frame=None):
        nonlocal deadline
        if deadline is None:
            deadline = time.monotonic() + grace
        if signum == signal.SIGUSR2:
            deadline = time.monotonic()

    for sig in (signal.SIGUSR1, signal.SIGUSR2, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    os.write(ready, b"1")
    os.close(ready)
    while True:
        done = False
        readable, _, _ = select.select([fd] if fd is not None else [], [], [], .1)
        if readable:
            data = os.read(fd, 65536)
            if not data:
                os.close(fd)
                fd = None
                stop()
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                message = json.loads(line)
                if "register" in message:
                    identity = message["register"]
                    owned[identity["pid"]] = identity
                done |= bool(message.get("done"))
        if not done and time.monotonic() < next_scan and (deadline is None or time.monotonic() < deadline):
            continue
        # Shared inspection finds orphaned descendants even after leader exit.
        # Sample leader liveness BEFORE the process-table scan. If a leader
        # exits/spawns during that scan, retain it for the next pass rather than
        # pruning a just-orphaned session absent from an earlier table snapshot.
        states = {pid: process_state(identity, uid) for pid, identity in owned.items()}
        groups = live_worker_sessions(owned)
        sessions = {session for session, _ in groups}
        owned = {pid: identity for pid, identity in owned.items()
                 if pid in sessions or states[pid] != "dead"}
        alive = [identity for pid, identity in owned.items() if states[pid] != "dead"]
        next_scan = time.monotonic() + 5
        if done:
            if not groups and not alive:
                return
            print("training cleanup returned with live compute; escalating", file=sys.stderr, flush=True)
            stop()
        if deadline is None or time.monotonic() < deadline:
            continue
        if not groups and not alive:
            if process_state(root, uid) == "alive":
                print("training stop deadline exceeded: killing blocked supervisor after compute exited",
                      file=sys.stderr, flush=True)
                os.kill(root["pid"], signal.SIGKILL)
            return
        sig = signal.SIGTERM if stage == 0 else signal.SIGKILL
        print(f"training stop escalation: {sig.name} roots={[p['pid'] for p in alive]} groups={sorted(groups)}",
              file=sys.stderr, flush=True)
        for _, group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
        # Also catch a newly spawned worker not yet in setsid().
        for identity in alive:
            if process_state(identity, uid) == "alive":
                try:
                    os.kill(identity["pid"], sig)
                except ProcessLookupError:
                    pass
        if stage >= 2:
            print("training stop: survivors possibly in kernel I/O; ownership remains held", file=sys.stderr, flush=True)
        stage += 1
        deadline = time.monotonic() + 5


if __name__ == "__main__":
    watch(int(sys.argv[1]), json.loads(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]))
