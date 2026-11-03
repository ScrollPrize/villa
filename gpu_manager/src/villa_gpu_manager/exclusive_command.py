"""Deferred subprocess adapter for ExclusiveTask (no CUDA imports)."""

import subprocess
import time

import psutil

from .processes import process_identity, process_state


class ExclusiveCommand:
    def __init__(self, command, **popen_options):
        self.command, self.options = command, popen_options
        self.process = None
        self.identities = {}
        self.closed = False

    @property
    def processes(self):
        return list(self.identities.values())

    def authorize(self):
        # ExclusiveTask confirms applied ownership BEFORE invoking this method.
        self.process = subprocess.Popen(self.command, **self.options)
        self.observe()

    def observe(self):
        if self.process is None:
            return
        roots = [self.process.pid, *self.identities]
        for pid in set(roots):
            old = self.identities.get(pid)
            if old is not None and process_state(old, psutil.Process().uids().real) != "alive":
                continue
            try:
                members = [psutil.Process(pid), *psutil.Process(pid).children(recursive=True)]
                for member in members:
                    self.identities[member.pid] = process_identity(member.pid)
            except psutil.NoSuchProcess:
                continue

    def abort(self):
        # Tests may create new sessions, so cleanup follows identities, not only
        # the launcher's process group. Freeze the owned tree before killing it.
        uid = psutil.Process().uids().real
        stopped = set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            self.observe()
            fresh = set(self.identities) - stopped
            if not fresh:
                break
            for pid in fresh:
                if process_state(self.identities[pid], uid) == "alive":
                    try:
                        psutil.Process(pid).suspend()
                    except psutil.NoSuchProcess:
                        pass
                stopped.add(pid)
        else:
            raise RuntimeError("test tree did not quiesce; reservation retained")
        for identity in self.identities.values():
            if process_state(identity, uid) == "alive":
                try:
                    psutil.Process(identity["pid"]).kill()
                except psutil.NoSuchProcess:
                    pass
        if self.process is not None:
            self.process.wait(timeout=10)
        deadline = time.monotonic() + 10
        while any(process_state(row, uid) != "dead" for row in self.identities.values()):
            if time.monotonic() >= deadline:
                raise RuntimeError("test descendants remain; reservation retained")
            time.sleep(0.05)
        self.closed = True
