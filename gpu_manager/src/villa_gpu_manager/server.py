"""Bounded nonblocking Unix socket service with durable ownership journal."""

import fcntl
import json
import os
from pathlib import Path
import selectors
import socket
import stat
import tempfile
import time

from .engine import Engine
from .processes import peer_uid
from .protocol import MAX_FRAME, VERSION, ProtocolError, decode, encode, require


def save_journal(path, state):
    fd, tmp = tempfile.mkstemp(prefix=".journal-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, allow_nan=False, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Server:
    def __init__(self, path, inventory, *, allowed_uids=None, controller_uids=None,
                 shared_group=False, scan_seconds=60, max_connections=128, engine_options=None,
                 monitor=None):
        self.path = Path(path)
        self.allowed = set(allowed_uids if allowed_uids is not None else [os.getuid()])
        self.controllers = set(controller_uids if controller_uids is not None else [os.getuid()])
        require(self.controllers <= self.allowed, "controllers must be allowed peers")
        require(scan_seconds > 0 and max_connections > 0, "invalid service limits")
        self.scan_seconds, self.max_connections = scan_seconds, max_connections
        self.selector = selectors.DefaultSelector()
        self.connections = {}
        self.listener = None
        self.lock = None
        self.bound = False
        self.shared_group = shared_group
        self.inventory = inventory
        self.engine_options = engine_options or {}
        self.monitor = monitor

    def __enter__(self):
        try:
            self._start()
            return self
        except BaseException:
            self.close(force=True)
            raise

    def _start(self):
        self.path.parent.mkdir(mode=0o1770 if self.shared_group else 0o700, parents=True, exist_ok=True)
        info = self.path.parent.lstat()
        require(stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid(), "socket directory must be owned by server user")
        forbidden = 0o007 if self.shared_group else 0o077
        require(not (stat.S_IMODE(info.st_mode) & forbidden), "socket directory permissions are too broad")
        require(not self.shared_group or bool(info.st_mode & stat.S_ISVTX), "shared socket directory requires sticky bit (1770)")
        lock_path = str(self.path) + ".lock"
        self.lock = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        lock_info = os.fstat(self.lock)
        require(stat.S_ISREG(lock_info.st_mode) and lock_info.st_uid == os.getuid()
                and not stat.S_IMODE(lock_info.st_mode) & 0o077, "lock ownership/permissions are unsafe")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.journal = Path(str(self.path) + ".json")
        state = None
        if self.journal.exists():
            fd = os.open(self.journal, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd) as f:
                info = os.fstat(f.fileno())
                require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid()
                        and not stat.S_IMODE(info.st_mode) & 0o077, "journal ownership/permissions are unsafe")
                state = json.load(f)
        elif self.path.exists() or os.fstat(self.lock).st_size:
            raise RuntimeError("missing recovery journal; refusing empty-ledger takeover")
        self.engine = Engine(self.inventory, controller_uids=self.controllers, state=state, **self.engine_options)
        if self.monitor is not None:
            self.engine.update_blocked_gpus(self.inventory["gpus"])
        save_journal(self.journal, self.engine.state)
        os.pwrite(self.lock, b"initialized\n", 0)
        os.fsync(self.lock)
        if self.path.exists():
            info = self.path.lstat()
            require(stat.S_ISSOCK(info.st_mode) and info.st_uid == os.getuid(), "refusing to remove unowned/non-socket entry")
            self.path.unlink()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.path))
        self.bound = True
        self.path.chmod(0o660 if self.shared_group else 0o600)
        self.listener.listen()
        self.listener.setblocking(False)
        self.selector.register(self.listener, selectors.EVENT_READ, None)
        self.last_scan = time.monotonic()

    def _drop(self, conn):
        self.selector.unregister(conn)
        self.connections.pop(conn, None)
        conn.close()

    def step(self, timeout=0.25):
        if self.monitor is not None:
            owned = {p["pid"] for t in self.engine.tasks.values() if t["granted"]["gpus"]
                     for p in t["processes"] if self.engine.inspect(p, t["uid"]) == "alive"}
            blocked = self.monitor.poll(owned)
            if blocked != self.engine.blocked_gpus:
                self.engine.update_blocked_gpus(blocked)
                save_journal(self.journal, self.engine.state)
        for key, events in self.selector.select(timeout):
            if key.data is None:
                conn, _ = self.listener.accept()
                try:
                    uid = peer_uid(conn)
                    if uid not in self.allowed or len(self.connections) >= self.max_connections:
                        conn.close()
                        continue
                    conn.setblocking(False)
                    state = {"uid": uid, "in": bytearray(), "out": bytearray(), "hello": False, "contact": time.monotonic()}
                    self.connections[conn] = state
                    self.selector.register(conn, selectors.EVENT_READ, state)
                except BaseException:
                    conn.close()
                    raise
                continue
            conn, state = key.fileobj, key.data
            try:
                if events & selectors.EVENT_READ:
                    data = conn.recv(65536)
                    if not data:
                        self._drop(conn)
                        continue
                    state["in"].extend(data)
                    state["contact"] = time.monotonic()
                    # Limit per-peer event work and queued output; pipelining is bounded.
                    processed = 0
                    while b"\n" in state["in"] and processed < 16:
                        processed += 1
                        frame, _, tail = state["in"].partition(b"\n")
                        state["in"] = bytearray(tail)
                        request = {}
                        response = {"version": VERSION, "id": None, "server_epoch": self.engine.epoch}
                        try:
                            request = decode(frame)
                            response["id"] = request.get("id")
                            require(state["hello"] or request.get("op") == "hello", "hello required", "handshake_required")
                            body = self.engine.dispatch(request, state["uid"])
                            # Persistence errors are fatal: do not run with an unjournaled ledger.
                            if request.get("op") not in ("hello", "status", "poll"):
                                save_journal(self.journal, self.engine.state)
                            state["hello"] = True
                            response.update(ok=True, body=body)
                        except ProtocolError as exc:
                            response.update(ok=False, error=exc.as_dict())
                        state["out"].extend(encode(response))
                        require(len(state["out"]) <= MAX_FRAME * 2, "slow client output overflow")
                    require(len(state["in"]) <= MAX_FRAME, "input overflow")
                    # The synchronous reference client never pipelines. Reject unconsumed
                    # complete-frame floods rather than strand them without a read event.
                    require(b"\n" not in state["in"], "too many pipelined requests")
                if events & selectors.EVENT_WRITE and state["out"]:
                    written = conn.send(state["out"])
                    del state["out"][:written]
                self.selector.modify(conn, selectors.EVENT_READ | (selectors.EVENT_WRITE if state["out"] else 0), state)
            except BlockingIOError:
                continue
            except (ConnectionError, ProtocolError):
                self._drop(conn)
        now = time.monotonic()
        for conn, state in list(self.connections.items()):
            if now - state["contact"] > 120:
                self._drop(conn)  # EOF never releases reservations.
        if now - self.last_scan >= self.scan_seconds:
            self.engine.scan_processes()
            save_journal(self.journal, self.engine.state)
            self.last_scan = now

    def drain(self):
        self.engine.drain()
        save_journal(self.journal, self.engine.state)

    def close(self, *, force=False):
        if self.bound and not force and any(t["granted"]["gpus"] or t["granted"]["cpu_cores"]
                                           for t in self.engine.tasks.values()):
            raise RuntimeError("active reservations: drain clients before shutdown, or explicitly force")
        if self.monitor is not None:
            self.monitor.close()
        for conn in list(self.connections):
            self._drop(conn)
        if self.listener is not None:
            self.listener.close()
        if self.bound:
            self.path.unlink(missing_ok=True)
            self.bound = False
        self.selector.close()
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None

    def __exit__(self, *exc):
        self.close(force=exc[0] is not None)
