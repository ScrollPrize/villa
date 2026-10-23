"""Public socket multiplexing separated from metadata work by OS processes."""

from collections import deque
import multiprocessing as mp
import os
import signal
import socket
import time

from .cohort import _Channel
from .engine import Engine
from .processes import NvidiaMonitor, process_identity, process_state
from .process_snapshot import ProcessCollector, collector_main
from .protocol import MAX_FRAME, decode
from .server import Server, save_journal

IPC_LIMIT = MAX_FRAME * 4
MAX_PENDING = 64


def _receive_fd(descriptor):
    return descriptor.detach()


class _LockDescriptor:
    """Transfer the same flock open-file-description during spawn."""
    def __init__(self, fd):
        self.fd = fd

    def __reduce__(self):
        from multiprocessing.reduction import DupFd
        return _receive_fd, (DupFd(self.fd),)


def _state_owner(sock, inventory, state, epoch, parent_identity, journal,
                 monitor_config, scan_seconds, collector_target, singleton_fd):
    """Only this process mutates policy/history or writes the durable journal."""
    def terminate(_signal, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    channel = _Channel(sock, max_frame=IPC_LIMIT)
    collector = ProcessCollector(mp.get_context("spawn"), target=collector_target)
    monitor = NvidiaMonitor(inventory, **monitor_config) if monitor_config is not None else None
    own = process_identity()
    collector_identity = process_identity(collector.process.pid)
    snapshot = None
    def protected():
        members = [parent_identity, own, collector_identity]
        if snapshot is not None:
            members.extend(snapshot.descendants(parent_identity, os.getuid()))
        return members
    engine = Engine(inventory, controller_uids=state.pop("_controllers"), protected_processes=protected)
    # Server._start already reconciled recovery exactly once before handing over.
    engine.state, engine.epoch = state, epoch
    if monitor is not None:
        engine.update_blocked_gpus(inventory["gpus"])
    runtime = Server.__new__(Server)
    runtime.engine, runtime.journal = engine, journal
    clients = {}
    pending = deque()
    next_scan = 0
    def summary():
        return {"live": any(not t["terminal"] for t in engine.tasks.values()),
                "held": any(t["granted"]["gpus"] or t["granted"]["cpu_cores"] for t in engine.tasks.values()),
                "draining": engine.draining}
    def owned():
        return {p["pid"] for t in engine.tasks.values() if t["granted"]["gpus"]
                for p in t["processes"] if process_state(p, t["uid"]) == "alive"}
    try:
        channel.send("ready", summary())
        collector.request()
        while True:
            observed = collector.poll()
            if observed is not None:
                snapshot = observed
                engine.discover = snapshot.descendants
                # Direct identity checks remain cheap and verify live reports.
                # Descendant discovery uses the single collected parent map.
                engine.inspect = snapshot.inspect
                try:
                    engine.scan_processes()
                finally:
                    # Client release reports always require fresh verification.
                    engine.inspect = process_state
                save_journal(journal, engine.state)
                next_scan = time.monotonic() + scan_seconds
                channel.send("summary", summary())
            if time.monotonic() >= next_scan:
                collector.request()
            if monitor is not None:
                blocked = monitor.poll(owned)
                if blocked != engine.blocked_gpus:
                    engine.update_blocked_gpus(blocked)
                    save_journal(journal, engine.state)
                    channel.send("summary", summary())
            for _ in range(16):
                value = channel.poll()
                if value is None:
                    break
                kind, payload = value
                if kind == "request":
                    if len(pending) >= MAX_PENDING:
                        raise RuntimeError("metadata request backlog exceeded bound")
                    pending.append((payload, time.time()))
                    clients.setdefault(payload["connection"], {"uid": payload["uid"], "hello": False, "out": bytearray()})
                elif kind == "drop":
                    clients.pop(payload, None)
                elif kind == "drain":
                    engine.drain()
                    save_journal(journal, engine.state)
                    channel.send("summary", summary())
                elif kind == "stop":
                    return
                else:
                    raise RuntimeError("invalid private service message")
            waiting_connections = set()
            for _ in range(len(pending)):
                payload, arrived = pending.popleft()
                cid = payload["connection"]
                connection = clients.get(cid)
                if connection is None:
                    channel.send("cancelled", payload["serial"])
                    continue
                frame = bytes.fromhex(payload["frame"])
                try:
                    needs_tree = decode(frame).get("op") == "register"
                except Exception:
                    needs_tree = False  # Shared dispatcher returns framing error.
                if cid in waiting_connections or (needs_tree and (snapshot is None or snapshot.started < arrived)):
                    waiting_connections.add(cid)
                    pending.append((payload, arrived))
                    collector.request()
                    continue
                runtime._submit(None, connection, frame)
                response = decode(bytes(connection["out"]).rstrip(b"\n"))
                connection["out"].clear()
                channel.send("response", {"serial": payload["serial"], "response": response, "summary": summary()})
                # Bound output without a background feeder thread.
                if len(channel.output) > IPC_LIMIT // 2:
                    break
            time.sleep(0.001)
    except (EOFError, BrokenPipeError, ConnectionResetError):
        return
    finally:
        collector.close()
        if monitor is not None:
            monitor.close()
        channel.close()
        # Retain the singleton even if the public parent dies during a write.
        os.close(singleton_fd)


class ProcessServer(Server):
    """Production server; socket loop contains no Engine or history traversal.

    The base Server supplies exactly the same framing, peer authentication and
    filesystem singleton protections. Its direct Engine mode remains useful for
    deterministic unit tests; production CLI always selects this process backend.
    """
    def __init__(self, *args, collector_target=collector_main, **kwargs):
        super().__init__(*args, **kwargs)
        self.collector_target = collector_target
        self.backend = None
        self.channel = None
        self.pending = {}
        self.serial = 0
        self.connection_serial = 0
        self.dropped = deque()
        self.summary = {"live": True, "held": True, "draining": False}

    def _start(self):
        super()._start()
        parent, child = socket.socketpair()
        self.channel = _Channel(parent, max_frame=IPC_LIMIT)
        state = self.engine.state
        state["_controllers"] = list(self.controllers)
        monitor_config = None if self.monitor is None else {
            "reserve_bytes": self.monitor.reserve, "interval": self.monitor.interval}
        self.backend = mp.get_context("spawn").Process(target=_state_owner, args=(child,
            self.inventory, state, self.engine.epoch, process_identity(), self.journal,
            monitor_config, self.scan_seconds, self.collector_target, _LockDescriptor(self.lock)))
        self.backend.start()
        child.close()
        del state["_controllers"]
        # No state/history traversals occur in the socket process after startup.
        self.engine = None
        self.monitor = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not self.backend.is_alive():
                raise RuntimeError("metadata owner failed during startup; journal retained")
            value = self.channel.poll()
            if value is not None:
                if value[0] != "ready":
                    raise RuntimeError("invalid metadata startup reply")
                self.summary = value[1]
                return
            time.sleep(0.005)
        raise RuntimeError("metadata owner startup timed out; journal retained")

    def _submit(self, conn, state, frame):
        if len(self.pending) >= MAX_PENDING:
            raise ConnectionError("metadata queue full; retry without assuming release")
        if "connection" not in state:
            self.connection_serial += 1
            state["connection"] = self.connection_serial
        self.serial += 1
        try:
            self.channel.send("request", {"serial": self.serial, "connection": state["connection"],
                                          "uid": state["uid"], "frame": frame.hex()})
        except RuntimeError as exc:
            raise ConnectionError("metadata backpressure; retry without assuming release") from exc
        self.pending[self.serial] = (conn, state)

    def _before_io(self):
        if not self.backend.is_alive():
            raise RuntimeError("metadata owner exited; journal retained, clients must reconcile")
        for _ in range(32):
            value = self.channel.poll()
            if value is None:
                break
            kind, payload = value
            if kind == "summary":
                self.summary = payload
            elif kind == "cancelled":
                self.pending.pop(payload, None)
            elif kind == "response":
                self.summary = payload["summary"]
                target = self.pending.pop(payload["serial"], None)
                if target is not None:
                    conn, state = target
                    if self.connections.get(conn) is state:
                        try:
                            self._respond(conn, state, payload["response"])
                            self.selector.modify(conn, 3, state)
                        except (OSError, ValueError):
                            self._drop(conn)
            else:
                raise RuntimeError("invalid metadata response")
        while self.dropped and len(self.channel.output) < IPC_LIMIT // 2:
            self.channel.send("drop", self.dropped.popleft())

    def _after_io(self):
        pass

    def step(self, timeout=0.25):
        super().step(min(timeout, 0.01))

    def _drop(self, conn):
        state = self.connections.get(conn, {})
        if self.channel is not None and "connection" in state:
            self.dropped.append(state["connection"])
        super()._drop(conn)

    def idle_expired(self):
        if self.connections or self.pending or self.summary["live"]:
            self.idle_since = None
            return False
        if self.idle_since is None:
            self.idle_since = time.monotonic()
        return time.monotonic() - self.idle_since >= self.idle_seconds

    @property
    def drained(self):
        return self.summary["draining"] and not self.summary["held"] and not self.pending

    def drain(self):
        self.channel.send("drain")

    def close(self, *, force=False):
        if self.bound and not force and (self.pending or self.summary["held"]):
            raise RuntimeError("active reservations: drain clients before shutdown")
        if self.backend is not None:
            if self.channel is not None:
                self.channel.close()
                self.channel = None
            self.backend.join(timeout=4)
            if self.backend.is_alive():
                self.backend.terminate()
                self.backend.join(timeout=2)
            if self.backend.is_alive():
                self.backend.kill()
                self.backend.join(timeout=2)
        super().close(force=True)
