"""One process-table collection, shared by all ownership-tree lookups."""

from collections import defaultdict
import os
import sys
import time

import psutil

from .processes import process_state


class ProcessSnapshot:
    def __init__(self, rows, started, finished):
        self.rows = {row[0]: row for row in rows}
        self.started, self.finished = started, finished
        self.children = defaultdict(list)
        for row in self.rows.values():
            if row[1] is not None:
                self.children[row[1]].append(row[0])

    def inspect(self, identity, uid):
        row = self.rows.get(identity["pid"])
        # The snapshot is non-atomic. Fresh reports and missing/inaccessible
        # rows need direct identity verification, never guessed death.
        if row is None or row[2] is None or identity["start_us"] > self.started * 1e6:
            return process_state(identity, uid)
        if row[2:4] != [identity["start_us"], identity["boot_us"]]:
            return process_state(identity, uid)
        if row[4] != uid:
            return "unknown"
        if row[5] == "zombie":
            return process_state(identity, uid)
        return "alive"

    def descendants(self, identity, uid):
        row = self.rows.get(identity["pid"])
        if row is None or row[2:4] != [identity["start_us"], identity["boot_us"]]:
            return []  # Never attach an old PID incarnation's children.
        if self.inspect(identity, uid) != "alive":
            return []
        found, seen = [], {identity["pid"]}
        todo = list(self.children.get(identity["pid"], ()))
        while todo:
            pid = todo.pop()
            if pid in seen:
                continue
            seen.add(pid)
            row = self.rows[pid]
            if row[2] is not None:
                found.append(dict(pid=pid, start_us=row[2], boot_us=row[3]))
            todo.extend(self.children.get(pid, ()))
        return found


def collect_process_rows():
    """Read each PID once. Unknown rows remain present, not silently absent."""
    started = time.time()
    boot = psutil.boot_time()
    boot_us = round(boot * 1e6)
    rows = []
    if sys.platform.startswith("linux"):
        ticks = os.sysconf("SC_CLK_TCK")
        for pid in psutil.pids():
            try:
                with open(f"/proc/{pid}/stat") as stream:
                    stat = stream.read().rsplit(")", 1)[1].split()
                with open(f"/proc/{pid}/status") as stream:
                    uid = next(int(line.split()[1]) for line in stream if line.startswith("Uid:"))
                rows.append([pid, int(stat[1]), round((int(stat[19]) / ticks + boot) * 1e6),
                             boot_us, uid, "zombie" if stat[0] == "Z" else "alive"])
            except (OSError, ValueError, IndexError, StopIteration):
                rows.append([pid, None, None, boot_us, None, "unknown"])
    else:
        for pid in psutil.pids():
            try:
                p = psutil.Process(pid)
                with p.oneshot():
                    rows.append([pid, p.ppid(), round(p.create_time() * 1e6), boot_us,
                                 p.uids().real, p.status()])
            except (psutil.Error, OSError):
                rows.append([pid, None, None, boot_us, None, "unknown"])
    return rows, started, time.time()


def collector_main(channel_socket):
    from .cohort import _Channel
    channel = _Channel(channel_socket)
    try:
        while True:
            kind, _ = channel.receive()
            if kind == "stop":
                return
            if kind != "collect":
                raise RuntimeError("invalid process collector command")
            rows, started, finished = collect_process_rows()
            for offset in range(0, len(rows), 512):
                channel.send("rows", rows[offset:offset + 512])
                channel.flush()
            channel.send("complete", [started, finished])
            channel.flush()
    except (EOFError, BrokenPipeError, ConnectionResetError):
        return
    finally:
        channel.close()


class ProcessCollector:
    """Persistent OS process; never collect or wait for a full frame in poll."""
    def __init__(self, context, *, target=collector_main):
        import socket
        from .cohort import _Channel
        parent, child = socket.socketpair()
        self.channel = _Channel(parent)
        self.process = context.Process(target=target, args=(child,), daemon=True)
        self.process.start()
        child.close()
        self.pending = False
        self.rows = []

    def request(self):
        if not self.pending:
            self.channel.send("collect")
            self.pending = True

    def poll(self):
        if not self.process.is_alive():
            raise RuntimeError("process collector exited; ownership evidence unavailable")
        for _ in range(8):
            value = self.channel.poll()
            if value is None:
                return None
            kind, payload = value
            if kind == "rows":
                self.rows.extend(payload)
                if len(self.rows) > 262144:
                    raise RuntimeError("process snapshot exceeds safety bound")
            elif kind == "complete":
                snapshot = ProcessSnapshot(self.rows, *payload)
                self.rows = []
                self.pending = False
                return snapshot
            else:
                raise RuntimeError("invalid process collector response")
        return None

    def close(self):
        self.channel.close()
        self.process.join(timeout=1)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=2)
