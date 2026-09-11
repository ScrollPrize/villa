"""Spawned cohorts: bounded control sockets, separate in-RAM state transport.

The manager wire protocol is not used for application state. Private child
socketpairs reuse its bounded JSON framing; model state lives in shared memory
and is decoded only in workers. The CPU supervisor never unpickles tensors.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
import pickle
import signal
import socket
import time
import traceback
import uuid

import psutil

from .processes import process_identity
from .protocol import MAX_FRAME, decode, encode


@contextmanager
def _spawn_environment(values):
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(frozen=True)
class SharedState:
    """Opaque CPU state; owner calls close after the last reader has finished."""

    name: str
    size: int

    def load(self):
        # Trusted private worker state, never an external/socket-supplied pickle.
        memory = SharedMemory(name=self.name)
        try:
            if not 0 < self.size <= len(memory.buf):
                raise ValueError("invalid shared handoff size")
            return pickle.loads(memory.buf[:self.size])
        finally:
            memory.close()

    def close(self):
        _unlink(self.name)


def _unlink(name):
    try:
        memory = SharedMemory(name=name)
    except FileNotFoundError:
        return
    try:
        memory.unlink()
    finally:
        memory.close()


class _Channel:
    """Nonblocking, bounded framing. poll/recv never waits for a full frame."""

    def __init__(self, sock):
        self.socket = sock
        sock.setblocking(False)
        self.input = bytearray()
        self.output = bytearray()

    def send(self, kind, payload=None):
        frame = encode(dict(kind=kind, payload=payload))
        if len(self.output) + len(frame) > MAX_FRAME:
            raise RuntimeError("cohort control output exceeds bound")
        self.output.extend(frame)

    def poll(self):
        if self.output:
            try:
                count = self.socket.send(self.output)
                del self.output[:count]
            except BlockingIOError:
                pass
        # Deliver frames already buffered before observing EOF on the socket.
        # booted+cancelled can arrive together followed by immediate child exit.
        if b"\n" not in self.input:
            try:
                data = self.socket.recv(65536)
                if not data:
                    raise EOFError("cohort control socket closed")
                self.input.extend(data)
            except BlockingIOError:
                pass
        if len(self.input) > MAX_FRAME:
            raise RuntimeError("cohort control input exceeds bound")
        if b"\n" not in self.input:
            return None
        frame, _, rest = self.input.partition(b"\n")
        self.input = bytearray(rest)
        value = decode(frame)
        if set(value) != {"kind", "payload"}:
            raise RuntimeError("invalid cohort control message")
        return value["kind"], value["payload"]

    def receive(self):
        # Only children use blocking receive. Parent continuously polls all
        # channels, manager directives and deadlines without a receive thread.
        while True:
            value = self.poll()
            if value is not None:
                return value
            time.sleep(0.001)

    def flush(self):
        while self.output:
            try:
                count = self.socket.send(self.output)
                del self.output[:count]
            except BlockingIOError:
                time.sleep(0.001)

    def close(self):
        self.socket.close()


def _worker(sock, factory, rank, grant, initial_state, state_name):
    # Loader processes inherit this session. Keep them separate from the CPU
    # supervisor so failure cleanup can stop orphaned descendants as well.
    os.setsid()
    channel = _Channel(sock)
    adapter = None
    try:
        channel.send("booted")
        command, _ = channel.receive()
        if command == "cancel":
            channel.send("cancelled")
            channel.flush()
            return
        if command != "initialize":
            raise RuntimeError("worker initialization was not authorized")
        state = initial_state.load() if isinstance(initial_state, SharedState) else initial_state
        adapter = factory(rank, grant, state)
        del state
        channel.send("ready")
        while True:
            command, sequence = channel.receive()
            if command == "unit":
                channel.send("unit", [sequence, adapter.run_unit()])
            elif command == "close":
                signature, state = adapter.capture()
                descriptor = None
                if rank == 0:
                    # Ordinary pickle embeds CPU tensor storage; it does not
                    # use multiprocessing's live-sender FD/resource_sharer.
                    data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
                    memory = SharedMemory(name=state_name, create=True, size=len(data))
                    try:
                        memory.buf[:] = data
                    finally:
                        memory.close()
                    descriptor = dict(name=state_name, size=len(data))
                adapter.close()
                adapter = None
                channel.send("closed", [signature, descriptor])
                acknowledgement, _ = channel.receive()
                if acknowledgement != "received":
                    raise RuntimeError("missing state ownership acknowledgement")
                return
            else:
                raise RuntimeError(f"unknown cohort command {command!r}")
    except BaseException:
        try:
            channel.send("error", traceback.format_exc()[-65536:])
            channel.flush()
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except BaseException:
                pass
        channel.close()


class ProcessCohort:
    """One spawned process per UUID. Parent owns no CUDA or tensor state.

    Report ``processes`` to the manager before ``authorize``: children initially
    wait behind an initialization gate and cannot yet create GPU contexts.
    Environment overrides apply before child module imports, then are restored.
    Construct on the sole process-launching supervisor thread.
    """

    def __init__(self, factory, grant, state=None, *, environment=None,
                 on_unit=None, startup_timeout=300.0, unit_timeout=900.0,
                 drain_timeout=300.0, clock=time.monotonic):
        if state is not None and not isinstance(state, SharedState):
            raise TypeError("cohort restore requires opaque SharedState, not a large spawn argument")
        self._children, self._channels = [], []
        self._identities = {}
        self._phases, self._replies = {}, {}
        self.ready = self.idle = self.closed = self.done = self.authorized = False
        self.state = state
        self._state_name = "vgm_" + uuid.uuid4().hex[:20]
        self._draining = self._closing = False
        self._sequence = 0
        self._on_unit = on_unit
        self._clock, self._started = clock, clock()
        self._startup_timeout = float(startup_timeout)
        self._unit_timeout = float(unit_timeout)
        self._drain_timeout = float(drain_timeout)
        if min(self._startup_timeout, self._unit_timeout, self._drain_timeout) <= 0:
            raise ValueError("cohort timeouts must be positive")
        devices = sorted(grant["gpus"])
        if not devices:
            raise ValueError("a cohort requires a nonempty GPU grant")
        context = mp.get_context("spawn")
        try:
            for rank, gpu in enumerate(devices):
                parent, child = socket.socketpair()
                self._channels.append(_Channel(parent))
                values = dict(environment or {}, CUDA_VISIBLE_DEVICES=gpu)
                process = context.Process(target=_worker, args=(
                    child, factory, rank, deepcopy(grant), state, self._state_name,
                ), name=f"cooperative-worker-{rank}")
                try:
                    with _spawn_environment(values):
                        process.start()
                    self._children.append(process)
                    self._identities[process.pid] = process_identity(process.pid)
                    self._phases[rank] = "starting"
                finally:
                    child.close()
        except BaseException:
            self.abort()
            raise

    @property
    def processes(self):
        return [process_identity(p.pid) for p in self._children if p.is_alive()]

    def authorize(self):
        if self._draining or self.authorized:
            raise RuntimeError("cannot authorize this cohort")
        self.authorized = True
        self._send("initialize")

    def _send(self, command, value=None):
        for channel in self._channels:
            channel.send(command, value)

    def start_unit(self):
        if not self.ready or not self.idle or self._draining or self.done:
            raise RuntimeError("cohort is not eligible for another work unit")
        self._sequence += 1
        self.idle = False
        self._started = self._clock()
        self._replies.clear()
        self._send("unit", self._sequence)

    def drain(self):
        self._draining = True
        if not self.authorized and not self._closing:
            self._closing = True
            self._started = self._clock()
            self._send("cancel")
        self._begin_close_if_idle()

    def _begin_close_if_idle(self):
        if self.ready and self.idle and self._draining and not self._closing:
            self._closing = True
            self._started = self._clock()
            self._replies.clear()
            self._send("close")

    def poll(self):
        if self.closed:
            return
        for rank, (process, channel) in enumerate(zip(self._children, self._channels, strict=True)):
            if self._phases[rank] in ("closed", "cancelled") and not channel.output:
                continue
            try:
                message = channel.poll()
                if message is not None:
                    kind, payload = message
                    if kind == "error":
                        raise RuntimeError(f"cohort worker {rank} failed:\n{payload}")
                    if kind == "booted" and self._phases[rank] == "starting":
                        self._phases[rank] = "booted"
                    elif kind == "ready" and self._phases[rank] == "booted":
                        self._phases[rank] = "ready"
                    elif kind == "unit" and not self.idle and not self._closing:
                        sequence, result = payload
                        if sequence != self._sequence or rank in self._replies:
                            raise RuntimeError("unexpected cohort completion sequence")
                        self._replies[rank] = result
                    elif kind == "closed" and self._closing and self.authorized:
                        self._phases[rank] = "closed"
                        self._replies[rank] = payload
                        if rank == 0:
                            descriptor = payload[1]
                            if descriptor["name"] != self._state_name or type(descriptor["size"]) is not int:
                                raise RuntimeError("invalid state ownership descriptor")
                        channel.send("received")
                    elif kind == "cancelled" and self._closing and not self.authorized:
                        self._phases[rank] = "cancelled"
                    else:
                        raise RuntimeError(f"unexpected worker {rank} response {kind}")
                elif process.exitcode is not None and self._phases[rank] not in ("closed", "cancelled"):
                    raise RuntimeError(f"cohort worker {rank} exited {process.exitcode} without completion")
            except EOFError as exc:
                if self._phases[rank] not in ("closed", "cancelled") or channel.output:
                    raise RuntimeError(f"cohort worker {rank} lost its control connection") from exc
            except OSError as exc:
                raise RuntimeError(f"cohort worker {rank} lost its control connection") from exc
        if not self.ready and all(v == "ready" for v in self._phases.values()):
            self.ready = self.idle = True
        if not self.idle and len(self._replies) == len(self._children) and not self._closing:
            results = [self._replies[i] for i in range(len(self._children))]
            self.idle = True
            if self._on_unit is not None:
                self.done = bool(self._on_unit(results))
        self._begin_close_if_idle()
        if self._closing and all(v in ("closed", "cancelled") for v in self._phases.values()):
            if self.authorized:
                signatures = [self._replies[i][0] for i in range(len(self._children))]
                if any(value != signatures[0] for value in signatures[1:]):
                    raise RuntimeError("cohort replicas disagree at handoff")
            for child in self._children:
                child.join(timeout=0)
            if all(child.exitcode is not None for child in self._children):
                if any(child.exitcode != 0 for child in self._children):
                    raise RuntimeError("cohort worker failed during shutdown")
                if self._live_worker_sessions():
                    raise RuntimeError("cohort loader descendants remain alive after worker shutdown")
                if self.authorized:
                    self.state = SharedState(**self._replies[0][1])
                self.closed = True
                for channel in self._channels:
                    channel.close()
                return
        timeout = (self._drain_timeout if self._closing else
                   self._startup_timeout if not self.ready else self._unit_timeout)
        if (self._closing or not self.ready or not self.idle) and self._clock() - self._started > timeout:
            raise TimeoutError(f"cohort timed out: phases={self._phases} unit={self._sequence} closing={self._closing}")

    def _live_worker_sessions(self):
        """Find owned sessions even after a rank exits and loaders are orphaned.

        Workers/loaders must not detach from their session. An inaccessible
        process inspection fails closed; only vanished processes and zombies
        are considered gone. Check leader identity against PID reuse.
        """
        sessions = set(self._identities)
        for pid in tuple(sessions):
            try:
                if process_identity(pid) != self._identities[pid]:
                    sessions.remove(pid)
            except psutil.NoSuchProcess:
                pass
        live = set()
        for process in psutil.process_iter():
            try:
                session = os.getsid(process.pid)
                if session in sessions and process.status() != psutil.STATUS_ZOMBIE:
                    # Descendants may create a process group, but not a session.
                    live.add((session, os.getpgid(process.pid)))
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
        return live

    def abort(self):
        """Stop owned worker sessions, including loaders, before claiming release."""
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for _session, group in self._live_worker_sessions():
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
            # Also handle a newly spawned child not yet inside _worker/setsid.
            for child in self._children:
                if child.is_alive():
                    (child.terminate if sig == signal.SIGTERM else child.kill)()
            deadline = time.monotonic() + 5
            while True:
                for child in self._children:
                    child.join(timeout=0)
                if all(child.exitcode is not None for child in self._children) and not self._live_worker_sessions():
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.02)
            if all(child.exitcode is not None for child in self._children) and not self._live_worker_sessions():
                break
        else:
            raise RuntimeError("cannot stop cohort worker/loader session; resources remain held")
        for channel in self._channels:
            channel.close()
        # A normal completed handoff is owned by the supervisor, not this cohort.
        if not self.closed:
            _unlink(self._state_name)
        self.closed = True
