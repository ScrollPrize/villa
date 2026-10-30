"""One CPU-owned TensorBoard writer across replaceable worker cohorts."""

from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
import queue
import uuid


@dataclass(frozen=True)
class WriterFactory:
    commands: object
    replies: object
    owner_pid: int

    def __call__(self, path, *, enabled):
        if not enabled:
            return None
        writer = WriterProxy(self)
        writer._call("open", str(path), wait=True)
        return writer


class WriterProxy:
    def __init__(self, factory):
        self.factory = factory

    def _call(self, method, *args, wait=False, **kwargs):
        key = str(uuid.uuid4()) if wait else None
        while True:
            if os.getppid() != self.factory.owner_pid:
                raise RuntimeError("training supervisor exited during logging")
            try:
                self.factory.commands.put((key, method, args, kwargs), timeout=0.1)
                break
            except queue.Full:
                pass
        if wait:
            while True:
                if os.getppid() != self.factory.owner_pid:
                    raise RuntimeError("training supervisor exited before logging acknowledgement")
                try:
                    received, error = self.factory.replies.get(timeout=0.1)
                    break
                except queue.Empty:
                    pass
            if received != key:
                raise RuntimeError("TensorBoard acknowledgement belongs to another request")
            if error is not None:
                raise RuntimeError(f"supervisor TensorBoard error: {error}")

    def add_text(self, *args, **kwargs):
        self._call("add_text", *args, **kwargs)

    def add_scalar(self, *args, **kwargs):
        self._call("add_scalar", *args, **kwargs)

    def add_image(self, *args, **kwargs):
        self._call("add_image", *args, **kwargs)

    def flush(self):
        self._call("flush", wait=True)

    def close(self):
        # Only the CPU owner closes the real writer when the task finishes.
        self.flush()


class WriterHub:
    def __init__(self, writer_factory, *, queue_depth=256):
        context = mp.get_context("spawn")
        self.commands = context.Queue(maxsize=queue_depth)
        self.replies = context.Queue(maxsize=1)
        self.writer_factory = writer_factory
        self.writer = self.path = None
        self.proxy_factory = WriterFactory(self.commands, self.replies, os.getpid())

    def poll(self, limit=64):
        for _ in range(limit):
            try:
                key, method, args, kwargs = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                if method == "open":
                    path = Path(args[0]).resolve()
                    if self.path is not None and path != self.path:
                        raise ValueError("replacement cohort changed the TensorBoard run directory")
                    if self.writer is None:
                        self.writer = self.writer_factory(path, enabled=True)
                        self.path = path
                elif method in ("add_text", "add_scalar", "add_image", "flush") and self.writer is not None:
                    getattr(self.writer, method)(*args, **kwargs)
                else:
                    raise ValueError(f"invalid TensorBoard command {method}")
            except Exception as exc:
                if key is not None:
                    self.replies.put_nowait((key, str(exc)))
                raise
            if key is not None:
                self.replies.put_nowait((key, None))

    def close(self):
        # Caller joins GPU writers and receives their final flush first.
        self.poll()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        for channel in (self.commands, self.replies):
            channel.cancel_join_thread()
            channel.close()
