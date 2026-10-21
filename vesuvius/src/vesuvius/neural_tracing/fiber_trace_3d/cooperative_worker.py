"""Cancellable worker views over the persistent CPU sample owner.

Extracted transport only: no model, manager registration or cohort launcher.
Closing a worker view never closes the shared loading pool.
"""

import os
import queue
import threading


class QueueBatchSource:
    """Cancellable input for the existing asynchronous CUDA batch preparer.

    CPU volume slots remain owned by the persistent broker through global
    update completion. Closing this view never closes the shared loading pool.
    """

    def __init__(self, batches, received, owner_pid, index_map, test_connection=None):
        self.batches, self.received, self.owner_pid = batches, received, owner_pid
        self.stopped = threading.Event()
        self.index_map = index_map
        self.ordinal = self.samples = 0
        self.test_connection = test_connection

    def test_batches(self, *, step, global_batch_indices, batch_size, start_sample_index=0,
                     sample_count=None):
        if self.test_connection is None:
            raise RuntimeError("managed evaluation requires the shared sample broker")
        return _QueueTestBatches(self, step, global_batch_indices, batch_size, start_sample_index, sample_count)

    def _test_boundary(self, kind, step):
        connection = self.test_connection
        if connection is None:
            raise RuntimeError("managed tests require shared sample phase coordination")
        connection.send((kind, step, ()))
        while not connection.poll(0.1):
            if self.stopped.is_set():
                raise RuntimeError("sample source closed during test phase transition")
            if os.getppid() != self.owner_pid:
                raise RuntimeError("sample supervisor exited during test phase transition")
        if connection.recv() != (kind, step):
            raise RuntimeError("invalid sample phase acknowledgement")

    def begin_test(self, step):
        self._test_boundary("begin-test", step)

    def end_test(self, step):
        self._test_boundary("end-test", step)

    def __iter__(self):
        return self

    def __next__(self):
        while not self.stopped.is_set():
            if os.getppid() != self.owner_pid:
                raise RuntimeError("training supervisor exited while awaiting CPU samples")
            try:
                batch = self.batches.get(timeout=0.1)
            except queue.Empty:
                continue
            with self.received.get_lock():
                self.received.value += 1
            sizes = self.index_map.batch_sizes
            if not sizes:
                raise RuntimeError("zero-work rank received a training sample")
            size = sizes[self.ordinal % len(sizes)]
            expected = [self.index_map(self.samples + i) for i in range(size)]
            if batch.stream_indices.tolist() != expected:
                raise RuntimeError("CPU input differs from the canonical training sample sequence")
            self.ordinal += 1
            self.samples += size
            return batch
        raise StopIteration

    def release_current(self):
        # Physical H2D completion is not effective-batch consumption. The broker
        # retains these CPU slots so a cohort change can discard GPU lookahead.
        pass

    def close(self):
        self.stopped.set()


class _QueueTestBatches:
    def __init__(self, source, step, groups, batch_size, offset, sample_count):
        self.source, self.step = source, step
        self.groups = iter(groups)
        self.batch_size, self.offset = batch_size, offset
        self.current = None
        self.sample_count = sample_count

    def __iter__(self):
        return self

    def __next__(self):
        if self.current is not None:
            raise RuntimeError("previous test batch has not been released")
        group = next(self.groups)
        stop = (group + 1) * self.batch_size
        if self.sample_count is not None:
            stop = min(stop, self.sample_count)
        indices = tuple(range(self.offset + group * self.batch_size, self.offset + stop))
        connection = self.source.test_connection
        connection.send(("test", self.step, indices))
        while not connection.poll(0.1):
            if self.source.stopped.is_set():
                raise StopIteration
            if os.getppid() != self.source.owner_pid:
                raise RuntimeError("sample supervisor exited during evaluation loading")
        batch = connection.recv()
        if batch.stream_indices.tolist() != list(indices):
            raise RuntimeError("test data differs from canonical sample sequence")
        self.current = indices
        return batch

    def release_current(self):
        if self.current is not None:
            self.source.test_connection.send(("release-test", self.step, self.current))
            self.current = None

    def close(self):
        self.release_current()
