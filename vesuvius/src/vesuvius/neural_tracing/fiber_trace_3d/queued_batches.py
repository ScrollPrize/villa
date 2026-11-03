"""Ordered batch views over the shared, source-tagged sample worker queue.

Views own no workers. A training admission boundary leaves queue credits for the
next evaluation, including when CUDA preparation is waiting for future training.
"""

import threading


class QueuedBatchView:
    def __init__(self, owner, source, batches, *, limit=None):
        self.owner, self.source = owner, source
        self.batches = batches
        self.count = len(batches)
        self.limit = self.count if limit is None else int(limit)
        self.ordinal = 0
        self.admitted = 0
        self.pending = {}
        self.current = None
        self.stopped = threading.Event()
        self.lock = threading.RLock()

    def __iter__(self):
        return self

    def set_limit(self, limit):
        with self.lock:
            if not self.admitted <= limit <= self.count:
                raise ValueError("batch admission limit cannot retract admitted work")
            self.limit = int(limit)

    def release_current(self):
        with self.lock:
            if self.current is not None:
                self.owner.release(self.current)
                self.current = None

    def __next__(self):
        self.release_current()
        while not self.stopped.is_set():
            with self.lock:
                if self.ordinal >= self.count:
                    raise StopIteration
                while self.admitted < self.limit:
                    ticket = self.owner.submit(self.source, self.batches[self.admitted])
                    if ticket is None:
                        break
                    self.pending[self.admitted] = ticket
                    self.admitted += 1
                ticket = self.pending.get(self.ordinal)
                if ticket is not None:
                    self.owner.poll()
                    batch = self.owner.batch(ticket)
                    if batch is not None:
                        self.current = self.pending.pop(self.ordinal)
                        self.ordinal += 1
                        return batch
            # Never retain a lock while waiting for another phase to release
            # credits, or while main-thread evaluation uses the same queue.
            self.stopped.wait(0.005)
        raise StopIteration

    def close(self):
        self.stopped.set()
        with self.lock:
            self.release_current()
            # Pending tasks may still be executing. Owner shutdown/reclamation
            # must join/drain them; a view never releases an in-flight slot.


class TrainingBatchIndices:
    """Lazy finite physical-batch mapping; no global job list allocation."""

    def __init__(self, index_map, batch_sizes, batch_count):
        self.index_map = index_map
        self.sizes = tuple(batch_sizes)
        self.count = int(batch_count)
        self.prefix = [0]
        for size in self.sizes:
            self.prefix.append(self.prefix[-1] + size)
        if self.count and not self.sizes:
            raise ValueError("nonempty batch stream requires a batch-size pattern")

    def __len__(self):
        return self.count

    def __getitem__(self, ordinal):
        if not 0 <= ordinal < self.count:
            raise IndexError(ordinal)
        cycle, part = divmod(ordinal, len(self.sizes))
        offset = cycle * self.prefix[-1] + self.prefix[part]
        return tuple(self.index_map(offset + i) for i in range(self.sizes[part]))


def test_batches(indices, *, batch_size, sample_count, start_sample_index):
    return tuple(
        tuple(range(start_sample_index + index * batch_size,
                    start_sample_index + min((index + 1) * batch_size, sample_count)))
        for index in indices
    )
