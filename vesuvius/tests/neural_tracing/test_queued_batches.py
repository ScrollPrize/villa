import threading

from vesuvius.neural_tracing.fiber_trace_3d.queued_batches import (
    QueuedBatchView, TrainingBatchIndices, test_batches as make_test_batches,
)


class Queue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.live = {}
        self.serial = 0

    def submit(self, source, indices):
        if sum(len(v[1]) for v in self.live.values()) + len(indices) > self.capacity:
            return None
        self.serial += 1
        self.live[self.serial] = (source, indices)
        return self.serial

    def poll(self):
        pass

    def batch(self, ticket):
        return self.live[ticket]

    def release(self, ticket):
        del self.live[ticket]


def test_lazy_batch_mapping():
    batches = TrainingBatchIndices(lambda i: i + 100, (3, 1), 4)
    assert [batches[i] for i in range(4)] == [(100, 101, 102), (103,), (104, 105, 106), (107,)]
    assert make_test_batches((0, 2), batch_size=3, sample_count=8, start_sample_index=5) == ((5, 6, 7), (11, 12))


def test_tiny_window_training_test_training_no_priority_inversion():
    queue = Queue(2)
    train = QueuedBatchView(queue, "train", ((0, 1), (2, 3)), limit=1)
    assert next(train) == ("train", (0, 1))
    train.release_current()
    result = []
    reader = threading.Thread(target=lambda: result.append(next(train)))
    reader.start()
    test = QueuedBatchView(queue, "test", ((40, 41),))
    assert next(test) == ("test", (40, 41))
    assert not result
    test.release_current()
    test.close()
    train.set_limit(2)
    reader.join(timeout=2)
    assert not reader.is_alive()
    assert result == [("train", (2, 3))]
    train.close()
    assert not queue.live


def test_close_wakes_reader_at_phase_boundary():
    source = QueuedBatchView(Queue(1), "train", ((0,),), limit=0)
    result = []
    def read():
        result.extend(source)
    thread = threading.Thread(target=read)
    thread.start()
    source.close()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert not result
