"""Decision-batch prefetch preserves update boundaries, overlaps loading and propagates failures."""
import threading

import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.regression.train import DecisionBatchPrefetch


def test_prefetch_preserves_boundaries_prepares_next_update_and_propagates_failures():
    ready = threading.Event()
    chunks = [dict(hist=torch.zeros(12,1,3)) for _ in range(4)]
    def loader():
        for i, chunk in enumerate(chunks):
            if i == 3:
                ready.set()
            yield chunk
    prefetch = DecisionBatchPrefetch(iter(loader()), 2)
    try:
        first = next(prefetch)
        assert all(a is b for a, b in zip(first, chunks[:2])) and len(first) == 2
        assert sum(len(chunk['hist']) for chunk in first) == 24
        assert ready.wait(5), 'Next update must be prepared before requesting it'
        second = next(prefetch)
        assert all(a is b for a, b in zip(second, chunks[2:])) and len(second) == 2
        with pytest.raises(StopIteration):
            next(prefetch)
    finally:
        prefetch.close()
    assert all(not thread.is_alive() for thread in prefetch.executor._threads)
    def failing():
        raise ValueError('invalid sample')
        yield
    prefetch = DecisionBatchPrefetch(iter(failing()), 1)
    try:
        with pytest.raises(ValueError, match='invalid sample'):
            next(prefetch)
    finally:
        prefetch.close()
    with pytest.raises(StopIteration):
        next(prefetch)
