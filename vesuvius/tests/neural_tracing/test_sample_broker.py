import time

import pytest
import torch
from types import SimpleNamespace

from tests.neural_tracing.test_fiber_trace_3d import _loader_with_mapping
from vesuvius.neural_tracing.fiber_trace_3d.sample_broker import CanonicalSampleBroker
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import FiberTrace3DSampleQueue, SampleQueueSource


def _queue(slots=4):
    loader = _loader_with_mapping({"augment_enabled": False, "batch_size": 2,
                                   "patch_shape_zyx": [8, 8, 8]})
    owner = FiberTrace3DSampleQueue(sources={
        "train": SampleQueueSource(loader.config),
        "historical": SampleQueueSource(loader.config),
        "test": SampleQueueSource(loader.config)}, workers=2,
        slot_count=slots, initial_in_flight=2, load_affinity_workers=1)
    return owner, loader


def _await(owner, function):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        owner.poll()
        value = function()
        if value is not None:
            return value
        time.sleep(0.005)
    raise AssertionError("sample queue stalled")


def test_shared_pool_switches_sources_without_recreating_workers_or_changing_seeds():
    owner, loader = _queue()
    pids = [process.pid for process in owner.pipeline.processes]
    try:
        for source, start in (("test", 11), ("train", 0), ("test", 11), ("train", 2)):
            ticket = owner.submit(source, range(start, start + 2))
            actual = _await(owner, lambda: owner.batch(ticket))
            expected = loader.load_batch(start, sample_mode="random", device="cpu")
            assert actual.stream_indices.tolist() == [start, start + 1]
            torch.testing.assert_close(actual.volume, expected.volume)
            owner.release(ticket)
        assert [process.pid for process in owner.pipeline.processes] == pids
    finally:
        owner.close()


def test_broker_preserves_leases_and_reserves_test_execution_order_at_tiny_capacity():
    owner, _ = _queue()
    broker = CanonicalSampleBroker(owner, effective_batch_size=4, start_step=0,
        max_steps=2, test_interval=1, test_sample_count=4, test_batch_size=2,
        test_start_sample_index=17)
    try:
        for step in range(3):
            # Later train requests cannot take even one slot before evaluation.
            if step < 2:
                assert broker.batch(tuple(range(step * 4, step * 4 + 4))) is None
            for ids in ((17, 18), (19, 20)):
                batch = _await(broker, lambda: broker.test_batch(step, ids))
                assert batch.stream_indices.tolist() == list(ids)
                broker.release_test(step, ids)
            broker.end_test(step)
            if step == 2:
                break
            ids = tuple(range(step * 4, step * 4 + 4))
            batch = _await(broker, lambda: broker.batch(ids))
            assert batch.stream_indices.tolist() == list(ids)
            broker.set_worker_limit(0)
            # Repartitioning the same retained update needs no reload.
            assert broker.batch(ids[:1]).stream_indices.tolist() == list(ids[:1])
            assert broker.batch(ids[1:]).stream_indices.tolist() == list(ids[1:])
            with pytest.raises(ValueError, match="exactly one"):
                broker.commit(step + 2)
            broker.commit(step + 1)
            broker.set_worker_limit(1)
        assert not owner.pipeline.leased
    finally:
        broker.close()


def test_historical_source_is_distinct_and_survives_partial_resume():
    owner, _ = _queue()
    broker = CanonicalSampleBroker(owner, effective_batch_size=4, start_step=0,
        max_steps=1, test_interval=1, test_sample_count=2, test_batch_size=2,
        test_sources={"historical": (3, 0)})
    try:
        with pytest.raises(RuntimeError, match="unconsumed"):
            broker.select_test_source(0, "historical")
        _await(broker, lambda: broker.test_batch(0, (0, 1)))
        broker.release_test(0, (0, 1))
        broker.select_test_source(0, "historical")
        assert broker._test_released == 0
        _await(broker, lambda: broker.test_batch(0, (0, 1)))
        broker.release_test(0, (0, 1))
        broker.begin_test(0)
        broker.select_test_source(0, "historical")
        assert broker._test_released == 2
        _await(broker, lambda: broker.test_batch(0, (2,)))
        broker.release_test(0, (2,))
        broker.end_test(0)
        assert broker.test_source == "test" and broker.test_count == 2
    finally:
        broker.close()


def test_partial_test_resume_and_repeated_visualization_do_not_double_consume():
    owner, _ = _queue()
    broker = CanonicalSampleBroker(owner, effective_batch_size=4, start_step=0,
        max_steps=1, test_interval=1, test_sample_count=5, test_batch_size=2,
        test_start_sample_index=17)
    try:
        broker.begin_test(0)
        _await(broker, lambda: broker.test_batch(0, (17, 18)))
        broker.release_test(0, (17, 18))
        broker.set_worker_limit(0)
        # A replacement GPU cohort resumes only remaining groups; it reuses
        # the same canonical owner and can repeat a completed visualization.
        broker.begin_test(0)
        broker.set_worker_limit(1)
        for ids in ((19, 20), (21,), (17, 18)):
            batch = _await(broker, lambda: broker.test_batch(0, ids))
            assert batch.stream_indices.tolist() == list(ids)
            broker.release_test(0, ids)
        assert broker._test_released == 5
        broker.end_test(0)
        assert not owner.pipeline.leased
        assert _await(broker, lambda: broker.batch((0, 1, 2, 3))) is not None
    finally:
        broker.close()


def test_logical_test_group_must_fit_credits_before_workers_are_used():
    queue = SimpleNamespace(pipeline=SimpleNamespace(max_in_flight=4), transport=None)
    with pytest.raises(ValueError, match="logical test batch"):
        CanonicalSampleBroker(queue, effective_batch_size=4, start_step=0, max_steps=1,
                              test_sample_count=10, test_batch_size=8)
    # A smaller final-only group and no evaluation need no extra slots.
    for count in (0, 3):
        CanonicalSampleBroker(queue, effective_batch_size=4, start_step=0, max_steps=1,
                              test_sample_count=count, test_batch_size=8)
