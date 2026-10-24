"""Persistent sample ownership on actual cached Paris4 CT/annotations."""

import time
from types import SimpleNamespace

import pytest
import torch

from tests.neural_tracing.test_sample_pipeline import _real_paris4_loader, _assert_same_batch
from vesuvius.neural_tracing.fiber_trace_3d.sample_broker import CanonicalSampleBroker
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import FiberTrace3DSampleQueue, SampleQueueSource


def _queue(tmp_path, slots=5):
    loader = _real_paris4_loader(tmp_path)
    owner = FiberTrace3DSampleQueue(sources={"train": SampleQueueSource(loader.config),
        "test": SampleQueueSource(loader.config)}, workers=2, slot_count=slots,
        initial_in_flight=2, load_affinity_workers=1)
    return owner, loader


def _await(owner, function):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        owner.poll()
        value = function()
        if value is not None:
            return value
        time.sleep(.005)
    raise AssertionError("sample queue stalled")


def test_real_broker_reserves_tests_and_replays_without_reloading(tmp_path):
    owner, loader = _queue(tmp_path)
    broker = CanonicalSampleBroker(owner, effective_batch_size=5, start_step=0,
        max_steps=2, test_interval=1, test_sample_count=5, test_batch_size=2,
        test_start_sample_index=17)
    pids = tuple(p.pid for p in owner.pipeline.processes)
    try:
        for step in range(3):
            if step < 2:
                assert broker.batch(tuple(range(step * 5, step * 5 + 5))) is None
            broker.begin_test(step)
            for ids in ((17, 18), (19, 20), (21,)):
                actual = _await(broker, lambda: broker.test_batch(step, ids))
                expected = loader.load_batch(ids[0])
                if len(ids) == 2:
                    _assert_same_batch(actual, expected)
                else:
                    torch.testing.assert_close(actual.volume[0], expected.volume[0], rtol=0, atol=0)
                broker.release_test(step, ids)
            broker.end_test(step)
            if step == 2:
                break
            ids = tuple(range(step * 5, step * 5 + 5))
            original = _await(broker, lambda: broker.batch(ids))
            owned = original.volume.clone()
            broker.set_worker_limit(0)
            _await(broker, lambda: True if broker.workers_within_limit() else None)
            # New physical partitions read the same leased CPU slots and RNG
            # outputs. No augmentation or volume read is repeated for this step.
            first, rest = broker.batch(ids[:1]), broker.batch(ids[1:])
            torch.testing.assert_close(torch.cat((first.volume, rest.volume)), owned, rtol=0, atol=0)
            assert len(owner.pipeline.leased) == 5
            with pytest.raises(ValueError, match="exactly one"):
                broker.commit(step + 2)
            broker.commit(step + 1)
            broker.set_worker_limit(1)
        assert tuple(p.pid for p in owner.pipeline.processes) == pids
        assert not owner.pipeline.leased
    finally:
        broker.close()


def test_real_overlapping_test_readers_keep_storage_until_last_release(tmp_path):
    owner, _ = _queue(tmp_path, slots=5)
    broker = CanonicalSampleBroker(owner, effective_batch_size=5, start_step=0,
        max_steps=1, test_interval=1, test_sample_count=5, test_batch_size=2,
        test_start_sample_index=17)
    try:
        a = _await(broker, lambda: broker.test_batch(0, (17, 18)))
        b = broker.test_batch(0, (18, 19))
        if b is None:
            b = _await(broker, lambda: broker.test_batch(0, (18, 19)))
        expected = b.volume.clone()
        broker.release_test(0, (17, 18))
        assert 18 in broker._test and broker._test_leases[18] == 1
        broker.poll()
        torch.testing.assert_close(b.volume, expected, rtol=0, atol=0)
        broker.release_test(0, (18, 19))
        for ids in ((20, 21), (17, 18)):
            _await(broker, lambda: broker.test_batch(0, ids))
            broker.release_test(0, ids)
        assert broker._test_released == 5
        with pytest.raises(RuntimeError, match="no live lease"):
            broker.release_test(0, (17,))
        broker.end_test(0)
        assert not owner.pipeline.leased
    finally:
        broker.close()


def test_real_partial_test_consumer_replacement_uses_same_workers(tmp_path):
    owner, _ = _queue(tmp_path)
    broker = CanonicalSampleBroker(owner, effective_batch_size=5, start_step=0,
        max_steps=1, test_interval=1, test_sample_count=5, test_batch_size=2,
        test_start_sample_index=17)
    try:
        _await(broker, lambda: broker.test_batch(0, (17, 18)))
        broker.release_test(0, (17, 18))
        broker.set_worker_limit(0)
        broker.begin_test(0)
        broker.set_worker_limit(1)
        for ids in ((19, 20), (21,), (17, 18)):
            batch = _await(broker, lambda: broker.test_batch(0, ids))
            assert batch.stream_indices.tolist() == list(ids)
            broker.release_test(0, ids)
        assert broker._test_released == 5
        broker.end_test(0)
        assert _await(broker, lambda: broker.batch((0, 1, 2, 3, 4))) is not None
    finally:
        broker.close()


@pytest.mark.parametrize("effective,slots,count,batch", [(5, 4, 0, 1), (4, 4, 10, 8)])
def test_capacity_is_rejected_before_using_workers(effective, slots, count, batch):
    queue = SimpleNamespace(pipeline=SimpleNamespace(max_in_flight=slots), transport=None)
    with pytest.raises(ValueError, match="slots|logical test batch"):
        CanonicalSampleBroker(queue, effective_batch_size=effective, start_step=0,
            max_steps=1, test_sample_count=count, test_batch_size=batch)


def test_real_create_parks_existing_pool_until_authorized(tmp_path):
    from tests.neural_tracing.test_resumable_training import _bounded_config
    from vesuvius.neural_tracing.fiber_trace_3d.loader import config_from_mapping
    raw, path = _bounded_config(tmp_path)
    broker = CanonicalSampleBroker.create(config_path=path, loader_config=config_from_mapping(raw),
        start_step=0, max_steps=4, effective_batch_size=5, workers=2, slot_count=5, initial_in_flight=2)
    try:
        assert broker.pipeline._active_workers.value == 0
        broker.set_worker_limit(2)
        assert _await(broker, lambda: broker.test_batch(0, (5, 6))) is not None
    finally:
        broker.close()
