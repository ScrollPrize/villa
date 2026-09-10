import time

import pytest

from tests.neural_tracing.test_ordered_sample_pipeline import _shared_pipeline, _TupleTransport
from vesuvius.neural_tracing.fiber_trace_3d import sample_broker


def test_broker_keeps_uncommitted_samples_across_pause_and_repartition(monkeypatch):
    monkeypatch.setattr(_TupleTransport, "batch_tensors", lambda self, ids, **kw: ids, raising=False)
    monkeypatch.setattr(sample_broker, "collate_pipeline_results", lambda rows, volumes, **kw: [r.value for r in rows])
    pipeline = _shared_pipeline(elastic_workers=True, shared_workers=2,
                               shared_stage1_affinity_workers=1, transport=_TupleTransport())
    broker = sample_broker.CanonicalSampleBroker(pipeline, pipeline.transport, effective_batch_size=4)
    try:
        def await_batch(indices):
            deadline = time.monotonic() + 10
            while True:
                broker.poll()
                batch = broker.batch(indices)
                if batch is not None:
                    return batch
                assert time.monotonic() < deadline
                time.sleep(0.01)

        assert await_batch((0, 1, 2, 3)) == [20, 22, 24, 26]
        broker.set_worker_limit(0)
        deadline = time.monotonic() + 10
        while not broker.workers_within_limit():
            broker.poll()
            assert time.monotonic() < deadline
            time.sleep(0.01)
        # A replacement cohort can request different physical batches without
        # recomputing, releasing or skipping the effective batch's samples.
        assert broker.batch((0,)) == [20]
        assert broker.batch((1, 2, 3)) == [22, 24, 26]
        assert set(range(4)) <= pipeline.leased
        with pytest.raises(ValueError, match="exactly one"):
            broker.commit(2)
        broker.commit(1)
        assert not set(range(4)) & pipeline.leased
        with pytest.raises(ValueError, match="outside"):
            broker.batch((0,))
        broker.set_worker_limit(1)
        for step in (2, 3):
            ids = tuple(range((step - 1) * 4, step * 4))
            assert await_batch(ids) == [(i + 10) * 2 for i in ids]
            assert len(broker.ready) <= pipeline.max_in_flight
            broker.commit(step)
        assert not broker.ready and not pipeline.leased
    finally:
        broker.close()
