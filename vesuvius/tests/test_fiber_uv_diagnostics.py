from threading import Event,get_ident
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_diagnostics import OrderedDiagnostics


def test_ordered_bounded_publication():
    ready=Event();published=[];main=get_ident()
    def publish(value):
        assert get_ident()==main
        published.append(value)
    def first():
        assert ready.wait(5)
        return 0
    def second():
        ready.set();return 1
    queue=OrderedDiagnostics(2,publish)
    try:
        queue.submit(first);queue.submit(second)
        queue.reserve()
        assert len(queue.pending)<2
        queue.submit(lambda:2)
        assert len(queue.pending)<=2
        queue.close()
    finally:queue.shutdown()
    assert published==[0,1,2]
    assert queue.peak_running==2
    assert queue.max_pending<=2


def test_worker_error_propagates():
    def fail():raise RuntimeError('audit failed')
    queue=OrderedDiagnostics(1,lambda x:None)
    queue.submit(fail)
    with pytest.raises(RuntimeError,match='audit failed'):queue.close()


def test_workers_positive():
    with pytest.raises(ValueError):OrderedDiagnostics(0,lambda x:None)
