"""The CT chunk cache's advisory lock (shared/io.lock_exclusive) on every platform (Linux/macOS flock, Windows msvcrt):
a second handle cannot take it while the first holds it, in this process or in another, and takes it once the first
handle is closed."""
import multiprocessing as mp
import threading

import pytest

from vesuvius.neural_tracing.tracers.shared.io import lock_exclusive


def _lock_in_child(path, blocking, queue):
    with open(path, 'a+b') as handle:
        try:
            lock_exclusive(handle, blocking=blocking)
            queue.put('locked')
        except BlockingIOError:
            queue.put('busy')


def _child(path, blocking):
    context = mp.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=_lock_in_child, args=(str(path), blocking, queue))
    process.start()
    result = queue.get(timeout=120)
    process.join(30)
    return result


def test_second_handle_is_refused_then_admitted(tmp_path):
    path = tmp_path/'chunk.lock'
    with open(path, 'a+b') as first:
        lock_exclusive(first)
        with open(path, 'a+b') as second:
            with pytest.raises(BlockingIOError):
                lock_exclusive(second, blocking=False)
    with open(path, 'a+b') as again:
        lock_exclusive(again)


def test_other_process_is_refused_then_admitted(tmp_path):
    path = tmp_path/'chunk.lock'
    with open(path, 'a+b') as first:
        lock_exclusive(first)
        assert _child(path, blocking=False) == 'busy'
    assert _child(path, blocking=True) == 'locked'


def test_blocking_lock_waits_for_release(tmp_path):
    path = tmp_path/'chunk.lock'
    order = []
    first = open(path, 'a+b')
    lock_exclusive(first)

    def waiter():
        with open(path, 'a+b') as handle:
            lock_exclusive(handle)
            order.append('waiter')

    thread = threading.Thread(target=waiter)
    thread.start()
    thread.join(.5)
    assert thread.is_alive()
    order.append('released')
    first.close()
    thread.join(60)
    assert order == ['released', 'waiter']
