"""gc.freeze() is process-wide: its lifetime must end with the last context that took it (review of the GC freeze policy)."""
import gc
import weakref

import pytest

import fit_spiral as fs


@pytest.fixture(autouse=True)
def _no_leaked_freeze():
    gc.unfreeze()
    fs._GC_FREEZE_OWNERS.clear()
    yield
    fs._GC_FREEZE_OWNERS.clear()
    gc.unfreeze()


class _Cycle:
    def __init__(self):
        self.me = self                     # a reference cycle: only the cyclic collector can free it


def test_the_freeze_ends_when_the_last_owner_lets_go():
    a, b = object(), object()
    fs._gc_freeze_acquire(a)
    assert gc.get_freeze_count() > 0
    fs._gc_freeze_acquire(b)
    fs._gc_freeze_release(a)
    assert gc.get_freeze_count() > 0       # another session still holds the process-wide freeze
    fs._gc_freeze_release(b)
    assert gc.get_freeze_count() == 0


def test_release_is_idempotent_and_ignores_strangers():
    owner = object()
    fs._gc_freeze_release(owner)           # never froze
    fs._gc_freeze_acquire(owner)
    fs._gc_freeze_release(owner)
    fs._gc_freeze_release(owner)
    assert gc.get_freeze_count() == 0


def test_garbage_frozen_by_a_closed_session_is_collected_at_the_end_of_the_freeze():
    owner = object()
    cycle = _Cycle()                       # alive when the freeze is taken, like the session <-> context cycle of a resident fit
    ref = weakref.ref(cycle)
    fs._gc_freeze_acquire(owner)
    del cycle                              # the session closes: only the cycle keeps it now
    gc.collect()
    assert ref() is not None               # frozen: the collector cannot see it
    fs._gc_freeze_release(owner)
    assert ref() is None                   # the release collects what the freeze hid


def _bare_context(store):
    context = fs.FitContext.__new__(fs.FitContext)
    context._lasagna_store = store
    context.winding_inference = object()
    return context


def test_closing_a_context_ends_its_freeze():
    context = _bare_context(None)
    fs._gc_freeze_acquire(context)
    context._gc_frozen = True
    context.close()
    assert gc.get_freeze_count() == 0 and context._gc_frozen is False


def test_a_failing_store_close_still_ends_the_freeze():
    class Store:
        def close(self):
            raise RuntimeError("store close failed")

    context = _bare_context(Store())
    fs._gc_freeze_acquire(context)
    context._gc_frozen = True
    with pytest.raises(RuntimeError, match="store close failed"):
        context.close()
    assert gc.get_freeze_count() == 0
