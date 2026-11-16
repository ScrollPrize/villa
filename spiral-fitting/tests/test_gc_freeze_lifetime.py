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


def test_main_closes_its_context_on_success_and_on_failure(monkeypatch):
    """Review: the headless entry point must not leave the context (and its share of the process-wide freeze) behind."""
    created = []

    class Context:
        def __init__(self, *args, fail=False, **kwargs):
            self.fail, self.closed = fail, False
            created.append(self)

        def run(self):
            if self.fail:
                raise RuntimeError("fit failed")
            return "ok"

        def close(self):
            self.closed = True

    monkeypatch.setattr(fs, "FitContext", Context)
    assert fs.main(None, scroll=None, paths=None) == "ok" and created[-1].closed
    monkeypatch.setattr(fs, "FitContext", lambda *a, **k: Context(fail=True))
    with pytest.raises(RuntimeError, match="fit failed"):
        fs.main(None, scroll=None, paths=None)
    assert created[-1].closed


def test_the_pressure_policy_watches_the_fits_own_device(monkeypatch):
    """Review: with FIT_SPIRAL_DEVICE=cuda:1 the counters of GPU 0 (the thread's current device) must not decide when to collect."""
    import types
    import torch
    seen, collected = [], []
    monkeypatch.setattr(fs, "_GC_POLICY", "pressure")
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: (seen.append(device), 90)[1])
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: types.SimpleNamespace(total_memory=100))
    monkeypatch.setattr(fs.gc, "collect", lambda *a: collected.append(1))
    on_card_1 = types.SimpleNamespace(device=torch.device("cuda:1"))
    fs.FitContext._collect_garbage_after_step(on_card_1, 1)
    assert seen and all(d == torch.device("cuda:1") for d in seen) and collected     # 90 allocated of 100, fraction 0.2: collect
    seen.clear(); collected.clear()
    fs.FitContext._collect_garbage_after_step(types.SimpleNamespace(device=torch.device("cpu")), 1)
    assert not seen and not collected                                                 # a CPU fit never looks at a GPU


def test_a_non_cuda_fit_never_touches_cuda_counters(monkeypatch):
    """MPS / CPU fits (no CUDA): neither the pressure policy nor the export guard may call into torch.cuda."""
    import types
    import torch

    def boom(*args, **kwargs):
        raise AssertionError("torch.cuda was called by a non-CUDA fit")

    monkeypatch.setattr(torch.cuda, "memory_allocated", boom)
    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    monkeypatch.setattr(torch.cuda, "empty_cache", boom)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(fs, "_GC_POLICY", "pressure")
    for kind in ("mps", "cpu"):
        fs.FitContext._collect_garbage_after_step(types.SimpleNamespace(device=torch.device(kind)), 1)

    class Optimiser:
        def __init__(self):
            self.state, self.cleared = {"x": 1}, False

        def zero_grad(self, set_to_none=True):
            self.cleared = True

    optimiser = Optimiser()
    fs.FitContext._free_fit_state_for_export(types.SimpleNamespace(optimiser=optimiser))
    assert optimiser.state == {} and optimiser.cleared
