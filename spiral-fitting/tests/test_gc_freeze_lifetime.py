"""gc.freeze() is process-wide: its lifetime must end with the last context that took it (review of the GC freeze policy)."""
import gc
import weakref

import pytest

import fit_spiral as fs


@pytest.fixture(autouse=True)
def _no_leaked_freeze(monkeypatch):
    gc.unfreeze()
    fs._GC_FREEZE_OWNERS.clear()
    monkeypatch.setattr(fs, "_GC_POLICY", "freeze")
    monkeypatch.setattr(fs, "_GC_EVERY_STEPS", 1)
    yield
    fs._GC_FREEZE_OWNERS.clear()
    gc.unfreeze()


class _Cycle:
    def __init__(self):
        self.me = self                     # a reference cycle: only the cyclic collector can free it


def _context():
    import types
    import torch
    return types.SimpleNamespace(device=torch.device("cpu"))


def _step(context):
    fs.FitContext._collect_garbage_after_step(context, 1)


def test_any_release_ends_the_freeze_for_everybody_and_the_others_refreeze_at_their_next_step():
    a, b = _context(), _context()
    _step(a); _step(b)
    assert gc.get_freeze_count() > 0 and a._gc_epoch == b._gc_epoch == fs._gc_freeze_epoch()
    fs._gc_freeze_release(a)
    assert gc.get_freeze_count() == 0 and b._gc_epoch != fs._gc_freeze_epoch()
    _step(b)                               # the other session notices the new epoch and freezes again
    assert gc.get_freeze_count() > 0 and b._gc_epoch == fs._gc_freeze_epoch()


def test_release_is_a_no_op_only_when_no_freeze_is_active_and_a_stranger_ends_an_active_one():
    owner, stranger = _context(), _context()
    epoch = fs._gc_freeze_epoch()
    fs._gc_freeze_release(stranger)                  # nothing frozen: nothing to do
    assert fs._gc_freeze_epoch() == epoch and gc.get_freeze_count() == 0
    _step(owner)
    fs._gc_freeze_release(stranger)                  # a context that never stepped still thaws an active freeze (its heap may be inside it)
    assert gc.get_freeze_count() == 0 and fs._gc_freeze_epoch() == epoch + 1
    fs._gc_freeze_release(owner)                     # already thawed: idempotent
    assert fs._gc_freeze_epoch() == epoch + 1


def test_a_context_frozen_by_another_session_before_its_first_step_still_thaws_its_cycles_on_close():
    """Review: C froze while B was alive but had not stepped; closing B must free B's discarded cycles."""
    b_cycle = _Cycle()
    ref = weakref.ref(b_cycle)
    b, c = _context(), _context()                    # B exists (heap alive) but never steps
    _step(c)                                         # C's first step freezes everything alive, including B's cycle
    del b_cycle
    gc.collect()
    assert ref() is not None
    fs._gc_freeze_release(b)                         # B is not a registered owner, yet its teardown must end the freeze
    assert ref() is None


def test_a_context_whose_registration_was_cleared_while_alive_still_thaws_on_close():
    """Review: after another context's release cleared the owners, a session that was re-frozen by a third one must still thaw."""
    a, b, c = _context(), _context(), _context()
    _step(a); _step(b)
    fs._gc_freeze_release(a)                         # clears every registration, bumps the epoch
    b_cycle = _Cycle(); ref = weakref.ref(b_cycle)   # B's discarded stage, alive when C freezes
    _step(c)                                         # C re-freezes (B has not stepped since)
    del b_cycle
    gc.collect()
    assert ref() is not None
    fs._gc_freeze_release(b)
    assert ref() is None


def test_a_rebuilt_context_frees_its_discarded_cycles_while_another_context_is_active():
    """Review: two contexts' models alive at the first freeze; rebuilding one must not leave its discarded cycles frozen."""
    discarded = _Cycle()                   # context A's old model stage, alive when the freeze is taken
    ref = weakref.ref(discarded)
    a, b = _context(), _context()
    _step(a); _step(b)
    del discarded
    gc.collect()
    assert ref() is not None               # frozen: the collector cannot see it
    fs._gc_freeze_release(a)               # A rebuilds; B is still a live session
    assert ref() is None                   # freed before A allocates its replacement


def test_a_session_that_starts_after_the_first_freeze_is_frozen_at_its_first_step():
    """Review: a second session's setup heap must not be scanned by every later per-step collection."""
    first, late = _context(), _context()
    _step(first)
    setup_heap = _Cycle()                  # the late session's setup objects, allocated after the first freeze
    ref = weakref.ref(setup_heap)
    _step(late)                            # its first step freezes what exists now, including that heap
    del setup_heap
    gc.collect()
    assert ref() is not None               # it was frozen, so the per-step collections do not scan it


def _bare_context(store):
    context = fs.FitContext.__new__(fs.FitContext)
    context._lasagna_store = store
    context.winding_inference = object()
    return context


def test_closing_a_context_ends_its_freeze():
    context = _bare_context(None)
    context.device = None
    context._gc_epoch = fs._gc_freeze_acquire(context)
    context.close()
    assert gc.get_freeze_count() == 0 and context._gc_epoch is None


def test_a_failing_store_close_still_ends_the_freeze():
    class Store:
        def close(self):
            raise RuntimeError("store close failed")

    context = _bare_context(Store())
    context._gc_epoch = fs._gc_freeze_acquire(context)
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


def test_rebuild_and_close_forget_the_pressure_watermark_together_with_the_freeze_epoch():
    """Review: after a rebuild to a smaller model the pressure policy must establish a fresh baseline, not wait for the old model's watermark."""
    import inspect
    import types
    ctx = types.SimpleNamespace(_gc_epoch=3, _gc_floor_bytes=10 ** 9)
    fs.FitContext._reset_gc_state(ctx)
    assert ctx._gc_epoch is None and ctx._gc_floor_bytes == 0
    for name in ("rebuild_model_state", "close"):
        assert "_reset_gc_state" in inspect.getsource(getattr(fs.FitContext, name)), name
