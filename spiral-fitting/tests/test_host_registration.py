"""brick_pool_offload._HostRegistration: teardown of page-locked pool memory (no CUDA device needed: the driver and the device context are stubbed).

Review of the offloaded pools: unregister must run on the OWNING card's context after that card's outstanding gathers finished, a refused or interrupted release must keep the
backing memory (and stay retryable), and a registration the driver will not release at garbage-collection time is parked, not freed."""
import contextlib
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import brick_pool_offload as bpo                      # noqa: E402
import host_pool_cuda                                 # noqa: E402


class FakeDriver:
    def __init__(self, events, results):
        self.events, self.results = events, list(results)

    def cuMemHostUnregister(self, ptr):
        self.events.append(("unregister", current_device(self.events)))
        r = self.results.pop(0) if self.results else 0
        if isinstance(r, BaseException):
            raise r
        return r


def current_device(events):
    entered = [e[1] for e in events if e[0] == "enter"]
    left = [e for e in events if e[0] == "leave"]
    return entered[len(entered) - 1 - len(left)] if len(entered) > len(left) else None


@pytest.fixture()
def driver(monkeypatch):
    events = []

    @contextlib.contextmanager
    def device(index):
        events.append(("enter", index))
        try:
            yield
        finally:
            events.append(("leave", index))
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda index=None: events.append(("sync", index)))
    box = {"driver": FakeDriver(events, [])}
    monkeypatch.setattr(host_pool_cuda, "_libs", lambda: (None, box["driver"]))
    monkeypatch.setattr(bpo, "_PARKED", [])
    return events, box


def make(dev_index=1):
    return bpo._HostRegistration(np.zeros(64, dtype=np.uint8), dev_index)


def test_release_runs_on_the_owning_card_after_its_gathers_finished(driver):
    events, _ = driver
    reg = make(dev_index=1)
    reg.release()
    names = [e[0] for e in events]
    assert names == ["enter", "sync", "unregister", "leave"], names
    assert ("sync", 1) in events and ("unregister", 1) in events, "released outside card 1's context"
    assert reg.array is None


def test_a_refused_release_keeps_the_backing_and_can_be_retried(driver):
    events, box = driver
    box["driver"].results = [201, 0]                  # CUDA_ERROR_INVALID_CONTEXT, then the driver recovers
    reg = make()
    backing = reg.array
    with pytest.raises(RuntimeError, match="failed with 201"):
        reg.release()
    assert reg.array is backing, "the backing array was dropped although the driver still has it registered"
    reg.release()
    assert reg.array is None
    reg.release()                                     # idempotent after success
    assert [e for e in events if e[0] == "unregister"].__len__() == 2


def test_not_registered_counts_as_released(driver):
    _, box = driver
    box["driver"].results = [713]                     # CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED
    reg = make()
    reg.release()
    assert reg.array is None


def test_an_interrupted_release_keeps_the_backing(driver):
    _, box = driver
    box["driver"].results = [KeyboardInterrupt(), 0]
    reg = make()
    backing = reg.array
    with pytest.raises(KeyboardInterrupt):
        reg.release()
    assert reg.array is backing
    reg.release()
    assert reg.array is None


def test_a_registration_the_driver_refuses_at_finalization_is_parked_and_recoverable(driver):
    _, box = driver
    box["driver"].results = [201]
    reg = make()
    backing = reg.array
    bpo._release_or_park(reg)                         # what the weakref finalizer runs
    assert bpo._PARKED == [reg] and reg.array is backing, "the refused registration must stay alive"
    assert bpo.retry_parked_registrations() == 0      # the driver can release it now
    assert bpo._PARKED == [] and reg.array is None


def test_retry_keeps_registrations_that_are_still_refused(driver):
    _, box = driver
    box["driver"].results = [201, 201]
    reg = make()
    bpo._release_or_park(reg)
    assert bpo.retry_parked_registrations() == 1 and reg.array is not None


class RegDriver(FakeDriver):
    """driver stub for register(): cuMemHostRegister_v2 / cuMemHostGetDevicePointer_v2 return 0 and are recorded; the pointer lookup can be made to raise."""
    def __init__(self, events, results, lookup_raises=None):
        super().__init__(events, results)
        self.lookup_raises = lookup_raises

    def cuMemHostRegister_v2(self, ptr, nbytes, flags):
        self.events.append(("register", current_device(self.events)))
        return 0

    def cuMemHostGetDevicePointer_v2(self, dptr_ref, ptr, flags):
        if self.lookup_raises is not None:
            raise self.lookup_raises
        dptr_ref._obj.value = 4096
        return 0


def test_an_interrupt_right_after_a_successful_registration_keeps_the_backing_memory_alive(driver, monkeypatch):
    """review of #2588: cuMemHostRegister returned (the driver now holds the range), then SIGINT hit before the pointer lookup: the array must stay owned, never be freed under the driver."""
    events, box = driver
    box["driver"] = RegDriver(events, [KeyboardInterrupt()])                      # the interrupt also hits the first unregister attempt
    def interrupted_chk(r, what):                                                  # SIGINT delivered right after cuMemHostRegister returned 0, BEFORE the lookup's try block
        raise KeyboardInterrupt()
    monkeypatch.setattr(host_pool_cuda, "_chk", interrupted_chk)
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: None)
    reg = make(0)
    with pytest.raises(KeyboardInterrupt):
        reg.register()
    assert ("register", 0) in events
    assert reg in bpo._PARKED and reg.array is not None, "interrupted unregister: the registration stays parked and keeps its array"
    box["driver"].results.clear()
    assert bpo.retry_parked_registrations() == 0 and reg.array is None, "a later retry releases it and only then drops the array"


def test_a_failure_after_registration_releases_it_at_once_and_leaves_nothing_parked(driver, monkeypatch):
    events, box = driver
    box["driver"] = RegDriver(events, [0], lookup_raises=RuntimeError("lookup failed"))
    monkeypatch.setattr(host_pool_cuda, "_chk", lambda r, what: None)
    monkeypatch.setattr(torch, "zeros", lambda *a, **k: None)
    reg = make(0)
    with pytest.raises(RuntimeError, match="lookup failed"):
        reg.register()
    assert ("unregister", 0) in events and reg not in bpo._PARKED and reg.array is None


def test_session_close_retries_a_refused_release_and_raises_until_it_succeeds():
    """review of #2588: the fitter thread's context.close() failed (the driver refused to unregister a pinned pool) but the thread just died and close() looked successful, so a rebuild reached Idle
    while the old pool was still pinned. close() must retry the release and raise RuntimeError until it works, keeping the context for the next retry."""
    import threading
    import spiral_runtime as sr
    cls = next(c for c in vars(sr).values() if isinstance(c, type) and "close" in vars(c) and "_enter_epoch_locked" in dir(c) and hasattr(c, "install_inputs"))
    s = object.__new__(cls)
    s._condition = threading.Condition(); s._enter_epoch_locked = lambda *a, **k: None; s._shutdown = False; s.session_generation = 0; s._commands = []
    s._thread = threading.Thread(target=lambda: None); s._thread.start(); s._thread.join()

    class Ctx:
        def __init__(self, fail): self.fail, self.calls = fail, 0
        def close(self):
            self.calls += 1
            if self.calls <= self.fail:
                raise RuntimeError("cuMemHostUnregister failed with 700")
    ctx = Ctx(fail=2)
    s._unreleased_context = ctx; s._release_error = "RuntimeError: cuMemHostUnregister failed with 700"       # what the fitter thread leaves behind
    with pytest.raises(RuntimeError, match="still held"):
        s.close()
    assert s._unreleased_context is ctx and ctx.calls == 1, "the context is kept for the next retry"
    with pytest.raises(RuntimeError, match="still held"):
        s.close()
    s.close()                                                                       # the driver can release it now
    assert s._unreleased_context is None and s._release_error is None and ctx.calls == 3
