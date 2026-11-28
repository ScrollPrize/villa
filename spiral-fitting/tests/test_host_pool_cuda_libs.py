"""host_pool_cuda._libs(): the NVRTC and CUDA driver handles are published TOGETHER, after both loaded (review: a concurrent first caller must never get (handle, None), and a failed second load must leave nothing half-published)."""
import ctypes
import os
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import host_pool_cuda as H  # noqa: E402


def _fake_libs(monkeypatch, cuda_error=None, delay=0.0):
    monkeypatch.setattr(H, "_nvrtc", None)
    monkeypatch.setattr(H, "_cuda", None)
    monkeypatch.setattr(H.glob, "glob", lambda *a, **k: ["/x/libnvrtc.so.12"])

    def cdll(name):
        if name == "libcuda.so.1":
            time.sleep(delay)
            if cuda_error:
                raise cuda_error
        return object()
    monkeypatch.setattr(ctypes, "CDLL", cdll)


def test_a_failed_driver_load_leaves_nothing_published(monkeypatch):
    _fake_libs(monkeypatch, cuda_error=OSError("no libcuda"))
    with pytest.raises(OSError):
        H._libs()
    assert H._nvrtc is None and H._cuda is None


def test_concurrent_first_callers_never_see_a_half_loaded_pair(monkeypatch):
    _fake_libs(monkeypatch, delay=0.05)
    seen, gate = [], threading.Barrier(8)

    def go():
        gate.wait()
        seen.append(H._libs())
    threads = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(seen) == 8 and all(a is not None and b is not None for a, b in seen)
    assert len({id(a) for a, _ in seen}) == 1                  # one load, shared
