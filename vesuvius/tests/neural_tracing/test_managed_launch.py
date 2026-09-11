"""Training defaults must never silently bypass cooperative allocation."""
import sys

import pytest

from vesuvius.neural_tracing.fiber_trace_3d import train, cooperative_training


def test_default_dispatch_is_managed_and_errors_do_not_fallback(monkeypatch):
    monkeypatch.setattr(train, "_require_single_process_cli_mode", lambda _: None)
    calls = []
    def managed(path, **kwargs):
        calls.append((path, kwargs))
        raise ValueError("missing memory calibration")
    monkeypatch.setattr(cooperative_training, "run_cooperative_training", managed)
    monkeypatch.setattr(train, "training_units", lambda *a, **k: pytest.fail("unmanaged fallback"))
    with pytest.raises(ValueError, match="missing memory calibration"):
        train.run_training("config.json")
    assert calls[0][1]["socket"] is None


def test_explicit_unmanaged_dispatch(monkeypatch):
    calls = []
    def units(*args, **kwargs):
        calls.append(args)
        yield None
    monkeypatch.setattr(train, "training_units", units)
    monkeypatch.setattr(cooperative_training, "run_cooperative_training",
                        lambda *a, **k: pytest.fail("unexpected manager"))
    train.run_training("config.json", unmanaged=True)
    assert calls == [("config.json",)]
    with pytest.raises(ValueError, match="cannot be combined"):
        train.run_training("config.json", unmanaged=True, manager_socket="custom.sock")


@pytest.mark.parametrize("extra", [[], ["--unmanaged"], ["--manager-socket", "custom.sock"]])
def test_cli_dispatch(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["train", "config.json", *extra])
    calls = []
    monkeypatch.setattr(train, "run_training", lambda *a, **k: calls.append(k))
    train.main()
    assert calls[0]["unmanaged"] == ("--unmanaged" in extra)
    assert calls[0]["manager_socket"] == ("custom.sock" if "--manager-socket" in extra else None)


@pytest.mark.parametrize("extra", [["--unmanaged", "--manager-socket", "x"],
                                   ["--unmanaged", "--prefetch"]])
def test_cli_rejects_conflicting_modes(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["train", "config.json", *extra])
    with pytest.raises(SystemExit):
        train.main()


def test_prefetch_does_not_launch_training_or_require_manager(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "config.json", "--prefetch"])
    monkeypatch.setattr(train, "_require_single_process_cli_mode", lambda _: None)
    calls = []
    monkeypatch.setattr(train, "run_prefetch", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(train, "run_training", lambda *a, **k: pytest.fail("training launch"))
    train.main()
    assert len(calls) == 1
