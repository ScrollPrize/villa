"""find_inconsistent_windings and flatten_spiral_checkpoint take their torch
device from devices.fit_device, so they also run where CUDA is missing."""
import pytest
import torch

import find_inconsistent_windings as fiw
import flatten_spiral_checkpoint as fsc


class _Picked(Exception):
    pass


def _fake_fit_device(calls):
    def fit_device(requested=None):
        calls.append(requested)
        raise _Picked
    return fit_device


def test_build_transform_asks_fit_device(monkeypatch):
    calls = []
    monkeypatch.setattr(fiw, 'fit_device', _fake_fit_device(calls))
    with pytest.raises(_Picked):
        fiw.build_transform(None, None, None, 0, 1)
    assert calls == [None]


def test_find_inconsistent_windings_main_asks_fit_device(monkeypatch):
    calls = []
    monkeypatch.setattr(fiw, 'fit_device', _fake_fit_device(calls))
    try:
        with pytest.raises(_Picked):
            fiw.main.callback(**{p.name: None for p in fiw.main.params})
    finally:
        torch.set_grad_enabled(True)
    assert calls == [None]


def test_flatten_device_defaults_to_an_available_backend(monkeypatch, tmp_path):
    (tmp_path / 'fit.pt').touch()
    argv = [str(tmp_path / 'fit.pt'), str(tmp_path / 'out')]
    assert fsc._parse_args(argv).device is None
    calls = []
    monkeypatch.setattr(fsc, 'fit_device', _fake_fit_device(calls))
    with pytest.raises(_Picked):
        fsc.main(argv)
    assert calls == [None]


def test_flatten_passes_an_explicit_device_through(monkeypatch, tmp_path):
    (tmp_path / 'fit.pt').touch()
    calls = []
    monkeypatch.setattr(fsc, 'fit_device', _fake_fit_device(calls))
    with pytest.raises(_Picked):
        fsc.main([str(tmp_path / 'fit.pt'), str(tmp_path / 'out'), '--device', 'cpu'])
    assert calls == ['cpu']
