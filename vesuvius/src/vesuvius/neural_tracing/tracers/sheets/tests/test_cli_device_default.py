"""The sheet tracer's evaluation and script CLIs pick cuda, else mps, else cpu when --device is not given
(they used to default to cuda, which fails at checkpoint load on a Mac or a CPU-only machine)."""
import importlib
from types import SimpleNamespace

import pytest

PKG = 'vesuvius.neural_tracing.tracers.sheets.'


class _Loaded(Exception):
    pass


def _no_cuda_but_mps(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)


def _stop_at_load(monkeypatch, mod, seen):
    def load(path, device, **kwargs):
        seen['device'] = device
        raise _Loaded
    monkeypatch.setattr(mod, 'load_checkpoint', load)


@pytest.mark.parametrize('name, argv', [
    ('evaluation.hard_spots', ['--checkpoint', 'x.pt', '--out', 'unused']),
    ('evaluation.rollouts', ['--checkpoint', 'x.pt']),
    ('evaluation.benchmark', ['--checkpoints', 'x.pt']),
    ('scripts.trace_from_fibers', ['--fibers', 'unused', '--checkpoint', 'x.pt', '--out', 'unused']),
])
def test_checkpoint_clis_default_to_mps_without_cuda(monkeypatch, name, argv):
    mod = importlib.import_module(PKG + name)
    _no_cuda_but_mps(monkeypatch)
    seen = {}
    _stop_at_load(monkeypatch, mod, seen)
    if hasattr(mod, 'resolve_checkpoint'):
        monkeypatch.setattr(mod, 'resolve_checkpoint', lambda value: value)
    with pytest.raises(_Loaded):
        mod.main(argv)
    assert seen['device'] == 'mps'


def test_explicit_device_still_wins(monkeypatch):
    mod = importlib.import_module(PKG + 'evaluation.rollouts')
    _no_cuda_but_mps(monkeypatch)
    seen = {}
    _stop_at_load(monkeypatch, mod, seen)
    with pytest.raises(_Loaded):
        mod.main(['--checkpoint', 'x.pt', '--device', 'cpu'])
    assert seen['device'] == 'cpu'


def test_bench_train_step_defaults_to_mps_without_cuda(monkeypatch):
    mod = importlib.import_module(PKG + 'scripts.bench_train_step')
    _no_cuda_but_mps(monkeypatch)
    seen = {}

    def device(name):
        seen['device'] = name
        raise _Loaded
    monkeypatch.setattr(mod, 'C', SimpleNamespace(load=lambda *a: SimpleNamespace(train=None, data=None)))
    monkeypatch.setattr(mod, 'torch', SimpleNamespace(device=device))
    with pytest.raises(_Loaded):
        mod.main(['--config', 'unused.yaml'])
    assert seen['device'] == 'mps'


def test_architecture_figures_default_to_mps_without_cuda(monkeypatch, tmp_path):
    mod = importlib.import_module(PKG + 'docs.make_architecture_figures')
    _no_cuda_but_mps(monkeypatch)
    seen = {}
    monkeypatch.setattr(mod, 'OUT', tmp_path)
    for figure in ('dataflow', 'layout_top', 'layout_section', 'tilt'):
        monkeypatch.setattr(mod, figure, lambda: None)

    def real_data(checkpoint, path_map, device):
        seen['device'] = device
        raise _Loaded
    monkeypatch.setattr(mod, 'real_data', real_data)
    with pytest.raises(_Loaded):
        mod.main(['--checkpoint', 'x.pt'])
    assert seen['device'] == 'mps'
