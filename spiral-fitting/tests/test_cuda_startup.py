from types import SimpleNamespace
from unittest import mock
import pytest
import torch
import fit_spiral


def test_startup_probe_synchronizes_without_consuming_rng(monkeypatch):
    calls = []
    rng = torch.get_rng_state().clone()
    monkeypatch.setattr(torch, 'empty', lambda *a, **kw: calls.append('allocate'))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: calls.append('synchronize'))
    fit_spiral.FitContext.check_cuda_ready(SimpleNamespace(progress=None))
    assert calls == ['allocate', 'synchronize']
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('failure_site', ['allocation', 'synchronization'])
def test_gb10_oom_reports_host_memory_and_preserves_cause(monkeypatch, failure_site):
    # Context creation raises AcceleratorError/RuntimeError, not necessarily
    # torch.OutOfMemoryError. Exercise the actual observed message as well.
    failure = RuntimeError('CUDA error: out of memory')
    monkeypatch.setattr(fit_spiral.sys, 'platform', 'linux')
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'NVIDIA GB10')
    monkeypatch.setattr(torch, 'empty', mock.Mock(
        side_effect=failure if failure_site == 'allocation' else None))
    monkeypatch.setattr(torch.cuda, 'synchronize', mock.Mock(
        side_effect=failure if failure_site == 'synchronization' else None))
    with mock.patch('builtins.open', mock.mock_open(
            read_data='MemFree: 9000 kB\nMemAvailable: 97000 kB\n')):
        with pytest.raises(RuntimeError, match='before loading fit inputs') as caught:
            fit_spiral.FitContext.check_cuda_ready(SimpleNamespace(progress=None))
    assert caught.value.__cause__ is failure
    assert 'drop_caches' in str(caught.value)
    assert 'MemAvailable: 97000 kB' in str(caught.value)


def test_headless_stops_before_loading_inputs_on_cuda_failure():
    context = SimpleNamespace(
        progress=None,
        check_cuda_ready=mock.Mock(side_effect=RuntimeError('CUDA unavailable')),
        load_host_inputs=mock.Mock())
    with pytest.raises(RuntimeError, match='CUDA unavailable'):
        fit_spiral.FitContext.run(context)
    context.load_host_inputs.assert_not_called()
