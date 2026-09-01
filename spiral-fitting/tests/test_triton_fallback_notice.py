"""The fused-kernel fallback says so once, and only when triton is missing."""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize(
    'module_name, gate_name',
    [('flow_triton', 'rk4_triton_available'),
     ('gap_triton', 'gap_triton_available')],
)
def test_missing_triton_warns_once(module_name, gate_name, monkeypatch, capsys):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, '_HAS_TRITON', False)
    monkeypatch.setattr(module, '_warned_missing_triton', False)
    monkeypatch.delenv('FIT_SPIRAL_TRITON', raising=False)
    gate = getattr(module, gate_name)

    assert gate() is False
    assert gate() is False

    out = capsys.readouterr().out
    assert out.count('triton is unavailable') == 1
    assert 'slower' in out


@pytest.mark.parametrize(
    'module_name, gate_name',
    [('flow_triton', 'rk4_triton_available'),
     ('gap_triton', 'gap_triton_available')],
)
def test_opting_out_is_silent(module_name, gate_name, monkeypatch, capsys):
    # Turning the kernels off by hand is a choice, not a surprise; say nothing.
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, '_HAS_TRITON', True)
    monkeypatch.setattr(module, '_warned_missing_triton', False)
    monkeypatch.setenv('FIT_SPIRAL_TRITON', '0')

    assert getattr(module, gate_name)() is False
    assert capsys.readouterr().out == ''
