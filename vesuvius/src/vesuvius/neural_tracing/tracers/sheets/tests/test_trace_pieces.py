"""Tracing a long seed row in pieces (``tracing/inference.py``) with a stand-in ``trace`` (no model or data files)."""
import numpy as np
import pytest

from vesuvius.neural_tracing.tracers.sheets.tracing import inference


class _Cfg:
    class trace:
        spline_samples = 6


def _fake(lengths):
    """Stand-in for ``trace``: piece k traces lengths[k] steps; x = front column, y = piece, z = step."""
    calls = []

    def trace(model, cfg, seed, direction, steps, device, **kw):
        k = len(calls)
        calls.append((float(seed[0, 0]), len(seed), kw))
        width = (len(seed)-1)*6+1
        rows = np.zeros((lengths[k]+1, width, 3), np.float32)
        rows[..., 0] = seed[0, 0]*6+np.arange(width)
        rows[..., 1] = k
        rows[..., 2] = np.arange(lengths[k]+1)[:, None]
        return rows, np.full((lengths[k]+1, width), .5+.1*k, np.float32)
    trace.calls = calls
    return trace


def _seed(n):
    s = np.zeros((n, 3), np.float32)
    s[:, 0] = np.arange(n)
    return s


def test_piece_starts():
    assert inference.piece_starts(15, 7, 2) == [0, 2, 4, 6, 8]
    assert inference.piece_starts(15, 7, 4) == [0, 4, 8]
    assert inference.piece_starts(16, 7, 4) == [0, 4, 8, 9]
    assert inference.piece_starts(7, 7, 2) == [0]
    for stride in (0, -1, 7):
        with pytest.raises(ValueError):
            inference.piece_starts(15, 7, stride)


def test_short_row_is_traced_whole(monkeypatch):
    fake = _fake([60])
    monkeypatch.setattr(inference, 'trace', fake)
    rows, confs = inference.trace_in_pieces(None, _Cfg, _seed(7), _seed(7), 60, 'cpu', stop_on_disconnect=False)
    assert rows.shape == (61, 37, 3) and fake.calls == [(0., 7, dict(stop_on_disconnect=False))]


def test_long_row_merges_central_live_pieces(monkeypatch):
    fake = _fake([60, 10, 60])
    monkeypatch.setattr(inference, 'trace', fake)
    rows, confs = inference.trace_in_pieces(None, _Cfg, _seed(15), _seed(15), 60, 'cpu', piece_stride=4)
    assert rows.shape == (61, 85, 3) and [c[1] for c in fake.calls] == [7, 7, 7]
    assert np.array_equal(rows[0, :, 0], np.arange(85))
    assert rows[5, 33, 1] == 1 and rows[20, 33, 1] == 0 and np.isclose(confs[20, 33], .5)
    assert rows[20, 42, 1] == 1 and rows[20, 42, 2] == 10 and confs[20, 42] == 0


def test_merge_needs_full_cover():
    with pytest.raises(ValueError):
        inference.merge_pieces([(np.ones((2, 37, 3)), np.ones((2, 37)))], [0], 85)
