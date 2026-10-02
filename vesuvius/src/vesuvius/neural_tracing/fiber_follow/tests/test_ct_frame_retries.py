"""Invalid current CT frames reject whole plans inside the selected source."""
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.data import data as data_module
from vesuvius.neural_tracing.fiber_follow.tracing.heading import orient_item, SeedHeadingError
from vesuvius.neural_tracing.fiber_follow.data.datasets import WeightedDatasets


def dataset(monkeypatch, bad=(0, 2), lookahead=None, error=None):
    events = []
    monkeypatch.setattr(data_module, 'FiberVolume', lambda *a, **kw: SimpleNamespace(ct='ct'))
    monkeypatch.setattr(data_module, 'crop_local_grid', lambda crop: np.zeros((1, 3)))
    def normal(vol, pos):
        if int(pos[0]) in bad and pos[1] == 1:
            if error is not None:
                raise error('unrelated failure')
            raise SeedHeadingError('CT seed context crosses the volume boundary')
        return np.diag([1., 0., 0.])
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.ct_tensor', normal)
    class Builder:
        def __call__(self, items, vol):
            events.append(('build', int(items[0]['pos'][0])))
            for item in items:
                orient_item(item, vol)
            return dict(hist=torch.tensor(np.stack([i['pos'] for i in items]))[:, None])
        def prefetch_bounds(self, item, vol):
            return [(int(item['pos'][0]), int(item['pos'][1]))]
    class Client:
        def ensure_metadata(self, spec): pass
        def ensure(self, ct, bounds): events.append(('ensure', bounds[0][0]))
        def lookahead(self, ct, windows, scope): events.append(('lookahead', [w[0][0] for w in windows]))
    ds = data_module.FollowDataset([SimpleNamespace(length=100.)], SimpleNamespace(ct_zarr='test-source'),
        SimpleNamespace(crop=None), None, chunk=2, batch_builder=Builder())
    def plans(vol, windows):
        for index in count():
            # The valid first member gets oriented before the second can fail.
            yield [dict(pos=np.array([index, member, 0.]), frame=np.eye(3)) for member in range(2)]
    ds._iter_plans = plans
    if lookahead is not None:
        ds.remote_prefetch = Client()
        ds.remote_prefetch_lookahead = lookahead
    return ds, events


@pytest.mark.parametrize('lookahead', [None, 2])
def test_rejects_whole_pair_advances_prefetch_and_resets_counter(monkeypatch, lookahead):
    ds, events = dataset(monkeypatch, lookahead=lookahead)
    monkeypatch.setattr(data_module, 'MAX_CT_FRAME_REJECTIONS', 2)
    stream = iter(ds)
    batches = [next(stream) for _ in range(3)]
    assert [b['hist'][:, 0, 0].tolist() for b in batches] == [[1., 1.], [3., 3.], [4., 4.]]
    assert [int(b['ct_frame_rejected_batches'].sum()) for b in batches[:2]] == [1, 1]
    assert 'ct_frame_rejected_batches' not in batches[2]
    assert [n for name, n in events if name == 'build'] == list(range(5))
    if lookahead:
        assert [n for name, n in events if name == 'ensure'] == list(range(5))
        windows = [v for name, v in events if name == 'lookahead']
        assert windows[0] == [0] and windows[1] == [1, 2, 3]
        return
    # Ambiguous CT orientation falls back inside ct_frame: both members kept, nothing retried.
    ds, events = dataset(monkeypatch)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.ct_tensor',
                        lambda vol, pos: np.diag([0., 0., 1.]))
    stream = iter(ds)
    batches = [next(stream) for _ in range(3)]
    assert [b['hist'][:, 0, 0].tolist() for b in batches] == [[0., 0.], [1., 1.], [2., 2.]]
    assert all('ct_frame_rejected_batches' not in b for b in batches)
    assert [n for name, n in events if name == 'build'] == [0, 1, 2]


def test_persistent_bad_geometry_raises_contextual_bounded_error(monkeypatch):
    ds, events = dataset(monkeypatch, bad=range(100))
    monkeypatch.setattr(data_module, 'MAX_CT_FRAME_REJECTIONS', 3)
    with pytest.raises(SeedHeadingError, match='3 consecutive rejected plans.*test-source.*worker=0') as caught:
        next(iter(ds))
    assert isinstance(caught.value.__cause__, SeedHeadingError)
    assert [n for name, n in events if name == 'build'] == [0, 1, 2]


def test_non_heading_errors_are_not_swallowed(monkeypatch):
    ds, events = dataset(monkeypatch, error=OSError)
    with pytest.raises(OSError, match='unrelated failure'):
        next(iter(ds))
    assert [n for name, n in events if name == 'build'] == [0]


def test_rejections_do_not_change_mixed_source_selection(monkeypatch):
    ds, _ = dataset(monkeypatch)
    # Independent iterators retry inside their selected source before returning.
    mixed = WeightedDatasets([ds, ds], ['a', 'b'], [.3, .7], seed=17)
    stream = iter(mixed)
    actual = [int(next(stream)['dataset_id'][0]) for _ in range(12)]
    rng = np.random.default_rng(np.random.SeedSequence([17, 7349, 0]))
    assert actual == [int(rng.choice(2, p=[.3, .7])) for _ in range(12)]
