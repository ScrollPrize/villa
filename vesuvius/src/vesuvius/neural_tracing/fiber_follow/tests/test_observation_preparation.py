"""Sparse preparation preserves consumed inputs/targets and future sampling."""
import copy
import threading

import numpy as np
import pytest
import torch

from test_identity import config
from test_neighbor_bank import make_bank, add_shard, publish
from sampling_fixtures import clean_sample
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.train import DecisionBatchPrefetch
from vesuvius.neural_tracing.fiber_follow.shared.data import make_sample
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec


@pytest.mark.parametrize('selected', [[False]*4, [False, True, False, True], [True]*4])
def test_sparse_targets_preserve_images_labels_and_sampling_feedback(tmp_path, monkeypatch, selected):
    from test_history_slabs import fake_ct
    fake_ct(monkeypatch)
    bank, fiber = make_bank(tmp_path)
    publish(tmp_path, [add_shard(tmp_path, 0, x=4., z_range=(20., 180.))])
    cfg = config(direction_inputs=True, fine=CropSpec(depth=40, width=25, behind=16, spacing=1.))
    sampling = IdentitySampling()
    full = IdentityObservationBuilder(cfg, [fiber], negative_bank=bank, augment=True, sampling=sampling)
    sparse = IdentityObservationBuilder(cfg, [fiber], negative_bank=bank, augment=True, sampling=sampling)
    rng = np.random.default_rng(782)
    items = []
    for j, source in enumerate((0, 2, 0, 3)):
        row = make_sample(fiber, 80.+j, False, clean_sample(cfg), rng)
        row.update(source=source, source_step=-1, fiber_ref=(0, 80.+j, False), memory_warm=True)
        full.prepare(row, fiber, rng)
        row.update(photometric=(1.1, .03, .02), blur_sigma=.8, drop_presence=j == 2)
        items.append(row)
    def images(rows, vol, crop, pool=None, **kwargs):
        image = torch.linspace(0., 1., 8*crop.depth*crop.width**2).reshape(8, crop.depth, crop.width, crop.width)
        return image[None].repeat(len(rows), 1, 1, 1, 1)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.data.image_crop', images)
    reference = full(copy.deepcopy(items), None)
    calls = []
    original = bank.candidates
    def tracked(item, *args, **kwargs):
        calls.append((item['source'], kwargs['rasterize']))
        return original(item, *args, **kwargs)
    monkeypatch.setattr(bank, 'candidates', tracked)
    got = sparse(copy.deepcopy(items), None, decision_mask=selected)
    for key in reference['x']:
        if key == 'history_load_seconds':
            continue
        torch.testing.assert_close(got['x'][key], reference['x'][key], rtol=0, atol=0)
    for key in ('hist', 'hmask'):
        torch.testing.assert_close(got[key], reference[key], rtol=0, atol=0)
    assert list(full.lateral) == list(sparse.lateral) and len(sparse.lateral) == 2
    assert calls == [(row['source'], selected[j]) for j, row in enumerate(items)
                     if selected[j] or row['source'] == 0]
    if any(selected):
        for key, value in reference.items():
            if key != 'x':
                torch.testing.assert_close(got[key][selected], value[selected], rtol=0, atol=0)
    else:
        assert not {'foreign', 'dense_ab', 'dense_mask', 'fut'} & got.keys()




def test_prefetch_preserves_boundaries_and_prepares_next_update_concurrently():
    ready = threading.Event()
    chunks = [dict(hist=torch.zeros(12,1,3)) for _ in range(4)]
    def loader():
        for i, chunk in enumerate(chunks):
            if i == 3:
                ready.set()
            yield chunk
    prefetch = DecisionBatchPrefetch(iter(loader()), 2)
    try:
        first = next(prefetch)
        assert all(a is b for a, b in zip(first, chunks[:2])) and len(first) == 2
        assert sum(len(chunk['hist']) for chunk in first) == 24
        assert ready.wait(5), 'Next update must be prepared before requesting it'
        second = next(prefetch)
        assert all(a is b for a, b in zip(second, chunks[2:])) and len(second) == 2
        with pytest.raises(StopIteration):
            next(prefetch)
    finally:
        prefetch.close()
    assert all(not thread.is_alive() for thread in prefetch.executor._threads)


def test_prefetch_propagates_loader_failure_and_closes():
    def loader():
        raise ValueError('invalid sample')
        yield
    prefetch = DecisionBatchPrefetch(iter(loader()), 1)
    try:
        with pytest.raises(ValueError, match='invalid sample'):
            next(prefetch)
    finally:
        prefetch.close()
    with pytest.raises(StopIteration):
        next(prefetch)




