"""Sparse preparation preserves consumed inputs/targets and future sampling."""
import copy
import threading

import numpy as np
import pytest
import torch

from test_identity import config
from test_neighbor_bank import make_bank, add_shard, publish
from test_neighbor_following import clean_sample
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.feature_sequences import sequence_batches
from vesuvius.neural_tracing.fiber_follow.regression.supervision import candidate_targets
from vesuvius.neural_tracing.fiber_follow.regression.train import DecisionBatchPrefetch, pack_feature_chunks
from vesuvius.neural_tracing.fiber_follow.shared.data import make_sample
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec


@pytest.mark.parametrize('selected', [[False]*4, [False, True, False, True], [True]*4])
def test_sparse_targets_preserve_images_labels_and_sampling_feedback(tmp_path, monkeypatch, selected):
    bank, fiber = make_bank(tmp_path)
    publish(tmp_path, [add_shard(tmp_path, 0, x=4., z_range=(20., 180.))])
    cfg = config(direction_inputs=True, fine=CropSpec(depth=40, width=25, behind=16, spacing=1.))
    sampling = IdentitySampling(decision_fraction=.5)
    full = IdentityObservationBuilder(cfg, [fiber], negative_bank=bank, augment=True, sampling=sampling)
    sparse = IdentityObservationBuilder(cfg, [fiber], negative_bank=bank, augment=True, sampling=sampling)
    rng = np.random.default_rng(782)
    items = []
    for j, source in enumerate((0, 2, 0, 5)):
        row = make_sample(fiber, 80.+j, False, clean_sample(cfg), rng)
        row.update(source=source, source_step=-1, stratum=-1, fiber_ref=(0, 80.+j, False), memory_warm=True)
        full.prepare(row, fiber, rng)
        row.update(photometric=(1.1, .03, .02), blur_sigma=.8, drop_presence=j == 2)
        if j < 2:
            row['candidate_points'] = np.repeat(row['fut_local'][None], 4, axis=0)
            row['candidate_mask'] = np.ones((4, cfg.n_future), np.float32)
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
        labels, known = candidate_targets(reference, cfg, sampling.candidate_tolerance)
        torch.testing.assert_close(got['candidate_mask'][selected], known[selected], rtol=0, atol=0)
        valid = known[selected].bool()
        torch.testing.assert_close(got['candidate_labels'][selected][valid], labels[selected][valid], rtol=0, atol=0)
    else:
        assert not {'foreign', 'dense_ab', 'dense_mask', 'candidate_points', 'candidate_labels', 'fut'} & got.keys()


def test_decision_plan_reaches_builder_before_target_construction(monkeypatch):
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.feature_sequences.stream_rows',
                        lambda item, builder, band: item)
    class Builder:
        cfg = config()
        def __call__(self, items, vol, *, decision_mask):
            return dict(hist=torch.zeros(len(items), 1, 3), prepared_decisions=decision_mask.clone())
    streams = [[dict(identity_seed=j, source=0) for _ in range(n)] for j, n in enumerate((9, 12))]
    rows = [b for c in sequence_batches(Builder(), streams, None) for b in c['feature_sequence']]
    assert sum(int(b['decision_mask'].sum()) for b in rows) == 6
    for row in rows:
        torch.testing.assert_close(row['prepared_decisions'], row['decision_mask'])


def test_prefetch_preserves_boundaries_and_prepares_next_update_concurrently():
    ready = threading.Event()
    chunks = [dict(feature_sequence=[dict(stream_id=torch.tensor([i]),
        decision_mask=torch.tensor([flag]), hist=torch.zeros(1, 1, 3))])
        for i, flag in enumerate((False, True, True, False, True, True))]
    def loader():
        for i, chunk in enumerate(chunks):
            if i == 5:
                ready.set()
            yield chunk
    prefetch = DecisionBatchPrefetch(iter(loader()), 2)
    try:
        first = next(prefetch)
        assert all(a is b for a, b in zip(first, chunks[:3])) and len(first) == 3
        assert ready.wait(5), 'Next update must be prepared before requesting it'
        second = next(prefetch)
        assert all(a is b for a, b in zip(second, chunks[3:])) and len(second) == 3
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


def test_packing_does_not_discard_later_decision_targets():
    def row(key, decision):
        result = dict(stream_id=torch.tensor([key]), hist=torch.zeros(1, 2, 3))
        if decision:
            result['foreign'] = torch.ones(1, 4, 4, 4)
        return result
    chunks = [dict(feature_sequence=[row(1, False), row(1, False)]),
              dict(feature_sequence=[row(2, False), row(2, True)])]
    assert pack_feature_chunks(chunks, 2) is chunks


def test_omitted_observation_targets_and_prefetch_preserve_update():
    from test_sparse_memory_training import stream
    from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
    from vesuvius.neural_tracing.fiber_follow.regression.train import optimizer_update, prepare_training
    torch.manual_seed(718)
    cfg = config()
    full = stream(cfg, length=7, batch=2)
    sparse = copy.deepcopy(full)
    required = {'x', 'hist', 'hmask', 'stream_id', 'stream_reset', 'stream_end',
                'stream_index', 'decision_mask', 'loss_weight', 'encoder_indices', 'retain_until'}
    for chunk in sparse:
        for row in chunk['feature_sequence']:
            if not row['decision_mask'].any():
                for key in set(row)-required:
                    del row[key]
    a = build_model(cfg)
    b = copy.deepcopy(a)
    prepare_training(a, backend='eager')
    prepare_training(b, backend='eager')
    updates = DecisionBatchPrefetch(iter(sparse), 6)
    metrics = []
    try:
        for model, chunks in ((a, full), (b, next(updates))):
            metrics.append(optimizer_update(model, copy.deepcopy(model),
                torch.optim.AdamW(model.parameters()), chunks, 1, .001, compute_metrics=False))
    finally:
        updates.close()
    for key in ('loss', 'grad_norm', 'observed_states', 'supervised_states', 'endpoint_states'):
        assert metrics[0][key] == metrics[1][key]
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0, msg=name)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
        else:
            assert q.grad is None
