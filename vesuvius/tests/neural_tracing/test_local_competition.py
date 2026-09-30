from dataclasses import replace
import torch

from test_fiber_embeddings import pair_batch, training_batch, tiny_config
from vesuvius.neural_tracing.fiber_trace_3d.embedding import (
    EmbeddingConfig, embedding_statistics, local_competition, similarity_diagnostics,
)
from vesuvius.neural_tracing.fiber_trace_3d.train import (
    _evaluate_loaded_batch, _mean_dense_test_rows, _slice_batch,
)
from vesuvius.neural_tracing.fiber_trace_3d.model import build_fiber_trace_3d_model
from test_fiber_trace_3d import _loader_with_mapping
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import MacroBatchTargetSource


def test_macro_batch_prepass_uses_actual_objective_count_shape():
    batch = training_batch()
    batch = replace(batch, embedding_geometry=tuple(
        replace(g, objective='local_contrastive') for g in batch.embedding_geometry))
    loader = _loader_with_mapping({'augment_enabled': False, 'batch_size': 2})
    config = replace(loader.config, embedding=EmbeddingConfig(enabled=True, objective='local_contrastive'))

    class Source:
        released = 0

        def __next__(self):
            return batch

        def release_current(self):
            self.released += 1

    source = Source()
    prepass = MacroBatchTargetSource(source, config=config, device=torch.device('cpu'),
                                    output_stride=1, batch_sizes=(2, 2))
    expected = prepass._counts(batch) * 2
    assert expected.numel() == 6
    first, second = next(prepass), next(prepass)
    torch.testing.assert_close(first.normalization_counts, expected)
    torch.testing.assert_close(second.normalization_counts, expected)
    assert source.released == 2


def test_budget_and_hard_negative_pressure():
    config = EmbeddingConfig(objective='local_contrastive')
    assert config.pairs((128,)*3) == 65536
    assert config.pairs((256,)*3) == 131072
    assert config.pairs((129,)*3) % 32 == 0
    p = torch.tensor([.8])
    n = torch.tensor([[.1, .9]], requires_grad=True)
    value = local_competition(p, n, .1, .2)
    expected = torch.log1p(torch.exp((n-p[:, None]+.2)/.1).sum(1))
    torch.testing.assert_close(value, expected)
    value.sum().backward()
    assert n.grad[0, 1] > n.grad[0, 0]*100


def test_presence_detached_and_zero_budget():
    b = pair_batch()
    b.embedding_geometry = tuple(replace(g, objective='local_contrastive') for g in b.embedding_geometry)
    e = torch.randn(1, 4, 2, 2, 3, requires_grad=True)
    p = torch.rand(1, 1, 2, 2, 3, requires_grad=True)
    d = []
    result = embedding_statistics(e, p, b, d)
    assert result[-1] == 2048
    (result[-2]/result[-1]).backward()
    assert p.grad is None and torch.isfinite(e.grad).all()
    empty = []
    result = embedding_statistics(e, p*0, b, empty)
    assert result[-1] == 2048 and result[-2] == 0
    assert 'negative_mean' not in empty[0]


def test_test_metrics_independent_of_microbatch_and_weighted_across_rows():
    torch.manual_seed(7)
    config = tiny_config()
    config['embedding']['objective'] = 'local_contrastive'
    model = build_fiber_trace_3d_model(config).eval()
    b = training_batch()
    b = replace(b, embedding_geometry=tuple(replace(g, objective='local_contrastive') for g in b.embedding_geometry))
    loader = _loader_with_mapping({'augment_enabled':False, 'batch_size':2})
    options = dict(device=torch.device('cpu'), direction_weight=1., presence_weight=1., mixed_precision=None)
    a = _evaluate_loaded_batch(model, b, loader.config, microbatch_size=2, **options)
    c = _evaluate_loaded_batch(model, b, loader.config, microbatch_size=1, **options)
    for key in a:
        torch.testing.assert_close(a[key], c[key], atol=2e-5, rtol=2e-4)
    assert a['embedding_similarity_count/positive_mean'] == 2
    assert a['embedding_similarity_count/negative_mean'] == 1
    key = 'embedding_similarity/positive_mean'
    count = 'embedding_similarity_count/positive_mean'
    r = _mean_dense_test_rows([(0, {'total':1.,key:.2,count:1}), (1, {'total':2.,key:.8,count:3})])
    assert abs(r[key]-.65)<1e-6 and r[count]==4


def test_evaluation_partial_batch_pads_forwards_without_changing_statistics():
    torch.manual_seed(11)
    config = tiny_config()
    config['embedding']['enabled'] = False
    inner = build_fiber_trace_3d_model(config).eval()

    class CountingModel(torch.nn.Module):
        output_stride = inner.output_stride
        embedding_channels = 0

        def __init__(self):
            super().__init__()
            self.inner = inner
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return self.inner(value)

    model = CountingModel().eval()
    batch = _slice_batch(training_batch(), 0, 1)
    loader = _loader_with_mapping({'augment_enabled': False, 'batch_size': 2})
    options = dict(device=torch.device('cpu'), microbatch_size=1,
                   direction_weight=1., presence_weight=1., mixed_precision=None)
    baseline = _evaluate_loaded_batch(model, batch, loader.config, **options)
    assert model.calls == 1
    model.calls = 0
    padded = _evaluate_loaded_batch(
        model, batch, loader.config, minimum_forward_count=2, **options)
    assert model.calls == 2
    for key in baseline:
        torch.testing.assert_close(baseline[key], padded[key], rtol=0, atol=0)
