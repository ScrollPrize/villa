"""Complete output-plane evidence alongside global image and historical context."""
import math

import pytest
import torch

from test_trajectory_memory import cfg, memory_batch
from vesuvius.neural_tracing.fiber_follow.regression.model import (
    DirectConfig, OutputPlaneFeatures, build_model,
)
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec


@pytest.mark.parametrize('width, step', [(17, 1.), (16, .75)])
def test_every_lateral_pixel_has_correct_coordinates_and_depth_interpolation(width, step):
    c = cfg(fine=CropSpec(depth=16, width=width, behind=3, spacing=.5), future_step=step)
    features = OutputPlaneFeatures(c)
    z, y, x = torch.meshgrid(torch.arange(c.fine.depth), torch.arange(width),
                            torch.arange(width), indexing='ij')
    # High-frequency lateral structure makes pooling or axis swaps observable.
    checker = ((x+y) % 2).float()
    volume = torch.stack((x.float(), y.float(), z.float(), checker), 0)[None].requires_grad_()
    sampled = features.sample(volume)
    expected = []
    positions = []
    expected_grad = torch.zeros_like(volume)
    for k in range(1, c.n_future+1):
        depth = c.fine.behind+k*step/c.fine.spacing
        lower, fraction = math.floor(depth), depth % 1
        expected.append(torch.stack((x[0].float(), y[0].float(),
                                    torch.full((width, width), depth), checker[0]), -1).reshape(-1, 4))
        positions.append(torch.stack(((x[0]-(width-1)/2)*c.fine.spacing,
                                      (y[0]-(width-1)/2)*c.fine.spacing,
                                      torch.full((width, width), k*step)), -1).reshape(-1, 3))
        expected_grad[:, :, lower] += 1-fraction
        expected_grad[:, :, lower+1] += fraction
    torch.testing.assert_close(sampled[0], torch.cat(expected), rtol=0, atol=0)
    torch.testing.assert_close(features.xyz, torch.cat(positions), rtol=0, atol=0)
    sampled.sum().backward()
    torch.testing.assert_close(volume.grad, expected_grad, rtol=0, atol=0)


def test_production_planes_cover_all_16_full_resolution_cross_sections():
    c = DirectConfig()
    features = OutputPlaneFeatures(c)
    assert features.xyz.shape == (163216, 3)
    assert math.prod(c.token_shape) == 39015
    torch.testing.assert_close(features.xyz[0], torch.tensor([-25., -25., 1.]))
    torch.testing.assert_close(features.xyz[-1], torch.tensor([25., 25., 16.]))
    assert features.lower.tolist() == list(range(50, 81, 2))


@pytest.mark.parametrize('refinements', [0, 1])
def test_all_generator_layers_read_fine_planes_and_keep_deep_and_history(monkeypatch, refinements):
    torch.manual_seed(81)
    c = cfg(decoder_layers=2, recurrent_refinement_steps=refinements)
    model = build_model(c).eval()
    batch = memory_batch(c, 1)
    captured = {}
    original_memory = model.decoder_memory

    def capture_memory(ctx):
        memory, padding = original_memory(ctx)
        captured.update(memory=memory, padding=padding, ctx=ctx)
        return memory, padding

    monkeypatch.setattr(model, 'decoder_memory', capture_memory)
    reads = []
    for index, layer in enumerate(model.decoder.layers):
        if refinements:
            original = layer.project_memory

            def read(memory, _original=original, _index=index):
                reads.append((_index, memory))
                return _original(memory)

            monkeypatch.setattr(layer, 'project_memory', read)
        else:
            original = layer._mha_block

            def read(query, memory, *args, _original=original, _index=index, **kwargs):
                reads.append((_index, memory))
                return _original(query, memory, *args, **kwargs)

            monkeypatch.setattr(layer, '_mha_block', read)
    output = model(batch['x'], batch['hist'], batch['hmask'])
    ctx, memory, padding = (captured[k] for k in ('ctx', 'memory', 'padding'))
    spatial_count = math.prod(c.token_shape)
    fine_count = c.n_future*c.fine.width**2
    assert memory.shape[1] == ctx['confidence_padding'].shape[1]
    torch.testing.assert_close(memory[:, :spatial_count], ctx['deep'].flatten(2).transpose(1, 2))
    torch.testing.assert_close(padding, ctx['confidence_padding'])
    assert not padding[:, -fine_count:].any()
    assert [index for index, _ in reads] == list(range(c.decoder_layers))
    assert all(value is memory for _, value in reads)
    # Even the FIRST point can read all lateral pixels on the LAST output plane.
    gradient, = torch.autograd.grad(output['initial_points'][0, 0, 0], memory, retain_graph=True)
    assert (gradient[:, -fine_count:].abs().sum(-1) > 0).all()
    assert (gradient[:, :spatial_count].abs().sum(-1) > 0).all()
    output['points'].square().mean().backward()
    for param in model.output_plane_features.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0


def test_generator_plane_projections_do_not_enter_candidate_scoring():
    torch.manual_seed(82)
    model = build_model(cfg(recurrent_refinement_steps=1)).eval()
    batch = memory_batch(model.cfg, 1)
    curves = torch.zeros(1, 2, model.cfg.n_future, 3)
    curves[..., 2] = model.planes
    curves[:, 1, :, 0] = 2.
    args = batch['x'], batch['hist'], batch['hmask']
    first = model(*args, candidates=curves)
    first['candidate_hazard_logits'].sum().backward()
    assert all(p.grad is None for p in model.output_plane_features.parameters())
    assert model.encoder.dense_decoder[-1].weight.grad.abs().sum() > 0
    for module in (model.confidence_scorer.plane_projection, model.confidence_scorer.plane_position):
        for param in module.parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0
    with torch.no_grad():
        model.output_plane_features.projection.weight.normal_(std=2.)
    second = model(*args, candidates=curves)
    assert (first['points']-second['points']).abs().max() > 1e-5
    torch.testing.assert_close(first['candidate_hazard_logits'], second['candidate_hazard_logits'], rtol=0, atol=0)
    torch.testing.assert_close(first['memory_cache'], second['memory_cache'], rtol=0, atol=0)


@pytest.mark.parametrize('scored', ['hazard_logits', 'candidate_hazard_logits'])
def test_all_scorer_layers_read_every_fine_pixel_and_reuse_projections(monkeypatch, scored):
    torch.manual_seed(84)
    model = build_model(cfg(recurrent_refinement_steps=1)).eval()
    batch = memory_batch(model.cfg, 1)
    samples, reads = [], []
    original_sample = model.output_plane_features.sample

    def sample(fine):
        value = original_sample(fine)
        samples.append(value)
        return value

    monkeypatch.setattr(model.output_plane_features, 'sample', sample)
    for index, layer in enumerate(model.confidence_scorer.layers):
        original = layer.project_memory

        def read(memory, _original=original, _index=index):
            reads.append((_index, memory))
            return _original(memory)

        monkeypatch.setattr(layer, 'project_memory', read)
    curves = torch.zeros(1, 2, model.cfg.n_future, 3, requires_grad=True)
    with torch.no_grad():
        curves[..., 2] = model.planes
        curves[:, 1, :, 0] = 2.
    output = model(batch['x'], batch['hist'], batch['hmask'], candidates=curves)
    # One shared plane gather, and one K/V projection per scorer layer for all paths.
    assert len(samples) == 1
    assert [index for index, _ in reads] == list(range(len(model.confidence_scorer.layers)))
    memory = reads[0][1]
    assert all(value is memory for _, value in reads)
    fine_count = model.cfg.n_future*model.cfg.fine.width**2
    expected = model.confidence_scorer.plane_tokens(samples[0], model.output_plane_features.xyz)
    torch.testing.assert_close(memory[:, -fine_count:], expected)
    # The first segment reaches every pixel, including distant/later planes,
    # through both projected tokens and the shared, unprojected fine samples.
    score = output[scored][..., 0].sum()
    memory_gradient, sample_gradient = torch.autograd.grad(score, (memory, samples[0]), retain_graph=True)
    assert (memory_gradient[:, -fine_count:].abs().sum(-1) > 0).all()
    assert (sample_gradient.abs().sum(-1) > 0).all()
    score.backward()
    assert curves.grad is None
    for name, param in model.named_parameters():
        if name.startswith(('coordinates.', 'decoder.', 'query.', 'refinement_', 'output_plane_features.')):
            assert param.grad is None, name


def test_scorer_plane_projections_change_scores_without_changing_paths():
    torch.manual_seed(85)
    model = build_model(cfg(recurrent_refinement_steps=1)).eval()
    batch = memory_batch(model.cfg, 1)
    args = batch['x'], batch['hist'], batch['hmask']
    with torch.no_grad():
        first = model(*args)
        model.confidence_scorer.plane_projection.weight.normal_(std=2.)
        second = model(*args)
    assert (first['hazard_logits']-second['hazard_logits']).abs().max() > 1e-5
    torch.testing.assert_close(first['points'], second['points'], rtol=0, atol=0)
    torch.testing.assert_close(first['memory_cache'], second['memory_cache'], rtol=0, atol=0)


def test_fractional_plane_sampling_compiles_with_gradients():
    torch.manual_seed(83)
    c = cfg(fine=CropSpec(depth=16, width=12, behind=3, spacing=.5), future_step=.75)
    module = OutputPlaneFeatures(c)
    volume = torch.randn(1, c.channels, c.fine.depth, c.fine.width, c.fine.width, requires_grad=True)
    eager = module(volume)
    compiled = torch.compile(module, backend='eager', fullgraph=True)(volume)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
    compiled.square().mean().backward()
    assert torch.isfinite(volume.grad).all() and volume.grad.abs().sum() > 0
