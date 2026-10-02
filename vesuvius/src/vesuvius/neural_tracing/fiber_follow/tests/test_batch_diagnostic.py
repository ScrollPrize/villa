"""Diagnostics observe the deployed policy without changing training state."""
import copy
import json

import numpy as np
from PIL import Image
import pytest
import torch

from test_regression import batch, config, proposal_output
from label_fixtures import set_terminal, state_labels
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectFollower
from vesuvius.neural_tracing.fiber_follow.regression.batch_diagnostic import (
    layer_capture, decision_details, render_microbatch,
)


@pytest.mark.parametrize('empty', [False, True])
@pytest.mark.parametrize('variant', ['fine', 'legacy'])
def test_layer_capture_preserves_predictions_and_cleans_up(empty, variant):
    cfg = config()
    cfg.history_encoder = variant
    cfg.encoder, cfg.token_only, cfg.stem_channels = 'patch4', True, 4
    cfg.recurrent_refinement_steps = 1
    model = DirectFollower(cfg).eval()
    data = batch(cfg, 1)
    if empty:
        data['x']['history_valid'][:] = False
    weights = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    with torch.no_grad():
        expected = model(data['x'], data['hist'], data['hmask'], confidence_threshold=1.)
        with layer_capture(model) as layers:
            actual = model(data['x'], data['hist'], data['hmask'], confidence_threshold=1.)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for key, value in weights.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert layers['statistics']['CT stem']['rms'] > 0
    assert layers['statistics']['patch embedding']['rms'] > 0
    assert set(layers['decoder']) == {'input', 'block 1', 'output'}
    assert set(layers['scorer']) == {'input', 'block 1', 'block 2'}
    assert layers['history_depths'] == {'generator': 1, 'scorer': 2}
    assert layers['encoder']['encoder output'].ndim == 2
    lateral = 17 if variant == 'fine' else 9
    assert layers['history']['tokens'].shape == (8, lateral, lateral)
    assert layers['history']['convolution'].shape == (0 if empty else 2, lateral, lateral)
    for head in ('generator', 'scorer'):
        for attention in layers['history'][head+'_attention']:
            assert torch.isfinite(attention).all()
            torch.testing.assert_close(attention.sum(-1), torch.full((cfg.n_future,), 0. if empty else 1.))
            assert not attention[:, 2:].any()
    # Distinct layer/attempt values catch averaging the wrong retry or layer count.
    from vesuvius.neural_tracing.fiber_follow.regression.diagnostic_plots import display_example
    assert len(layers['history']['scorer_attention']) == 4
    for head, depth in layers['history_depths'].items():
        layers['history'][head+'_attention'] = [torch.full((cfg.n_future, 8), 100.*attempt+layer)
            for attempt in range(2) for layer in range(depth)]
    selected = dict(actual, selected_refinement=torch.ones(1, dtype=torch.long))
    example = display_example(data, selected, {}, layers, cfg, 'test', {})
    np.testing.assert_allclose(example['attention']['scorer'], 100.5)
    np.testing.assert_allclose(example['attention']['generator'], 100.)
    assert 'forward_cached' not in model.history_attention.__dict__
    assert not model.encoder.patch_projection._forward_hooks
    with pytest.raises(RuntimeError):
        with layer_capture(model):
            raise RuntimeError('inference failed')
    assert 'forward_cached' not in model.confidence_scorer.history_attention.__dict__
    assert not model.encoder.blocks[0]._forward_pre_hooks


def test_decision_details_keep_wrong_unknown_and_skipped_distinct():
    cfg = config()
    data = batch(cfg, 1)
    data['dense_ab'][:] = 0
    curves = torch.zeros(1, 2, cfg.n_future, 3)
    curves[..., 2] = torch.arange(1, cfg.n_future+1)
    curves[:, 1, :, 0] = 2
    output = proposal_output(curves, torch.full((1, 2, cfg.n_future), -10.))
    details = decision_details(output, data, cfg, 4, 1.5)
    assert details['commit'] == 4
    assert details['known'].all() and not details['labels'].any()
    assert details['attempts'][0]['error'] == 0
    assert details['attempts'][1]['error'] == 2
    data['dense_mask'][:] = 0
    details = decision_details(output, data, cfg, 4, 1.5)
    assert not details['known'].any()
    assert details['attempts'][1]['error'] is None
    for row in range(len(data['terminal'])):
        set_terminal(data, row)
    details = decision_details(output, data, cfg, 4, 1.5)
    assert details['known'].all() and not details['labels'].any()
    data['identity_observable'] = torch.zeros(1)
    assert not decision_details(output, data, cfg, 4, 1.5)['known'].any()


def test_render_emits_readable_images_strict_json_and_preserves_rng(tmp_path):
    cfg = config()
    cfg.recurrent_refinement_steps = 1
    model = DirectFollower(cfg).train()
    data = batch(cfg, 2)
    data['dense_mask'][1] = 0
    data['dense_ab'][1] = float('nan')
    data['x']['history_valid'][1] = False
    rng, numpy_rng = torch.get_rng_state(), np.random.get_state()
    data['dataset_id'] = torch.tensor([0, 1])
    rows = render_microbatch(model, data, tmp_path, 1000, device='cpu', n_commit=4, tolerance=1.5,
                            dataset_names=['ordinary', 'unknown'], training_metrics={'loss': .2})
    assert model.training and all(p.grad is None for p in model.parameters())
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    folder = tmp_path/'diagnostic_images'/'1000'
    report = json.loads((folder/'metrics.json').read_text())
    assert report['rows'][1]['attempts'][0]['error'] is None
    assert report['examples'] == rows['examples'] == 2
    assert report['training_update']['loss'] == .2
    assert report['history_encoder'] == dict(variant='fine', token_shape=[2, 17, 17],
        tokens_per_slab=578, feature_channels=128)
    assert [r['dataset'] for r in report['rows']] == ['ordinary', 'unknown']
    assert {p.name for p in folder.iterdir()} == {
        'predictions.png', 'crop_orientation.png', 'encoder.png', 'decoder.png', 'history.png', 'metrics.json'}
    assert not any(p.suffix in ('.npz', '.npy', '.pt') for p in tmp_path.rglob('*'))
    for path in folder.glob('*.png'):
        with Image.open(path) as image:
            assert image.width > 500 and image.height > 500
            image.verify()




def test_diagnostic_annotation_uses_final_crop_frame_without_changing_world_geometry(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.regression.data import ObservationBuilder
    from vesuvius.neural_tracing.fiber_follow.regression.batch_diagnostic import measured_geometry
    from vesuvius.neural_tracing.fiber_follow.shared.heading import orient_item
    cfg = config()
    builder = ObservationBuilder(cfg)
    curve = np.c_[np.ones(9), np.zeros(9), np.arange(-4, 5)]
    item = dict(pos=np.array([10., 20., 30.]), frame=np.eye(3), identity_curve=curve.copy(),
                hist_local=np.array([[0., 0., -1.]]), hmask=np.ones(1))
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.ct_tensor',
                        lambda *a: np.diag([1., 2., .1]))
    def images(items, vol):
        for value in items:
            orient_item(value, vol)
        return {}
    monkeypatch.setattr(builder, 'images', images)
    data = builder.observations([item], None)
    local = data['diagnostic_annotation'][0][data['diagnostic_annotation_mask'][0]].numpy()
    np.testing.assert_allclose(local @ data['crop_frame'][0].numpy().T, curve, atol=1e-6)
    assert measured_geometry(data)['heading_annotation_angle_degrees'] == pytest.approx(0.)
    np.testing.assert_array_equal(data['crop_pos'][0], [10., 20., 30.])


def test_activation_stats_report_nonfinite_and_empty_values_as_unknown():
    from vesuvius.neural_tracing.fiber_follow.regression.batch_diagnostic import tensor_stats
    assert tensor_stats(torch.tensor([float('nan')]))['rms'] is None
    assert tensor_stats(torch.empty(0))['rms'] is None
    measured = tensor_stats(torch.tensor([1., 3., float('nan')]))
    assert measured['mean'] == 2 and measured['std'] == 1
    assert measured['finite_fraction'] == pytest.approx(2/3)


def test_orientation_clips_segments_to_actual_section_including_crossings():
    from vesuvius.neural_tracing.fiber_follow.regression.diagnostic_plots import clipped_polyline
    lower, upper = np.array([-2., -.5, -2.]), np.array([2., .5, 2.])
    # Both sampled endpoints are outside the section, but the segment crosses it.
    line = clipped_polyline([[-1., -2., -1.], [1., 2., 1.]], lower, upper)
    np.testing.assert_allclose(line[:2], [[-.25, -.5, -.25], [.25, .5, .25]])
    assert np.isnan(line[2]).all()
    # Projected coordinates look inside the image, but hidden coordinates aren't.
    assert not len(clipped_polyline([[-1., 3., -1.], [1., 3., 1.]], lower, upper))
    assert not len(clipped_polyline([[4., -2., 0.], [4., 2., 0.]], lower, upper))
    assert not len(clipped_polyline([[0., 0., 0.], [np.nan]*3, [1., 0., 1.]], lower, upper))


def test_orientation_does_not_project_distant_annotation_onto_ct():
    from vesuvius.neural_tracing.fiber_follow.regression.diagnostic_plots import orientation_sheet, CELL
    cfg = config()
    e = dict(label='off-plane', sections=[np.zeros((cfg.fine.depth, cfg.fine.width))]*2
             + [np.zeros((cfg.fine.width, cfg.fine.width))],
             annotation=np.array([[-1., 0., 2.], [1., 0., 3.]]),
             history=np.empty((0, 3)), metrics={'heading_annotation_angle_degrees': 0.})
    pixels = np.asarray(orientation_sheet([e], cfg, 1))
    # In the f=0 image this curve must vanish, while remaining in the v=0 image.
    is_green = (pixels[..., 1].astype(int) > pixels[..., 0].astype(int)+30) & (pixels[..., 2] < 100)
    assert is_green[62:, :CELL[0]].any()
    assert not is_green[62:, 2*CELL[0]:3*CELL[0]].any()
