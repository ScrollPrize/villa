"""Diagnostics observe the deployed policy without changing training state."""
import copy
import json

import numpy as np
from PIL import Image
import pytest
import torch

from model_fixtures import coordinate_batch as batch, coordinate_config
from label_fixtures import set_terminal
from vesuvius.neural_tracing.fiber_follow.models.model import CoordinateRegressionFollower
from vesuvius.neural_tracing.fiber_follow.evaluation.batch_diagnostic import layer_capture, render_microbatch


def test_layer_capture_preserves_predictions_and_cleans_up():
    cfg = coordinate_config(recurrent_refinement_steps=1)
    model = CoordinateRegressionFollower(cfg).eval()
    data = batch(cfg, 1)
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
    assert set(layers['scorer']) == {'input', *[f'block {i+1}' for i in range(model.cfg.scorer_layers)]}
    assert layers['history_depths'] == {'generator': model.cfg.decoder_layers, 'scorer': model.cfg.scorer_layers}
    assert layers['encoder']['encoder output'].ndim == 2
    assert layers['history']['tokens'].shape == (8, 17, 17)
    assert layers['history']['convolution'].shape == (2, 17, 17)
    for head in ('generator', 'scorer'):
        for attention in layers['history'][head+'_attention']:
            assert torch.isfinite(attention).all()
            torch.testing.assert_close(attention.sum(-1), torch.ones(cfg.n_future))
            assert not attention[:, 2:].any()
    # Distinct layer/attempt values catch averaging the wrong retry or layer count.
    from vesuvius.neural_tracing.fiber_follow.evaluation.diagnostic_plots import display_example
    assert len(layers['history']['scorer_attention']) == 2*model.cfg.scorer_layers
    for head, depth in layers['history_depths'].items():
        layers['history'][head+'_attention'] = [torch.full((cfg.n_future, 8), 100.*attempt+layer)
            for attempt in range(2) for layer in range(depth)]
    selected = dict(actual, selected_refinement=torch.ones(1, dtype=torch.long))
    example = display_example(data, selected, {}, layers, cfg, 'test', {})
    np.testing.assert_allclose(example['attention']['scorer'], 100.+(model.cfg.scorer_layers-1)/2)
    np.testing.assert_allclose(example['attention']['generator'], 100.)
    assert 'forward_cached' not in model.history_attention.__dict__
    assert not model.encoder.patch_projection._forward_hooks
    with pytest.raises(RuntimeError):
        with layer_capture(model):
            raise RuntimeError('inference failed')
    assert 'forward_cached' not in model.confidence_scorer.history_attention.__dict__
    assert not model.encoder.blocks[0]._forward_pre_hooks


@pytest.mark.parametrize('kind', ['coordinate_regression', 'flow_matching'])
def test_render_emits_readable_images_strict_json_and_preserves_rng(tmp_path, kind):
    from test_flow_model import config as flow_config
    from vesuvius.neural_tracing.fiber_follow.models.model import build_model
    cfg = coordinate_config(recurrent_refinement_steps=1) if kind == 'coordinate_regression' else flow_config()
    model = build_model(cfg).train()
    data = batch(cfg, 3)
    data['identity_observable'] = torch.tensor([0., 1., 1.])
    data['dense_mask'][1] = 0
    data['dense_ab'][1] = float('nan')
    data['x']['history_valid'][1] = False
    set_terminal(data, 2)
    rng, numpy_rng = torch.get_rng_state(), np.random.get_state()
    data['dataset_id'] = torch.tensor([0, 1, 0])
    rows = render_microbatch(model, data, tmp_path, 1000, device='cpu', n_commit=4, tolerance=1.5,
                            dataset_names=['ordinary', 'unknown'], training_metrics={'loss': .2})
    assert model.training and all(p.grad is None for p in model.parameters())
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
    folder = tmp_path/'diagnostic_images'/'1000'
    report = json.loads((folder/'metrics.json').read_text())
    # Unobservable identity, unknown geometry and terminal labels stay distinct.
    assert not any(report['rows'][0]['known'])
    assert report['rows'][1]['attempts'][0]['error'] is None and not any(report['rows'][1]['known'])
    assert all(report['rows'][2]['known']) and not any(report['rows'][2]['labels'])
    assert report['examples'] == rows['examples'] == 3
    assert report['training_update']['loss'] == .2
    assert report['history_encoder'] == dict(variant='fine', token_shape=[2, 17, 17],
        tokens_per_slab=2*17*17+3, feature_channels=128)
    assert [r['dataset'] for r in report['rows']] == ['ordinary', 'unknown', 'ordinary']
    assert {p.name for p in folder.iterdir()} == {
        'predictions.png', 'crop_orientation.png', 'encoder.png', 'decoder.png', 'history.png', 'metrics.json'}
    assert not any(p.suffix in ('.npz', '.npy', '.pt') for p in tmp_path.rglob('*'))
    for path in folder.glob('*.png'):
        with Image.open(path) as image:
            assert image.width > 500 and image.height > 500
            image.verify()


def test_diagnostic_annotation_uses_final_crop_frame_without_changing_world_geometry(monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.data.observations import ObservationBuilder
    from vesuvius.neural_tracing.fiber_follow.evaluation.batch_diagnostic import measured_geometry
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import orient_item
    cfg = coordinate_config()
    builder = ObservationBuilder(cfg)
    curve = np.c_[np.ones(9), np.zeros(9), np.arange(-4, 5)]
    item = dict(pos=np.array([10., 20., 30.]), frame=np.eye(3), identity_curve=curve.copy(),
                hist_local=np.array([[0., 0., -1.]]), hmask=np.ones(1))
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.ct_tensor',
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


def test_diagnostic_helpers_report_unknown_values_and_clip_annotation_to_sections():
    from vesuvius.neural_tracing.fiber_follow.evaluation.batch_diagnostic import tensor_stats
    from vesuvius.neural_tracing.fiber_follow.evaluation.diagnostic_plots import clipped_polyline, orientation_sheet, CELL
    assert tensor_stats(torch.tensor([float('nan')]))['rms'] is None
    assert tensor_stats(torch.empty(0))['rms'] is None
    measured = tensor_stats(torch.tensor([1., 3., float('nan')]))
    assert measured['mean'] == 2 and measured['std'] == 1
    assert measured['finite_fraction'] == pytest.approx(2/3)
    lower, upper = np.array([-2., -.5, -2.]), np.array([2., .5, 2.])
    # Both sampled endpoints are outside the section, but the segment crosses it.
    line = clipped_polyline([[-1., -2., -1.], [1., 2., 1.]], lower, upper)
    np.testing.assert_allclose(line[:2], [[-.25, -.5, -.25], [.25, .5, .25]])
    assert np.isnan(line[2]).all()
    # Projected coordinates look inside the image, but hidden coordinates aren't.
    assert not len(clipped_polyline([[-1., 3., -1.], [1., 3., 1.]], lower, upper))
    assert not len(clipped_polyline([[4., -2., 0.], [4., 2., 0.]], lower, upper))
    assert not len(clipped_polyline([[0., 0., 0.], [np.nan]*3, [1., 0., 1.]], lower, upper))
    cfg = coordinate_config()
    e = dict(label='off-plane', sections=[np.zeros((cfg.fine.depth, cfg.fine.width))]*2
             + [np.zeros((cfg.fine.width, cfg.fine.width))],
             annotation=np.array([[-1., 0., 2.], [1., 0., 3.]]),
             history=np.empty((0, 3)), metrics={'heading_annotation_angle_degrees': 0.})
    pixels = np.asarray(orientation_sheet([e], cfg, 1))
    # In the f=0 image this curve must vanish, while remaining in the v=0 image.
    is_green = (pixels[..., 1].astype(int) > pixels[..., 0].astype(int)+30) & (pixels[..., 2] < 100)
    assert is_green[62:, :CELL[0]].any()
    assert not is_green[62:, 2*CELL[0]:3*CELL[0]].any()


def test_render_resolves_decision_memory_before_slicing_rows(tmp_path):
    from test_decision_memory import memory_config, decision_inputs
    from vesuvius.neural_tracing.fiber_follow.models.model import build_model
    from vesuvius.neural_tracing.fiber_follow.models.decision_memory import ENTRY_FEATURES
    from vesuvius.neural_tracing.fiber_follow.train.live_continuation import DecisionStore, LiveContinuation
    cfg = memory_config(recurrent_refinement_steps=1)
    model = build_model(cfg).eval()
    data = batch(cfg, 3)
    data['x'] = {k: v for k, v in data['x'].items() if not k.startswith('history_')}
    data['x'].update(decision_inputs(cfg, 3, crops=1))
    # Row 1 reads one entry encoded from a crop; row 2 one recorded chain entry; row 0 none.
    data['x']['history_valid'][1, 0] = data['x']['history_encode'][1, 0] = True
    data['x']['history_crops'][:] = torch.rand_like(data['x']['history_crops'])
    data['x']['history_valid'][2, 0] = True
    data['x']['history_keys'][2, 0], data['x']['history_chain'][2] = 4, 77
    data['x']['history_path_valid'][1:, 0] = True
    live = LiveContinuation.__new__(LiveContinuation)
    live.memory = DecisionStore(max_age=10)
    live.memory.put(77, 4, torch.randn(ENTRY_FEATURES, cfg.hidden).to(torch.bfloat16), {4}, step=0)
    data['dataset_id'] = torch.tensor([0, 0, 0])
    rows = render_microbatch(model, data, tmp_path, 1000, device='cpu', n_commit=4, tolerance=1.5,
                             dataset_names=['ordinary'], memory=live)
    report = json.loads((tmp_path/'diagnostic_images'/'1000'/'metrics.json').read_text())
    assert rows['examples'] == 3
    assert [sum(r['history_valid']) for r in report['rows']] == [0, 1, 1]
    assert report['history_encoder']['token_shape'] == [3, 11, 11]
    # Without the store, recorded slots are dropped instead of read as zeros.
    rows = render_microbatch(model, data, tmp_path/'none', 1000, device='cpu', n_commit=4, tolerance=1.5,
                             dataset_names=['ordinary'])
    report = json.loads((tmp_path/'none'/'diagnostic_images'/'1000'/'metrics.json').read_text())
    assert [sum(r['history_valid']) for r in report['rows']] == [0, 1, 0]
