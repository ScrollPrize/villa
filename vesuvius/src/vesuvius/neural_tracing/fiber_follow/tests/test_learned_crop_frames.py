"""Learned frame plumbing: observable-only inputs, orientation, causality and CT footprints."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, crop_local_grid, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import orient_items, crop_frame_bounds, predict_frames
from vesuvius.neural_tracing.fiber_follow.tracing.heading import LEARNED_FRAME_POLICY, FRAME_POLICY, ct_seed_heading
from vesuvius.neural_tracing.fiber_follow.tracing.trace import ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.data.history_slabs import load_slabs, slab_layout


class Predictor:
    def __init__(self):
        self.calls = []
        self.model = SimpleNamespace(cfg=SimpleNamespace(patch=CropSpec(depth=32, width=32, behind=4, spacing=1.25)))

    def predict_frames(self, vol, positions, priors, paths, families, previous=None, pool=None):
        self.calls.append(dict(positions=positions, priors=priors, paths=paths, families=families, previous=previous))
        return [frame_from_heading(np.array([.3, .1, 1.]), np.array([1., 0., 0.])) for _ in positions]

    def predict_with_normals(self, vol, positions, priors, paths, pool=None, *, families=None):
        self.calls.append(dict(positions=positions, priors=priors, paths=paths, families=families))
        return [np.array([1., 0., 0.]) for _ in positions], [np.array([1., .3, .1]) for _ in positions]


def item(length=80):
    path = np.c_[np.full(length+1, 100.), np.full(length+1, 100.), 100.+np.arange(length+1)]
    return dict(pos=path[-1], frame=np.eye(3), observed_path=path, fiber_family='V',
                seed_pos=path[0], seed_tangent=np.array([0., 0., 1.]), seed_valid=True, seed_age=float(length),
                hist_local=np.array([[0., 0., -1.], [0., 0., -2.]]), hmask=np.ones(2),
                fut_local=np.array([[1., .5, 1.], [2., .7, 2.]]),
                plane_ab=np.array([[1., .5], [2., .7]]), planes=np.array([1., 2.]),
                identity_curve=np.array([[0., 0., -10.], [1., .5, 1.]]))


def test_current_frame_uses_observed_inputs_and_preserves_world_labels():
    before = item()
    a, b = deepcopy(before), deepcopy(before)
    b['fut_local'] *= 1000
    b['identity_curve'] *= -1000
    predictor = Predictor()
    orient_items([a, b], None, predictor)
    assert len(predictor.calls) == 1
    np.testing.assert_array_equal(a['frame'], b['frame'])
    assert a['frame_policy'] == LEARNED_FRAME_POLICY and a['ct_frame_diagnostics']['source'] == 3
    for key in ('hist_local', 'fut_local', 'identity_curve'):
        np.testing.assert_allclose(a[key] @ a['frame'].T, before[key], atol=1e-12)
    np.testing.assert_allclose(np.c_[a['plane_ab'], a['planes']] @ a['frame'].T,
                               np.c_[before['plane_ab'], before['planes']], atol=1e-12)
    np.testing.assert_array_equal(predictor.calls[0]['paths'][0], before['observed_path'])
    assert predictor.calls[0]['families'] == ['V', 'V']
    orient_items([a], None, predictor)
    assert len(predictor.calls) == 1  # recorded/final geometry is not predicted again


def test_learned_bounds_cover_any_new_heading_and_model_patch():
    predictor, sample = Predictor(), item()
    crop = CropSpec(depth=120, width=104, behind=48, spacing=.5)
    vol = SimpleNamespace(input_scale=2.)
    bounds = list(crop_frame_bounds(sample, crop, vol, predictor))
    rng = np.random.default_rng(4)
    for spec, (start, size) in zip((crop, predictor.model.cfg.patch), bounds):
        corners = crop_local_grid(spec)[np.ix_([0, spec.depth-1], [0, spec.width-1], [0, spec.width-1])].reshape(-1, 3)
        for _ in range(20):
            frame = frame_from_heading(rng.normal(size=3), rng.normal(size=3))
            xyz = (corners @ frame.T+sample['pos'])*vol.input_scale
            assert np.all(xyz[:, ::-1] >= start) and np.all(xyz[:, ::-1] < start+size)


def test_historical_roll_keeps_path_heading_and_only_supplies_prefix(monkeypatch):
    predictor, sample = Predictor(), item(160)
    sample['frame_policy'] = LEARNED_FRAME_POLICY
    original = slab_layout(sample)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.data.history_slabs.frame_predictor', lambda cfg: predictor)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.data.crop_sampling.scalar_crops',
        lambda items, vol, crop, *args, **kwargs: torch.zeros(len(items), 1, crop.depth, crop.width, crop.width))
    result = load_slabs([sample], None, SimpleNamespace(fine=CropSpec()), pool=None)
    assert (result['history_frame_source'][result['history_valid']] == 3).all()
    for old, slab in zip(original, sample['_sampled_slabs']):
        np.testing.assert_allclose(slab['frame'][:, 2], old['frame'][:, 2], atol=1e-12)
    for call in predictor.calls:
        for position, path in zip(call['positions'], call['paths']):
            np.testing.assert_array_equal(path[-1], position)
            assert np.all(path[:, 2] <= position[2])


def test_tracer_uses_learned_frame_at_seed_and_each_decision(monkeypatch):
    predictor = Predictor()
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.trace.frame_predictor', lambda cfg: predictor)
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.trace.ct_frame',
                        lambda *args, **kwargs: pytest.fail('CT roll must not be called'))
    class Model(torch.nn.Module):
        cfg = SimpleNamespace(n_future=1, max_recovery_distance=6., frame_checkpoint='frozen.pt')
        def forward(self, x, hist, mask):
            return dict(points=torch.tensor([[[0., 0., 1.]]]).expand(len(hist), -1, -1),
                        confidence=torch.ones(len(hist), 1))
    class Tracer(ModelTracer):
        def build_inputs(self, *args):
            return None
    tracer = Tracer(Model(), SimpleNamespace(shape=(1000,)*3), CropSpec(), n_history=4,
                    params=TraceParams(n_commit=1, max_len=2.), device='cpu')
    seen = []
    try:
        paths, reasons = tracer.trace(np.array([[100., 100., 100.]]), np.array([[0., 0., 1.]]),
                                      families=['H'], on_decision=lambda i, s: seen.append(s))
    finally:
        tracer.close()
    assert reasons == ['max_len'] and len(seen) == 2 and len(predictor.calls) == 2
    assert all(s['frame_policy'] == LEARNED_FRAME_POLICY and s['fiber_family'] == 'H' for s in seen)
    assert predictor.calls[0]['paths'][0].shape == (1, 3)
    assert len(predictor.calls[1]['paths'][0]) > 1
    np.testing.assert_allclose(predictor.calls[1]['previous'][0], seen[0]['frame'])


def test_seed_tensor_uses_2_8_with_full_halo(monkeypatch):
    calls = []
    def tensor(cube, center, **kwargs):
        calls.append((cube.shape, center, kwargs))
        return np.diag([4., 1., .1])
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.ct_structure_tensor', tensor)
    vol = SimpleNamespace(input_scale=2., ct=SimpleNamespace(shape=(200,)*3,
        read=lambda start, size: np.zeros(tuple(size))))
    ct_seed_heading(vol, np.array([50., 50., 50.]), 'H')
    assert calls[0][0] == (77,)*3
    np.testing.assert_array_equal(calls[0][1], [38.]*3)
    assert calls[0][2] == dict(derivative_sigma=2., integration_sigma=8.)
