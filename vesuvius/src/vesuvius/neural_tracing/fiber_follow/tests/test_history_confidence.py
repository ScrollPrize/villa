import copy


from dataclasses import replace


import json


from types import SimpleNamespace


import numpy as np


import pytest


import torch


from vesuvius.neural_tracing.fiber_follow import data as D


from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, arclength, crop_local_grid, frame_from_heading


from vesuvius.neural_tracing.fiber_follow.history_audit import HistoryAudit


from vesuvius.neural_tracing.fiber_follow.history_metrics import observed_measurements


from vesuvius.neural_tracing.fiber_follow.model import FollowNet, FollowNetConfig, history_tangent


from vesuvius.neural_tracing.fiber_follow.supervision import loss_fn


from vesuvius.neural_tracing.fiber_follow.trace import ModelTracer, TraceParams, field_axis


from vesuvius.neural_tracing.fiber_follow.train import load_checkpoint, save_checkpoint


from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec


def array_at(path, values):
    path.mkdir(parents=True)
    (path / '.zarray').write_text(json.dumps(dict(shape=list(values.shape), chunks=list(values.shape),
                                                dtype='|u1', fill_value=0, order='C', filters=None,
                                                compressor=None, zarr_format=2)))
    (path / '0.0.0').write_bytes(values.astype(np.uint8).tobytes())


def volume(tmp_path):
    z, y, x = np.indices((16, 16, 16))
    array_at(tmp_path / 'fields/test_presence.ome.zarr/3', 20 + x + 2*y + 3*z)
    z, y, x = np.indices((32, 32, 32))
    array_at(tmp_path / 'ct/0', x + 2*y + 3*z)
    spec = FiberVolumeSpec(str(tmp_path / 'fields'), ct_zarr=str(tmp_path / 'ct'),
                           ct_level=0, ct_grid_scale=4, inputs='ct+presence')
    return FiberVolume(spec, cache_bytes=1 << 20)


@pytest.mark.parametrize('fused', [False, True])
def test_native_ct_and_presence_sample_the_same_world_positions(tmp_path, monkeypatch, fused):
    vol = volume(tmp_path)
    assert vol.nx is None and vol.ny is None
    assert vol.input_scale == 2 and vol.channels == 2
    crop = CropSpec(depth=8, width=7, behind=3, spacing=.5, history_render='segments')
    frame = frame_from_heading(np.array([1., 2., 3.]))
    item = dict(pos=np.array([7.25, 7.5, 7.75]), frame=frame,
                hist_local=np.zeros((8, 3)), hmask=np.zeros(8))
    grid = torch.from_numpy(crop_local_grid(crop)).float()
    world = item['pos'] + grid.numpy() @ frame.T
    ramp = world[..., 0] + 2*world[..., 1] + 3*world[..., 2]
    monkeypatch.setattr(D, 'FUSED_SAMPLER', fused)
    batch = D.collate_with_volume([item], vol, crop, grid)
    assert batch['x'].shape == (1, 3, 8, 7, 7)
    np.testing.assert_allclose(batch['x'][0, 0], 2*ramp/255, atol=3e-4)
    np.testing.assert_allclose(batch['x'][0, 1], (20+ramp)/255, atol=3e-4)
    assert not batch['x'][0, 2].any()
    assert np.isfinite(field_axis(vol, item['pos'])[0]).all()
    assert vol.sample_image_nearest(np.array([[4., 4., 4.]]))[0] == 48


def test_presence_zero_padding_preserves_ct_and_history(tmp_path):
    vol = volume(tmp_path)
    crop = CropSpec(depth=3, width=3, behind=1)
    grid = torch.from_numpy(crop_local_grid(crop)).float()
    items = [dict(pos=np.array([-10., -10., -10.]), frame=np.eye(3))]
    x = torch.rand(1, 2, 3, 3, 3)
    result = D.add_presence_input(x, items, vol, crop, grid)
    torch.testing.assert_close(result[:, 0], x[:, 0])
    torch.testing.assert_close(result[:, 2], x[:, 1])
    assert not result[:, 1].any()


def test_tangent_fit_uses_several_points_and_ignores_missing_tail():
    p = torch.zeros(1, 6, 3)
    p[0, :, 2] = -torch.arange(6)
    p[0, 1, 0] = 1
    tangent2, _ = history_tangent(p, torch.ones(1, 6), 2)
    tangent6, valid = history_tangent(p, torch.ones(1, 6), 6)
    assert valid.item() and abs(tangent6[0, 0]) < abs(tangent2[0, 0])
    mask = torch.tensor([[1., 1., 0., 1., 1., 1.]])
    p[:, 2:] = 1000
    torch.testing.assert_close(history_tangent(p, mask, 6)[0], tangent2)
    tangent, valid = history_tangent(p, torch.tensor([[1., 0., 0., 0., 0., 0.]]), 6)
    assert not valid.item()
    torch.testing.assert_close(tangent, torch.tensor([[0., 0., 1.]]))


def test_observed_measurements_ignore_absent_and_departed_geometry():
    _, _, hist, mask = score_inputs()
    target = torch.cat([torch.zeros(1, 1, 3), hist[:, :4]], 1)
    hist[:, :4, 0] = 1
    target_mask = torch.ones(1, 5)
    result = observed_measurements(hist, mask, target, target_mask, 4)
    assert result['observed_history_error'][0].item() == pytest.approx(.8)
    mask[:, 2:] = 0
    hist[:, 2:] = 10000
    result = observed_measurements(hist, mask, target, target_mask, 4)
    assert result['observed_history_error'][0].item() == pytest.approx(2/3)
    result = observed_measurements(hist, mask, target, target_mask*0, 4)
    assert all(not valid.item() for _, valid in result.values())


def test_diagnostic_batching_preserves_all_seeds_and_metrics(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.diag import rollout_diag
    points = np.array([[float(x), 0., 0.] for x in range(21)])
    fiber = D.TracedFiber('line', points, arclength(points), '')
    seeds = [dict(fiber=0, t=float(t), sign=1., pos=np.array([t, 0., 0.]),
                  heading=np.array([1., 0., 0.])) for t in range(4, 9)]
    calls = []
    def trace(pos, heading):
        calls.append(len(pos))
        paths = [p[None] + np.arange(3)[:, None]*h[None] for p, h in zip(pos, heading)]
        return paths, ['max_len']*len(pos)
    tracer = SimpleNamespace(p=SimpleNamespace(max_len=12.), trace=trace,
                             vol=SimpleNamespace(sample_image_nearest=lambda q: np.zeros(q.shape[:-1])))
    small = rollout_diag(tracer, [fiber], seeds, tmp_path/'small.png', max_len=4., half=2, batch=2)
    assert calls == [2, 2, 1] and tracer.p.max_len == 12.
    large = rollout_diag(tracer, [fiber], seeds, tmp_path/'large.png', max_len=4., half=2, batch=5)
    assert small == large



def score_inputs():
    hist = torch.zeros(1,8,3)
    hist[0,:,2] = -torch.arange(1,9)
    return None,None,hist,torch.ones(1,8)
