"""Fixed-UV surface regression tests: no volume, normals, or GPU needed."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiber_surface import prepare, ruled_init, geometry_stats, fit, default_stages
from opt_loss_fixed_uv import FixedUVTargets, ramp_weight


def curves():
    u = np.linspace(0, 400, 17)
    return [dict(piece=i, source=i, u=u, v=v,
                 xyz=np.column_stack((u, np.full(len(u), v), 500+15*np.sin(u/100)+v*.1)))
            for i, v in enumerate([0., 100., 200.])]


def test_prepare_endpoints_crossings_and_arc_weights():
    c = curves()
    c[0] = {**c[0], 'u': c[0]['u']+7}
    selected, x, y, guv, xyz, weights, owners = prepare(c, step=100, sample_spacing=71)
    q = guv[owners == 0, 0]*100+x[0]
    assert q[0] == pytest.approx(7) and q[-1] == pytest.approx(407)
    assert all(edge in q for edge in [100, 200, 300, 400])
    assert weights.sum() == pytest.approx(1200.)
    assert np.all(weights > 0)
    np.testing.assert_allclose(xyz[owners == 0][0], c[0]['xyz'][0])


def test_ruled_extrapolation_and_h_exact():
    c = curves()
    x, y = np.arange(-100, 601, 100), np.arange(-100, 401, 100)
    mesh, support = ruled_init(c, x, y, max_extrapolation_slope=2.)
    np.testing.assert_allclose(mesh[1:4, 1:6, 1], np.broadcast_to([0, 100, 200], (5, 3)).T)
    np.testing.assert_allclose(mesh[:, :, 0], np.broadcast_to(x, (6, 8)))
    assert (support[0, 1:6] == 1).all()
    assert (support[:, 0] == 4).all()
    assert geometry_stats(mesh)['opposing_triangle_quads'] == 0


def test_single_h_column_borrows_transverse_slope():
    c = curves()
    c[1] = {**c[1], 'u': c[1]['u'][4:], 'xyz': c[1]['xyz'][4:]}
    c[2] = {**c[2], 'u': c[2]['u'][4:], 'xyz': c[2]['xyz'][4:]}
    mesh, support = ruled_init(c, np.arange(0, 401, 100), np.arange(0, 201, 100), max_extrapolation_slope=2.)
    assert (support[:, 0] == 2).all()
    np.testing.assert_allclose(mesh[:, 0, 1], [0, 100, 200])
    with pytest.raises(ValueError, match='two distinct H'):
        ruled_init(c[:1], np.arange(0, 401, 100), np.arange(0, 201, 100))


def test_fixed_uv_gradient_and_boundary():
    a = FixedUVTargets([[.25, .5], [2., 2.]], [[1, 2, 3], [4, 5, 6]], [1, 2], (3, 3))
    xyz = torch.zeros((1, 3, 3, 3), requires_grad=True)
    loss, _, _ = a.loss(res=SimpleNamespace(xyz_lr=xyz, params=SimpleNamespace(mesh_step=1)))
    loss.backward()
    assert torch.count_nonzero(xyz.grad.norm(dim=-1)) == 5
    np.testing.assert_allclose(a.weights.sum(1), [1, 1])
    assert not a.weights.requires_grad and not a.xyz.requires_grad
    with pytest.raises(ValueError, match='unchanged'):
        a.predict(torch.zeros(1, 4, 4, 3))
    with pytest.raises(ValueError, match='outside'):
        FixedUVTargets([[2.1, 0]], [[0, 0, 0]], [1], (3, 3))


@pytest.mark.parametrize('bounds', [[0, 2], [-1, 2], [3, 2], [1, float('nan')], [1]])
def test_bad_ramp(bounds):
    with pytest.raises(ValueError):
        ramp_weight(bounds, 0, 10)


def test_positive_ramp():
    assert ramp_weight([1, 100], 0, 10) == 1
    assert ramp_weight([1, 100], 5, 10) == 10
    assert ramp_weight([1, 100], 10, 10) == 100


def test_nearly_coincident_heights_do_not_explode_extrapolation():
    c = curves()[:2]
    c[1]['v'] = .00001
    mesh, _ = ruled_init(c, np.arange(0, 401, 100), np.arange(-100, 201, 100))
    assert np.isfinite(mesh).all()
    assert np.max(np.abs(mesh)) < 1000


def test_single_h_uses_continuous_bounded_donor_profile():
    c = curves()
    for i in (1, 2):
        c[i] = {**c[i], 'u': c[i]['u'][4:], 'xyz': c[i]['xyz'][4:].copy()}
    c[2]['xyz'][:, 1] += 500
    x, y = np.arange(0, 401, 100), np.arange(0, 201, 10)
    mesh, support = ruled_init(c, x, y)
    assert (support[:, 0] == 2).all()
    assert np.linalg.norm(np.diff(mesh[:, 0], axis=0), axis=1).max() <= 10.0001
    np.testing.assert_allclose(mesh[0, 0], c[0]['xyz'][0])


def test_ramp_must_not_restart_weaker_in_later_stage():
    c, x, y, guv, target, weights, _ = prepare(curves(), step=100)
    initial, _ = ruled_init(c, x, y)
    targets = FixedUVTargets(guv, target, weights, initial.shape[:2])
    stages = default_stages(1)
    stages['stages'] = stages['stages']*2
    with pytest.raises(ValueError, match='decrease between stages'):
        fit(initial, targets, step=100, stages=stages)


def test_existing_optimizer_synthetic_fit():
    torch.set_num_threads(1)
    c, x, y, guv, target, weights, _ = prepare(curves(), step=100)
    initial, _ = ruled_init(c, x, y)
    initial[..., 2] += 8
    targets = FixedUVTargets(guv, target, weights, initial.shape[:2])
    before = targets.statistics(torch.as_tensor(initial)[None])
    stages = default_stages(60, .3)
    stages['base'].update(smooth=.05, step=.05, bend=.05)
    final, stats = fit(initial, targets, step=100, stages=stages)
    assert stats['rms_base'] < before['rms_base']*.5
    assert np.isfinite(final).all()
    assert geometry_stats(final)['opposing_triangle_quads'] == 0


def test_correspondence_independent_of_pyramid_trainability():
    import model
    c, x, y, guv, target, weights, _ = prepare(curves(), step=100)
    initial, _ = ruled_init(c, x, y)
    mdl = model.Model3D.from_tifxyz_crop(torch.as_tensor(initial), torch.ones(initial.shape[:2], dtype=torch.bool),
                                       device=torch.device('cpu'), mesh_step=100)
    targets = FixedUVTargets(guv, target, weights, initial.shape[:2])
    ref = targets.predict(mdl._grid_xyz()).detach()
    for level in range(len(mdl.mesh_ms)):
        for i, p in enumerate(mdl.mesh_ms):
            p.requires_grad_(i >= level)
        torch.testing.assert_close(targets.predict(mdl._grid_xyz()), ref)
