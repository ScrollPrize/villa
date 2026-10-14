import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from opt_loss_fixed_uv import SlidingUVTargets
from fiber_surface import prepare_sliding, ruled_init, fit, default_stages


def test_shared_geometry_masks_match_counts():
    from fiber_surface import geometry_masks, geometry_stats
    xyz = np.array([[[0., 0., 0.], [1., 0., 0.]],
                    [[0., 1., 0.], [-1., -1., 0.]]])
    degenerate, folded = geometry_masks(xyz)
    stats = geometry_stats(xyz)
    assert folded.item() and not degenerate.item()
    assert stats['opposing_triangle_quads'] == int(folded.sum()) == 1


def fixture():
    y, x = np.mgrid[:4, :5]
    mesh = torch.tensor(np.stack((x*100, y*100, x*0), -1)[None], dtype=torch.float32, requires_grad=True)
    targets = SlidingUVTargets([[.8, 1.], [1.8, 1.]], [[100, 100, 20], [200, 100, 20]],
                               [50, 50], (4, 5), arc=[0, 100], scales=[[500, 1000]]*2,
                               source=[1, 1], step=100)
    return mesh, targets


def test_gradient_routing():
    mesh, targets = fixture()
    normal, tangent, penalty = targets.components(mesh)
    gm, gu = torch.autograd.grad(normal, (mesh, targets.delta), allow_unused=True)
    assert gm.abs().sum() > 0 and gu is None
    gm, gu = torch.autograd.grad(tangent, (mesh, targets.delta), allow_unused=True)
    assert gm is None and gu.abs().sum() > 0
    penalty.backward()
    assert torch.isfinite(targets.delta.grad).all()


def test_cross_cell_and_boundary_derivatives():
    mesh, targets = fixture()
    for u in (.99, 1.01, -.1, 4.1):
        point = torch.tensor([[u, 1.2]], requires_grad=True)
        xyz, _ = targets.sample(mesh.detach(), point)
        grad, = torch.autograd.grad(xyz.sum(), point)
        torch.testing.assert_close(grad, torch.tensor([[100., 100.]]))


def test_curved_uv_sampling_arc_weights_and_initialization():
    curves = [dict(piece=i, source=i, arc=np.array([0., 200., 400.]), winding_width=1000.,
                   u=np.array([0., 200., 400.]), v=np.array([v, v+60, v]),
                   xyz=np.array([[0,v,0], [200,v,5], [400,v,0]], dtype=float))
              for i, v in enumerate([0., 200.])]
    result = prepare_sliding(curves, step=100)
    selected, x, y, guv, xyz, weight, owner, arc, scales, source = result
    assert abs(weight.sum()-800) < 1e-6
    assert np.ptp(guv[owner == 0, 1]) > 0
    initial, _ = ruled_init(selected, x, y)
    assert np.isfinite(initial).all()
    targets = SlidingUVTargets(guv, xyz, weight, initial.shape[:2], arc=arc, scales=scales,
                               source=source, piece=owner, step=100)
    stages = default_stages(steps=5, lr=.1)
    final, stats = fit(initial, targets, step=100, stages=stages)
    assert np.isfinite(final).all() and np.isfinite(stats['mean_base'])
    assert targets.delta.detach().abs().sum() > 0
