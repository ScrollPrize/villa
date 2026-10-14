import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiber_surface_repair import repair, defects, bilinear_matrix
from opt_loss_surface_guard import SurfaceGuard
from fiber_surface_repair import restore_targets


def plane():
    y,x = np.mgrid[:11,:11]
    return np.stack([x*100., y*100., x*0.],-1).astype(np.float32)


def test_fold_repair_boundary_and_identity():
    good = plane()
    x = good.copy(); x[5,5,0] += 270
    uv = np.array([[2.,2.],[8.,8.]])
    target = bilinear_matrix(uv, x.shape[:2])@good.reshape(-1,3)
    repaired, records = repair(x, uv, target, np.ones(2)*100, 100, halos=(2,))
    assert sum(m.sum() for m in defects(repaired,100)) == 0
    assert any(r['selected'] for r in records)
    np.testing.assert_array_equal(repaired[[0,-1]], x[[0,-1]])
    np.testing.assert_array_equal(repaired[:,[0,-1]], x[:,[0,-1]])
    unchanged, records = repair(good, uv, target, np.ones(2), 100, halos=(2,))
    np.testing.assert_array_equal(unchanged,good)
    assert records == []


def test_guard_and_corner_fold():
    ref = torch.tensor(plane())
    guard = SurfaceGuard(ref)
    assert guard(ref[None]).item() == 0
    x = ref.clone(); x[5,5,0]+=270
    x = x[None].requires_grad_()
    loss = guard(x); loss.backward()
    assert loss>0 and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    # Concave quad: first diagonal's triangles agree, but one corner reverses.
    quad = np.array([[[0,0,0],[2,0,0]],[[0,2,0],[.5,.5,0]]], dtype=float)
    assert defects(quad,1)[0].item()


def test_impossible_frozen_boundary_unchanged():
    x = np.array([[[0,0,0],[2,0,0]],[[0,2,0],[-1,-1,0]]],dtype=np.float32)
    output, records = repair(x, np.array([[.5,.5]]), np.zeros((1,3)), np.ones(1),1,halos=(2,))
    np.testing.assert_array_equal(output,x)
    assert records == []


def test_fit_guard_rejects_and_does_not_move_uv():
    x = plane(); x[5,5,0]+=270
    uv = np.array([[5.,5.]])
    before_uv = uv.copy()
    output, records = repair(x, uv, x[5,5][None], np.array([100.]),100,
                             halos=(2,), guide_weights=(0.,))
    np.testing.assert_array_equal(output,x)
    np.testing.assert_array_equal(uv,before_uv)
    assert records and not any(r['selected'] for r in records)
    assert any(not r['global_fit_ok'] for r in records)


def test_restore_preserves_reference_and_saved_uv():
    uv = np.array([[2.,2.],[3.,2.]])
    data = dict(grid_uv=uv, xyz=np.zeros((2,3)), piece=np.array([1,1]), arc_weight=np.ones(2))
    a = dict(initial_grid_uv=uv, final_grid_uv=uv+.2, xyz=data['xyz'], piece=data['piece'],
             arc=np.array([0.,100.]), scales=np.ones((2,2))*1000, source=np.array([4,4]))
    t = restore_targets(data,a,(11,11),100,'cpu')
    np.testing.assert_allclose(t.current_uv().detach(),uv+.2)
    np.testing.assert_allclose(t.initial_uv,uv)
    assert t.delta.requires_grad


def test_guard_excludes_near_collapsed_cells():
    x = torch.tensor(plane())
    x[:,:,0]*=1e-6
    guard = SurfaceGuard(x,step=100)
    assert not guard.valid.any()


def test_tangent_blended_repair_preserves_boundary_and_improves():
    x = plane(); x[5,5,0]+=270
    uv = np.array([[2.,2.],[8.,8.]])
    target = bilinear_matrix(uv,x.shape[:2])@plane().reshape(-1,3)
    y, records = repair(x,uv,target,np.ones(2)*100,100,halos=(4,2),largest_first=True,
                         seam_weight=10.,blend_fractions=(1.,.5,.25))
    assert sum(m.sum() for m in defects(y,100))<sum(m.sum() for m in defects(x,100))
    assert all(not (b&~a).any() for a,b in zip(defects(x,100),defects(y,100)))
    np.testing.assert_array_equal(x[[0,-1]],y[[0,-1]])
    np.testing.assert_array_equal(x[:,[0,-1]],y[:,[0,-1]])
    assert any(r['selected'] for r in records)
