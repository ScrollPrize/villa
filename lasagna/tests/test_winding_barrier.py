import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiber_surface_barrier import prepare_barriers, WindingBarrier, attach_barrier
from test_connected_fiber_surface import fixture
from fiber_surface_repair import restore_targets


def prepare(**kwargs):
    return prepare_barriers([[.5,.5],[2,.5]], [[0,0,0],[0,0,10]],
        [[0,0,-1]]*2, [1,1], [100,200,300], np.array([0.,0.]), 100., (2,7),
        support_inner=1., support_outer=20., **kwargs)


def test_width_mapping_support_and_normalization():
    d=prepare()
    for sample,k,coord,side in zip(d['barrier_source_sample'],d['barrier_winding'],
                                  d['barrier_grid_uv'],d['barrier_side']):
        np.testing.assert_allclose(coord, [[.5,.5],[2,.5],[4.5,.5]][k])
        assert side == (1 if k < sample else -1)
    assert np.isclose(d['barrier_arc_weight'].sum(),1)
    np.testing.assert_array_equal(d['barrier_gate'],d['barrier_winding']==2)
    q=prepare_barriers([[1,.5]], [[0,0,0]], [[0,0,-2]], [1], [100,200,300],
                       np.array([0.,0.]),100.,(2,4))
    assert np.all(q['barrier_own_winding']==1)
    assert set(q['barrier_winding'])=={0,2}
    np.testing.assert_allclose(np.linalg.norm(q['barrier_normal'],axis=1),1)
    assert (q['barrier_grid_uv'][:,0]<=3).all()


def test_sign_clearance_and_geometry_only():
    d=prepare(); d['barrier_gate'][:]=1
    b=WindingBarrier(d,(2,7),100)
    mesh=torch.zeros((1,2,7,3),requires_grad=True)
    with torch.no_grad():
        mesh[0,:,0:2,2]=-10
        mesh[0,:,2:4,2]=10
        mesh[0,:,4:,2]=20
    assert b(mesh)==0
    wrong=-mesh.detach().clone().requires_grad_(True)
    assert b(wrong)>0
    assert b.statistics(wrong)['all_wrong_side_percent']==75
    assert b.statistics(wrong)['all_clearance_violation_percent']==100
    _,t=fixture()
    gm,gc,gl=torch.autograd.grad(b(wrong),(wrong,t.component_delta,t.delta),allow_unused=True)
    assert gm.abs().sum()>0 and gc is None and gl is None
    d['barrier_weight']=np.array(0.)
    assert WindingBarrier(d,(2,7),100)(wrong)==0


def test_saved_raw_xyz_and_roundtrip(tmp_path):
    mesh,t=fixture()
    d=prepare_barriers(t.initial_uv.numpy(),np.zeros((4,3)),[[0,0,-1]]*4,
                      [1]*4,[100,100,300],np.array([0.,0.]),100,(4,5))
    assert np.all(d['barrier_xyz']==0) and np.any(t.xyz.numpy()!=0)
    path=tmp_path/'barriers.npz'; np.savez(path,**d)
    with np.load(path) as f: saved={k:f[k] for k in f.files}
    attach_barrier(t,saved)
    arr=lambda x:x.detach().numpy()
    a=dict(initial_grid_uv=arr(t.initial_uv),xyz=arr(t.xyz),piece=np.array([1,1,2,2]),
        source=np.array([1,1,2,2]),arc=np.array([0,100,0,100]),scales=arr(t.scales),
        component=arr(t.component),crossing=arr(t.crossing),family=arr(t.family),
        v_data_weight=np.array(.1),**t.attachment_state())
    data=dict(grid_uv=a['initial_grid_uv'],xyz=a['xyz'],piece=a['piece'],arc_weight=np.ones(4),
              **{k:a[k] for k in ['component','crossing','family']},**saved)
    restored=restore_targets(data,a,(4,5),100,'cpu')
    torch.testing.assert_close(t.winding_barrier(mesh),restored.winding_barrier(mesh))
    assert t.statistics(mesh)['winding_barrier']==restored.statistics(mesh)['winding_barrier']


def test_invalid_settings():
    with pytest.raises(ValueError): prepare(weight=-1)
    with pytest.raises(ValueError): prepare(clearance=float('nan'))
    with pytest.raises(ValueError, match='finite UV'):
        prepare_barriers([[float('nan'),0]], [[0,0,0]], [[0,0,1]], [1],
                         [100,100], [0,0], 100, (2,3))
