import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fiber_surface_connected import ConnectedUVTargets, source_components, interpolate_offsets
from fiber_surface_repair import restore_targets
from fiber_surface import fit, default_stages
from fiber_surface_connected import filter_components, selected_crossings
import pytest


def test_component_selection():
    # Triple, pair, singleton, transitive chain; repeat a piece of the pair.
    sources = [10,11,12,20,21,30,40,41,42,43,20]
    families = [0,1,0,0,1,1,0,1,0,1,0]
    curves = [dict(source=s, family=f) for s,f in zip(sources, families)]
    saved = dict(v_offset_source=np.array(sources),
                 v_offset_a=np.array([0,1,3,6,7,8]),
                 v_offset_b=np.array([1,2,4,7,8,9]))
    kept, info = filter_components(curves, saved)
    assert [c['source'] for c in kept] == [10,11,12,40,41,42,43]
    assert info['retained_components'] == 2
    assert info['retained_sources'] == 7
    assert info['input_sources'] == 10
    a,b = selected_crossings(saved, info['retained_source_ids'])
    assert len(a) == len(b) == 5
    assert len(filter_components(curves, saved, 0)[0]) == len(curves)
    assert len(filter_components(curves, saved, 4)[0]) == 4
    with pytest.raises(ValueError, match='cuts'):
        selected_crossings(saved, [10])
    with pytest.raises(ValueError, match='No fiber'):
        filter_components(curves, saved, 5)
    with pytest.raises(ValueError, match='consistent'):
        filter_components(curves+[dict(source=10, family=1)], saved)
    with pytest.raises(ValueError, match='nonnegative'):
        filter_components(curves, saved, 2.5)


def fixture():
    y, x = np.mgrid[:4, :5]
    mesh = torch.tensor(np.stack([x*100, y*100, x*0], -1)[None], dtype=torch.float32, requires_grad=True)
    targets = ConnectedUVTargets([[1,1],[2,1],[1,1],[1,2]],
        [[110,100,20],[210,100,20],[110,100,20],[110,200,20]],
        [50]*4, (4,5), arc=[0,100,0,100], source=[1,1,2,2],
        piece=[1,1,2,2], scales=[[500,1000]]*4, step=100,
        component=[0]*4, crossing=[True,False,True,False], family=[0,0,1,1])
    return mesh, targets


def test_offsets_and_components():
    np.testing.assert_allclose(interpolate_offsets([-10,0,5,10,20], [0,10], [10,30], 5), [10,10,20,30,30])
    np.testing.assert_allclose(interpolate_offsets([1,2], [], [], -5), [-5,-5])
    np.testing.assert_allclose(interpolate_offsets([0], [0,0], [10,30], 5), [20])
    labels = source_components([1,1,2,3,4], [(1,2),(2,3)])
    assert labels[0] == labels[1] == labels[2] == labels[3]
    assert labels[4] != labels[0]


def test_exact_locks_gradient_routing_and_weight():
    mesh, t = fixture()
    assert torch.isclose(t.data_weights[t.family==0].sum(), torch.tensor(1.))
    assert torch.isclose(t.data_weights[t.family==1].sum(), torch.tensor(.1))
    normal, tangent, penalty = t.components(mesh)
    gm,gc = torch.autograd.grad(normal, (mesh,t.component_delta), allow_unused=True)
    assert gm.abs().sum() > 0 and gc is None
    gm,gc,gl = torch.autograd.grad(tangent, (mesh,t.component_delta,t.delta), allow_unused=True)
    assert gm is None and gc.abs().sum()>0
    assert gl[t.crossing].abs().sum() == 0 and gl[~t.crossing].abs().sum()>0
    optimizer = torch.optim.Adam(t.optimizer_groups())
    for _ in range(5):
        optimizer.zero_grad(); sum(t.components(mesh.detach())).backward(); optimizer.step()
    torch.testing.assert_close(t.current_uv()[0],t.current_uv()[2],rtol=0,atol=0)
    assert t.component_delta.abs().sum()>0
    assert t.delta[~t.crossing].abs().sum()>0
    assert t.statistics(mesh)['crossing_lock_error_base'] < 1e-4


def test_roundtrip_and_real_optimizer():
    mesh, t = fixture()
    t.geometry_tangent_weight = .3
    final, stats = fit(mesh.detach().numpy()[0], t, step=100, stages=default_stages(steps=3,lr=.01))
    assert np.isfinite(final).all() and stats['crossing_lock_error_base']<1e-4
    arr=lambda x: x.detach().cpu().numpy()
    a=dict(initial_grid_uv=arr(t.initial_uv),final_grid_uv=arr(t.current_uv()),xyz=arr(t.xyz),
        piece=np.array([1,1,2,2]),source=np.array([1,1,2,2]),arc=np.array([0,100,0,100]),
        scales=arr(t.scales), component=arr(t.component),crossing=arr(t.crossing),family=arr(t.family),
        component_delta=arr(t.component_delta),local_delta=arr(t.delta),v_data_weight=np.array(.1),
        geometry_tangent_weight=np.array(.3))
    d=dict(grid_uv=a['initial_grid_uv'],xyz=a['xyz'],piece=a['piece'],arc_weight=np.ones(4),
        **{k:a[k] for k in ['component','crossing','family']})
    restored=restore_targets(d,a,(4,5),100,'cpu')
    assert restored.geometry_tangent_weight == .3
    torch.testing.assert_close(t.current_uv(),restored.current_uv())
    torch.testing.assert_close(t.data_weights,restored.data_weights)
    with torch.no_grad():
        restored.component_delta.add_(.125)
        restored.delta[~restored.crossing] += .25
    continued = {**a, **restored.attachment_state()}
    second = restore_targets(d,continued,(4,5),100,'cpu')
    torch.testing.assert_close(second.current_uv(),restored.current_uv())
