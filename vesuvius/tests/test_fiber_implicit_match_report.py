import numpy as np
import torch
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_implicit_match_report import candidates,select


def fid(k):
    return f'00000000-0000-4000-8000-{k:012d}'


def test_cutoff_per_side_and_dedup():
    c=dict(same=torch.ones((1,2,2),dtype=torch.bool),ph=torch.tensor([[.6,.3]]),pv=torch.tensor([[.6,.3]]))
    for key in ('u','x','v'):c[key]=torch.arange(4.).reshape(1,2,2)
    r=select(c,.5)
    assert r['count']==1 and r['candidate_pairs']==1
    assert np.isclose(r['probability'][0],.36)  # product below .5 is allowed
    assert select(c,.2)['candidate_pairs']==4
    assert select(c,.2)['count']==1
    assert select(c,.7)['count']==0


def test_per_fiber_sigma_changes_cutoff_not_geometry():
    h=torch.tensor([[2.,.1,0.,10.]],dtype=torch.float64)
    v=h.clone();zero=torch.zeros(1,dtype=torch.float64)
    widths=torch.ones(6,dtype=torch.float64)*100
    fixed=candidates(h,v,zero,zero,widths,.5,maximum_sigma=1.)
    narrow=candidates(h,v,zero,zero,widths,torch.tensor([.2]),
        sigma_v=torch.tensor([.2]),maximum_sigma=1.)
    assert select(fixed,.9)['count']==0
    assert select(narrow,.9)['count']==1
    torch.testing.assert_close(fixed['x'],narrow['x'])


def test_corrected_u_crosses_width_boundary_without_reclassifying():
    h=torch.tensor([[0.,.9,0.,10.]],dtype=torch.float64)
    v=torch.tensor([[1.,.05,0.,10.]],dtype=torch.float64)
    widths=torch.tensor([100.,200.,300.],dtype=torch.float64)
    raw=candidates(h,v,torch.zeros(1),torch.zeros(1),widths,.01)
    corrected=candidates(h,v,torch.zeros(1),torch.zeros(1),widths,.01,
        torch.tensor([20.]),torch.tensor([0.]))
    assert torch.equal(raw['same'],corrected['same'])
    a,b=select(raw,.5),select(corrected,.5)
    assert a['x'][0]==pytest.approx(20.)
    assert b['x'][0]==pytest.approx(0.,abs=1e-10)
    assert b['u'][0]==pytest.approx(0.,abs=1e-10)


def test_phase_alignment_gauge_and_voxels():
    h=torch.tensor([[2.,.9,10.,10.]],dtype=torch.float64)
    v=torch.tensor([[3.,-.09,14.,10.]],dtype=torch.float64)
    widths=torch.tensor([100.,200.,300.,400.,500.],dtype=torch.float64)
    zero=torch.zeros(1,dtype=torch.float64)
    c=candidates(h,v,zero,zero,widths,.1)
    r=select(c,.9)
    assert r['count']==1
    np.testing.assert_allclose(r['u'],.01)
    np.testing.assert_allclose(r['x'],3.)
    np.testing.assert_allclose(r['v'],4.)
    h[:,0]+=2;h[:,1]-=2;v[:,0]-=1;v[:,1]+=1
    other=select(candidates(h,v,zero,zero,widths,.1),.9)
    for key in ('u','x','v','probability'):np.testing.assert_allclose(r[key],other[key])


def test_same_index_different_phase_branch_and_disabled():
    h=torch.tensor([[2.,.9,10.,10.]],dtype=torch.float64)
    v=torch.tensor([[2.,-.09,14.,10.]],dtype=torch.float64)
    widths=torch.ones(6,dtype=torch.float64)*100
    zero=torch.zeros(1,dtype=torch.float64)
    assert select(candidates(h,v,zero,zero,widths,.1),.9)['count']==0
    v[:,1]=.91;v[:,3]=-10.
    assert select(candidates(h,v,zero,zero,widths,.1),.1)['count']==0


@pytest.mark.parametrize('filter_families', [False, True])
@pytest.mark.parametrize('offsets', [False, True])
@pytest.mark.parametrize('application', ['added_only', 'all_constraints'])
def test_saved_comparison_tracks_lost_events(tmp_path, filter_families, offsets, application):
    import json
    from lasagna.inference_provenance import sha256_file
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_implicit_match_report import run
    d=dict(family=np.array([0,1,0,1]),source_id=np.array([fid(k) for k in range(4)]),start=np.zeros(4),end=np.ones(4),
        i=np.array([0,2]),j=np.array([1,3]),ds_a=np.zeros(2),ds_b=np.zeros(2),
        sign=np.ones(2),distance=np.ones(2)*.5,parallel=np.zeros(2,dtype=bool))
    data=tmp_path/'tensors.npz';np.savez(data,**d)
    if filter_families:
        # Invalid perpendicular H/H event must not enter the report denominator.
        for key, value in dict(i=0,j=2,ds_a=0.,ds_b=0.,sign=1.,distance=.5,parallel=False).items():
            d[key]=np.append(d[key],value)
        np.savez(data,**d)
    roots=[]
    for idx in range(2):
        root=tmp_path/str(idx);root.mkdir();roots.append(str(root))
        state=np.array([[2.,.1,0.,10.],[2.,.2,10.,10.],[2.,.1,0.,10.],[2.,.2,10.,10.]])
        if idx:state[1,0]+=1;state[3,2]=5
        extra={}
        if offsets and idx:
            extra=dict(v_offsets=np.array([0.,-2.,0.,-2.]),u_offsets=np.array([0.,-2.,0.,-2.]),v_offset_a=d['i'],v_offset_b=d['j'])
        np.savez(root/'solution.npz',state=state,widths=np.ones(6)*100,**extra,
            **{k:d[k] for k in ('family','source_id','start','end')})
        (root/'summary.json').write_text(json.dumps(dict(model='implicit_discrete_winding_v1')))
        (root/'config.json').write_text(json.dumps(dict(tensor_cache=str(data),tensor_sha256=sha256_file(data),
            winding_sigma=.1,same_sheet_match_multiplier=10 if idx else 1,
            filter_constraint_families=filter_families,v_offset_match_weight=2,u_offset_match_weight=2,
            offset_application=application)))
    result=run(dict(solve_roots=roots,output_root=str(tmp_path/'out'),cutoffs=[.5],compare_first=True))
    comp=result[1]['rows'][0]['common_with_reference']
    assert comp['count']==1 and comp['lost']==1 and comp['gained']==0
    assert comp['reference_v']['l1']==10 and comp['current_v']['l1']==(3 if offsets else 5)
    if offsets:
        assert result[1]['rows'][0]['uncorrected_v_base_voxels']['l1']==5
        assert 'v_offset_match' in result[1]['rows'][0]['losses']
        assert result[1]['rows'][0]['u_base_voxels']['l1']==pytest.approx(8.)
        assert result[1]['rows'][0]['uncorrected_u_base_voxels']['l1']==pytest.approx(10.)
    correction=.8**2 if offsets and application=='all_constraints' else 1.
    assert np.isclose(result[1]['rows'][0]['losses']['u']['selected_mean'],
                      10*correction*result[0]['rows'][0]['losses']['u']['selected_mean'])
    assert np.isclose(result[1]['rows'][0]['losses']['u']['full_objective_contribution'],
                      result[1]['rows'][0]['losses']['u']['selected_mean']/2)


@pytest.mark.parametrize('mapping', [False, True])
def test_u_offsets_preserve_shared_event_geometry(tmp_path, monkeypatch, mapping):
    import json
    from lasagna.inference_provenance import sha256_file
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_shared_curve as curve
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_implicit_match_report import run
    d=dict(family=np.array([0,1]),source_id=np.array([fid(0),fid(1)]),start=np.zeros(2),end=np.ones(2),
        i=np.array([0]),j=np.array([1]),ds_a=np.zeros(1),ds_b=np.zeros(1),parallel=np.array([False]))
    data=tmp_path/'tensors.npz';np.savez(data,**d)
    root=tmp_path/'solve';root.mkdir()
    state=np.array([[2.,.1,0.,10.],[2.,.9,10.,10.]])
    np.savez(root/'solution.npz',state=state,widths=np.ones(6)*100,
        u_offsets=np.array([0.,-2.]),v_offsets=np.array([0.,3.]),
        v_offset_a=d['i'],v_offset_b=d['j'],
        **{k:d[k] for k in ('family','source_id','start','end')})
    def saved_points(d,s,state,widths,ids,ds,as_mapping=False):
        prepared=state[ids].copy()
        prepared[:,1]=np.array([.3,.4])[ids];prepared[:,2]=np.array([50.,80.])[ids]
        if as_mapping:return prepared,np.zeros_like(ds)
        return prepared[:,0]+prepared[:,1],prepared[:,1],prepared[:,2]
    monkeypatch.setattr(curve,'saved_points',saved_points)
    (root/'summary.json').write_text(json.dumps(dict(model='implicit_discrete_winding_v1')))
    (root/'config.json').write_text(json.dumps(dict(tensor_cache=str(data),tensor_sha256=sha256_file(data),
        winding_sigma=.1,shared_curve_geometry=True,shared_curve_hypothesis_mapping=mapping,
        u_offset_match_weight=2,v_offset_match_weight=2)))
    result=run(dict(solve_roots=[str(root)],output_root=str(tmp_path/'out'),cutoffs=[.5]))[0]['rows'][0]
    assert result['count']==1
    assert result['u_base_voxels']['l1']==pytest.approx(8.)
    assert result['v_base_voxels']['l1']==pytest.approx(33.)
    assert result['uncorrected_u_base_voxels']['l1']==pytest.approx(10.)
    assert result['uncorrected_v_base_voxels']['l1']==pytest.approx(30.)
