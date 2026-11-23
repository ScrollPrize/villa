import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d import fiber_wrapped_irls as w


def graph(device='cpu'):
    f=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    return dict(phase=f([.12,.87,.15,.85]),keep=f([.99]*4),
                i=torch.tensor([0,1,2,0,0],device=device),j=torch.tensor([1,2,3,3,2],device=device),
                target=f([1,1,1,1,0]),weight=f([.2]*5),prior_target=f([0,1,0,1]),
                prior_weight=f([.01]*4),prior_constant=f([0]*4),rejection=f([.25]*4))


def test_periodic_cost_boundaries_and_no_zero_collapse():
    x=torch.tensor([-.000001,.000001],dtype=torch.float64)
    torch.testing.assert_close(w.l1_cost(w.residual(x,1)),torch.full_like(x,.999999))
    for shift in [-4,-2,0,2,4]:
        torch.testing.assert_close(w.l1_cost(w.residual(x+shift,1)),w.l1_cost(w.residual(x,1)),atol=1e-14,rtol=0)
    assert w.l1_cost(w.residual(torch.tensor(0.),1))==1
    assert w.l1_cost(w.residual(torch.tensor(1.),1))==0


def test_gradient_away_from_corners():
    x=torch.tensor([-.72,-.02,.03,.45,.95],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda t:w.l1_cost(w.residual(t,1),.05).sum(),(x,))


def test_quadratic_matches_dense_including_self_edges_and_isolated():
    d=graph();d['i']=torch.cat((d['i'],torch.tensor([3])));d['j']=torch.cat((d['j'],torch.tensor([3])))
    c=torch.tensor([.1,.4,.2,.3,.2,.9],dtype=torch.float64)
    target=torch.tensor([1.,-1.,1.,1.,0.,1.],dtype=torch.float64)
    x,used=w.quadratic_step(d['phase'],d['i'],d['j'],target,c,d['prior_target'],d['prior_weight'],max_cg=20,cg_tolerance=1e-10)
    b=torch.zeros(6,4,dtype=torch.float64)
    b.index_put_((torch.arange(6),d['i']),-torch.ones(6,dtype=torch.float64),accumulate=True)
    b.index_put_((torch.arange(6),d['j']),torch.ones(6,dtype=torch.float64),accumulate=True)
    dense=torch.linalg.solve(b.T@(c[:,None]*b)+torch.diag(d['prior_weight']),b.T@(c*target)+d['prior_weight']*d['prior_target'])
    torch.testing.assert_close(x,dense,atol=1e-10,rtol=1e-10)
    assert used<=20
    z=torch.zeros(2,dtype=torch.float64);empty=torch.empty(0,dtype=torch.long)
    same,_=w.quadratic_step(z+3,empty,empty,z[:0],z[:0],z,z,max_cg=10)
    torch.testing.assert_close(same,z+3)


@pytest.mark.parametrize('mask_prior',[False,True])
def test_keep_majorizer_never_increases_objective(mask_prior):
    torch.manual_seed(4)
    d=graph();d['phase']=torch.rand(4,dtype=torch.float64)*4
    d['rejection']*=.02
    kw=dict(mask_prior=mask_prior,epsilon=.05)
    old=w.objective(**d,**kw)[0]
    k=w.keep_step(**d,**kw)
    assert w.objective(**dict(d,keep=k),**kw)[0]<=old+1e-12
    assert ((k>=0)&(k<=1)).all()


def test_consistent_graph_recovers_binary_labels_and_descent():
    d=graph();r=w.solve(**d,iterations=50,max_cg=30,tolerance=1e-10)
    p=w.class_score(r['phase'])
    torch.testing.assert_close(p,d['prior_target'],atol=1e-5,rtol=0)
    assert (r['keep']>.999).all()
    assert np.all(np.diff([row['loss'] for row in r['history']])<=1e-12)
    assert r['history'][-1]['loss']<1e-9


def test_keep_is_per_fiber_and_can_reenable():
    d=graph();d['keep'].zero_()
    r=w.solve(**d,iterations=50,max_cg=30)
    assert (r['keep']>.99).all()
    # A wrong self-constraint cannot be repaired by phase movement.
    d=graph();d['i']=torch.tensor([0]);d['j']=torch.tensor([0]);d['target']=torch.ones(1,dtype=torch.float64)
    d['weight']=torch.ones(1,dtype=torch.float64);d['rejection']*=.01
    r=w.solve(**d,iterations=30,max_cg=30)
    assert r['keep'][0]<.5 and (r['keep'][1:]>.99).all()


def test_fixed_keep_preserves_upstream_classification():
    d=graph();d['keep']=torch.tensor([1.,0.,1.,1.],dtype=torch.float64)
    result=w.solve(**d,fixed_keep=True,iterations=20,max_cg=30)
    torch.testing.assert_close(result['keep'],d['keep'],atol=0,rtol=0)
    assert np.all(np.diff([row['loss'] for row in result['history']])<=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA requires managed host validation')
def test_cuda_matches_cpu():
    cpu=w.solve(**graph(),iterations=40,max_cg=30)
    gpu=w.solve(**graph('cuda'),iterations=40,max_cg=30)
    for key in ('phase','keep'):
        torch.testing.assert_close(cpu[key],gpu[key].cpu(),atol=1e-8,rtol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA requires managed host validation')
@pytest.mark.parametrize('stage',['normal_sign','classification'])
@pytest.mark.parametrize('backend',['adam','wrapped_irls'])
def test_cache_only_stage_keeps_arrays_without_store(tmp_path,stage,backend,monkeypatch):
    import json
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_normal_sign as normal, fiber_hv_stage as hv, fiber_wrapped_stages as wrapped
    for module in (normal,hv,wrapped):
        def unexpected(*args,**kwargs):
            raise AssertionError('cache-only benchmark must not import/write a store')
        monkeypatch.setattr(module,'write_results',unexpected)
    cache=tmp_path/'input.npz'
    np.savez(cache,ids=np.arange(4),prior=np.array([.8,-.8,.8,-.8],np.float32),
        a=np.array([-.8,.8,-.8,.8],np.float32),i=np.array([0,1,2,0]),j=np.array([1,2,3,2]),
        dot=np.array([-1,-1,-1,1],np.float32),relation=np.array([-1,-1,-1,1],np.float32),
        degree=np.array([2,2,3,1],np.float32),lengths=np.ones(4),active=np.arange(4),inherited=np.zeros(4,bool))
    cache.with_suffix('.json').write_text(json.dumps(dict(all_fibers=4,active_fibers=4,inherited_broken=0,original_parallel=1,original_perpendicular=3)))
    cfg=dict(tensor_cache=str(cache),output_root=str(tmp_path/'output'),optimizer=backend,write_result_store=False,
        steps=2,log_every=1,broken_penalties=[.1],cost_modes=['degree_mix'],entropy_weights=[0],irls_iterations=2,irls_cg_iterations=10)
    (normal if stage=='normal_sign' else hv).solve(cfg)
    paths=list((tmp_path/'output').glob('solution_*.npz'))
    assert len(paths)==1 and not list((tmp_path/'output').glob('*.vcobj'))
    with np.load(paths[0]) as saved:
        np.testing.assert_array_equal(saved['ids'],np.arange(4))


@pytest.mark.parametrize('bad_ids',[False,True])
def test_cached_benchmark_identity_and_unchanged_evidence(tmp_path,bad_ids):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_wrapped_benchmark import prepare_cached
    ids=np.arange(4);lengths=np.array([2.,3.,4.,5.]);active=np.array([0,2,3])
    normal=tmp_path/'normal.npz';hv=tmp_path/'hv.npz';catalog=tmp_path/'lengths.npz'
    np.savez(normal,ids=ids,i=np.array([0,1,2]))
    np.savez(hv,ids=ids,active=active,inherited=np.array([False,True,False,False]),lengths=lengths[active],i=np.array([0,1]))
    np.savez(catalog,ids=ids[::-1] if bad_ids else ids,lengths=lengths)
    before={p:p.read_bytes() for p in (normal,hv,catalog)}
    config=dict(normal_cache=str(normal),hv_cache=str(hv),length_catalog=str(catalog),output_root=str(tmp_path/'output'))
    if bad_ids:
        with pytest.raises(ValueError,match='identities differ'):
            prepare_cached(config)
        assert not (tmp_path/'output').exists()
    else:
        result=prepare_cached(config)
        assert result['fibers']==4 and result['inherited_broken']==1 and result['hv_events']==2
        assert result['total_length_base']==14 and result['import_wall_seconds']==0
    assert all(p.read_bytes()==data for p,data in before.items())
