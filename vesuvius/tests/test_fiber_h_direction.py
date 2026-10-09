import numpy as np
import torch
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength,tangents_at,tangent_at,interp_at,normalize
from vesuvius.neural_tracing.fiber_trace_3d.fiber_normal_sign import losses,sign_alignment_terms
from vesuvius.neural_tracing.fiber_trace_3d.fiber_direction import circulation_prior,objective,readout,breakage_objective,breakage_readout


def test_shared_tangent_batch_matches_original_scalar():
    rng=np.random.default_rng(0);p=np.cumsum(rng.normal(size=(30,3)),axis=0);s=arclength(p)
    positions=np.r_[0,np.linspace(0,s[-1],50),s[-1]];half=3.
    expected=[]
    for t in positions:
        a=interp_at(p,s,np.array([max(t-half,0.)]))[0]
        b=interp_at(p,s,np.array([min(t+half,s[-1])]))[0]
        expected.append(normalize(b-a))
    np.testing.assert_array_equal(tangents_at(p,s,positions,half),expected)
    for t,e in zip(positions,expected):np.testing.assert_array_equal(tangent_at(p,s,t,half),e)
    np.testing.assert_allclose(tangents_at(p[::-1],s[-1]-s[::-1],s[-1]-positions,half),-np.asarray(expected),atol=1e-13)


def test_prior_rotation_reversal_and_neutral_axis():
    theta=np.linspace(0,1.5,100);p=np.c_[100*np.cos(theta),100*np.sin(theta),np.zeros(100)]
    axis=np.array([[0.,0.,-100.],[0.,0.,100.]])
    prior,support=circulation_prior(p,axis)
    assert prior>.999 and support==1
    reversed_prior,reversed_support=circulation_prior(p[::-1],axis)
    np.testing.assert_allclose(reversed_prior,-prior,atol=1e-14)
    assert reversed_support==1
    assert circulation_prior(np.array([[0.,0.,0.],[0.,0.,50.]]),axis)==(0.,0.)


def test_shared_normal_loss_unchanged():
    torch.manual_seed(2);u=torch.randn(6,dtype=torch.double,requires_grad=True)
    b=torch.randn(6,dtype=torch.double);a=torch.rand(6,dtype=torch.double)*2-1
    i=torch.tensor([0,1,2,3,4]);j=i+1;dot=torch.tensor([.8,-.9,.6,-.7,.9],dtype=torch.double)
    s=2*u.sigmoid()-1;q=b.sigmoid()
    pair=((1-q[i])*(1-q[j])*(dot.abs()-dot*s[i]*s[j])*.5).sum()/dot.abs().sum()
    prior=((1-a*s)*.5).mean();expected=pair+.001*prior+.1*q.mean()
    actual=losses(u,b,a,i,j,dot,.001,.1)
    torch.testing.assert_close(actual[0],expected,rtol=0,atol=0)
    assert torch.autograd.gradcheck(lambda x:losses(x,b,a,i,j,dot,.001,.1)[0],(u,))


def test_circulation_gauge_and_no_pair_weight():
    u=torch.tensor([1.,-2.,.5],requires_grad=True);a=torch.tensor([.5,-.2,.3]);i=torch.tensor([0,1]);j=torch.tensor([1,2]);dot=torch.tensor([-.9,-.8])
    terms=objective(u,a,i,j,dot,.001,.01)
    opposite=objective(-u,-a,i,j,dot,.001,.01)
    torch.testing.assert_close(terms[0],opposite[0])
    empty=torch.empty(0,dtype=torch.long)
    pair,prior=sign_alignment_terms(u,a,empty,empty,torch.empty(0))
    assert pair==0 and torch.isfinite(prior)


def test_connected_sign_fit_overcomes_wrong_local_prior():
    # Stored source directions alternate; a weak incorrect unary on fiber2 must
    # not overwhelm the parallel relationships to both of its neighbors.
    target=torch.tensor([1.,-1.,1.,-1.]);a=torch.tensor([.8,-.8,-.1,-.8])
    i=torch.tensor([0,1,2]);j=i+1;dot=target[i]*target[j]
    u=torch.nn.Parameter(2*a.clone());opt=torch.optim.Adam([u],lr=.1)
    for _ in range(150):
        opt.zero_grad();objective(u,a,i,j,dot,.001,.001)[0].backward();opt.step()
    torch.testing.assert_close(torch.where(u>=0,1.,-1.),target)
    metric=readout(u.detach().numpy(),a.numpy(),i.numpy(),j.numpy(),dot.numpy(),np.zeros(4,dtype=int),np.ones(4))
    assert metric['disagreeing_events']==0


def test_degenerate_tangent_is_not_a_direction():
    p=np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,0.]])
    np.testing.assert_array_equal(tangents_at(p,arclength(p),np.array([1.]),half=1.),[[0,0,0]])


def test_breakage_objective_gating_prior_and_gauge():
    u=torch.tensor([3.,-2.,1.],dtype=torch.double);a=torch.tensor([.7,.3,-.1],dtype=torch.double)
    i=torch.tensor([0,1]);j=torch.tensor([1,2]);dot=torch.tensor([.9,-.8],dtype=torch.double)
    b=torch.full_like(u,-100.)
    old=objective(u,a,i,j,dot,.001,.001)
    new=breakage_objective(u,b,a,i,j,dot,.001,.001,.1)
    torch.testing.assert_close(old[0],new[0],atol=1e-12,rtol=1e-12)
    b=torch.tensor([-100.,100.,-100.],dtype=torch.double)
    new=breakage_objective(u,b,a,i,j,dot,.001,.001,.1)
    assert abs(float(new[1]))<1e-40
    expected=((1-a*(2*u.sigmoid()-1))*.5*torch.tensor([1.,0.,1.])).mean()
    torch.testing.assert_close(new[2],expected)
    torch.testing.assert_close(new[0],breakage_objective(-u,b,-a,i,j,dot,.001,.001,.1)[0])


def test_breakage_conflicting_pair_and_isolated_survivor():
    # Two incompatible observations for one pair; one endpoint must be rejected.
    i=torch.tensor([0,0]);j=torch.tensor([1,1]);dot=torch.tensor([1.,-1.])
    a=torch.tensor([.9,.6,0.]);u=torch.nn.Parameter(torch.tensor([3.,2.,.001]))
    b=torch.nn.Parameter(torch.full_like(a,-4.));opt=torch.optim.Adam([u,b],lr=.1)
    for step in range(1000):
        opt.zero_grad();breakage_objective(u,b,a,i,j,dot,.001,.001,.03)[0].backward();opt.step()
    stats,p,rejected=breakage_readout(u.detach().numpy(),b.detach().numpy(),a.numpy(),i.numpy(),j.numpy(),dot.numpy(),np.array([0,0,1]),np.ones(3))
    assert stats['disagreeing_events']==0 and stats['new_rejections']>=1
    assert not rejected[2]  # an unconstrained neutral fiber is NOT discarded
    assert stats['retained_events']==0 and stats['disagreement_percent'] is None
    np.testing.assert_allclose(p.sum(1),1,atol=1e-7)


def test_breakage_metrics_denominators():
    args=(np.array([2.,2.,-2.]),np.array([-100.,-100.,100.]),np.zeros(3),
          np.array([0,1]),np.array([1,2]),np.array([-1.,1.]),np.zeros(3,int),np.array([1.,2.,3.]))
    stats,_,rejected=breakage_readout(*args)
    assert stats['retained_arc_percent']==50 and stats['retained_events_percent']==50
    assert stats['disagreeing_events']==1 and stats['disagreement_percent']==100
    assert stats['disagreement_original_percent']==50
    assert stats['weighted_disagreement_original_percent']==50
