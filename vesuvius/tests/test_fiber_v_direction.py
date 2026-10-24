import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_direction import direction_prior,circulation_prior,family_settings,inherit_h_labels,label_metrics


def test_axial_up_and_down_including_on_axis():
    axis=np.array([[0.,0.,-100.],[0.,0.,100.]])
    for x in (0.,80.):
        points=np.array([[x,0.,-50.],[x,0.,50.]])
        np.testing.assert_allclose(direction_prior(points,axis,family='V'),(1.,1.),rtol=0,atol=1e-14)
        np.testing.assert_allclose(direction_prior(points[::-1],axis,family='V'),(-1.,1.),rtol=0,atol=1e-14)
    assert circulation_prior(np.array([[0.,0.,-50.],[0.,0.,50.]]),axis)==(0.,0.)
    assert direction_prior(np.zeros((2,3)),axis,family='V')==(0.,0.)


def test_axial_local_tilt_and_gauge():
    z=np.linspace(-200,200,21);axis=np.c_[.5*z,np.zeros_like(z),z]
    points=np.array([[-25.,10.,-50.],[25.,10.,50.]])
    up,support=direction_prior(points,axis,family='V')
    assert up>1-1e-14
    np.testing.assert_allclose(support,1.,rtol=0,atol=1e-14)
    np.testing.assert_allclose(direction_prior(points[::-1],axis,family='V')[0],-up,atol=1e-14)
    # Whole-fiber sign does not impose pointwise monotonic Z on a folded trace.
    folded=np.array([[0.,0.,0.],[0.,0.,50.],[0.,0.,25.]])
    straight=np.array([[0.,0.,-100.],[0.,0.,100.]])
    np.testing.assert_allclose(direction_prior(folded,straight,family='V')[0],1/3,atol=1e-14)


def test_families_and_h_prior_preserved():
    assert family_settings({})==('H',0,('ccw','cw'))
    assert family_settings({'family':'V'})==('V',1,('up','down'))
    with pytest.raises(ValueError):family_settings({'family':'unknown'})
    axis=np.array([[0.,0.,-100.],[0.,0.,100.]])
    p=np.array([[100.,0.,0.],[90.,30.,0.],[60.,70.,0.]])
    assert direction_prior(p,axis,family='H')==circulation_prior(p,axis)
    assert direction_prior(p,axis,family='h')==circulation_prior(p,axis)


def test_inherited_h_labels_are_immutable():
    ids=np.arange(5);labels=np.array([0,1,0,2,1])
    h=dict(ids=ids,labels=np.array([0,1,2,2,1]),newly_broken=np.array([0,0,1,0,0],bool),
           direction_sign_ccw=np.array([-1,0,0,0,0]),direction_sign_cw=np.array([1,0,0,0,0]))
    np.testing.assert_array_equal(inherit_h_labels(ids,labels,h),h['labels'])
    np.testing.assert_array_equal(labels,[0,1,0,2,1])
    for wrong in [dict(h,ids=ids+1),dict(h,labels=labels),
                  dict(h,newly_broken=np.array([0,1,0,0,0],bool)),dict(h,direction_sign_cw=-h['direction_sign_cw'])]:
        with pytest.raises(ValueError):inherit_h_labels(ids,labels,wrong)


def test_axial_metric_names():
    result=label_metrics(dict(all_original_h_sign_diagnostics=dict(circulation_agreement=.5,length_weighted_circulation_agreement=.6)),'V')
    assert result=={'all_original_v_sign_diagnostics':{'axial_agreement':.5,'length_weighted_axial_agreement':.6}}
