"""The sole CT-normal crop convention, coordinate labels and held failure views."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest

from test_heading import sheet, CTOnlyVolume
from test_trace_heading import record
from vesuvius.neural_tracing.fiber_follow.shared.heading import (
    ct_frame, normal_frame, ct_normal, orient_item, FRAME_POLICY, SeedHeadingError,
)
from vesuvius.neural_tracing.fiber_follow.shared.geometry import frame_from_heading, arclength
from vesuvius.neural_tracing.fiber_follow.shared.data import TracedFiber, SampleConfig, make_sample, continuation_targets


def test_ct_frame_estimates_the_normal_and_keeps_heading():
    image,normal=sheet()
    vol=CTOnlyVolume(image)
    heading=np.array([.1,0.,1.]);heading/=np.linalg.norm(heading)
    frame=ct_frame(vol,[24.,24.,24.],heading)
    projected=normal-(normal@heading)*heading;projected/=np.linalg.norm(projected)
    np.testing.assert_allclose(frame[:,2],heading,atol=1e-12)
    assert abs(frame[:,0]@projected)>.99999
    np.testing.assert_allclose(frame.T@frame,np.eye(3),atol=1e-12)
    assert np.linalg.det(frame)==pytest.approx(1.)


def test_eigenvector_sign_flips_cannot_flip_a_trace_frame():
    h=np.array([0.,0.,1.]);previous=None
    for angle in np.linspace(0,2*np.pi,101):
        normal=np.array([np.cos(angle),np.sin(angle),0.])
        frame=normal_frame(h,normal,previous)
        np.testing.assert_allclose(frame,normal_frame(h,-normal,previous),atol=1e-12)
        if previous is not None:assert frame[:,0]@previous[:,0]>.99
        previous=frame


def test_unusable_ct_retains_only_an_already_established_roll(monkeypatch):
    h=np.array([0.,0.,1.]);previous=normal_frame(h,np.array([1.,2.,0.]))
    def unusable(vol,pos):raise SeedHeadingError('no identifiable sheet normal')
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.ct_normal',unusable)
    np.testing.assert_allclose(ct_frame(None,[0]*3,h,previous),previous,atol=1e-12)
    with pytest.raises(SeedHeadingError):ct_frame(None,[0]*3,h)
    np.testing.assert_allclose(normal_frame(h,h,previous),previous,atol=1e-12)
    with pytest.raises(SeedHeadingError):normal_frame(h,h)


def test_training_rotation_preserves_world_history_candidates_and_supervision():
    image,_=sheet();vol=CTOnlyVolume(image)
    z=np.linspace(4,44,81)
    p=np.c_[24+np.sin(z/12),np.full(len(z),24.),z]
    fiber=TracedFiber('test',p,arclength(p),'V')
    cfg=SampleConfig(n_history=8,n_future=4,lateral_sigmas=(0.,),lateral_probs=(1.,),
        angle_sigmas_deg=(0.,),angle_probs=(1.,),history_drift=0.,history_jitter=0.,history_wobble=0.,no_history_prob=0.)
    item=make_sample(fiber,20.,False,cfg,np.random.default_rng(4))
    item['candidate_points']=np.stack([item['fut_local']]*4)
    before=deepcopy(item)
    orient_item(item,vol)
    assert item['frame_policy']==FRAME_POLICY
    for key in ('hist_local','gt_history','fut_local','end_local','candidate_points'):
        np.testing.assert_allclose(item[key]@item['frame'].T,before[key]@before['frame'].T,atol=2e-6)
    expected=continuation_targets(fiber,20.,False,item['pos'],item['frame'],cfg)
    for key in ('plane_ab','dense_ab','plane_mask','dense_mask','gt_history','fut_local','end_local'):
        np.testing.assert_allclose(item[key],expected[key],atol=2e-6)
    old_frame=item['frame'].copy();reads=len(vol.ct.reads)
    orient_item(item,vol)
    assert len(vol.ct.reads)==reads
    np.testing.assert_array_equal(item['frame'],old_frame)


def test_failed_decisions_never_reestimate_roll_but_accepted_short_steps_do(monkeypatch):
    calls=[]
    def changing_normal(vol,pos):
        calls.append(np.asarray(pos).copy())
        angle=.2*(len(calls)-1)
        return np.array([np.cos(angle),np.sin(angle),0.])
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.ct_normal',changing_normal)
    fail=([[0.,0.,1.],[0.,0.,2.]],[.1,.1])
    rows=record([fail]*3,explore_calls=3)
    assert len(calls)==1
    for row in rows:np.testing.assert_array_equal(row['frame'],rows[0]['frame'])
    calls.clear()
    accept=([[0.,0.,1.],[0.,0.,2.]],[.9,.1])
    rows=record([accept]*3)
    assert len(calls)==3
    for row in rows:np.testing.assert_array_equal(row['frame'][:,2],[0.,0.,1.])
    assert rows[0]['frame'][:,0]@rows[2]['frame'][:,0] < .95
