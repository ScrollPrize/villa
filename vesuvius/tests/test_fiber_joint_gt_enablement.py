import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_joint_gt_eval import configure_gt_enablement


def U(n):
    return f'00000000-0000-4000-8000-{n:012d}'


def test_gt_selectable_removes_all_force_flags_and_saturated_logits():
    cfg=dict(force_keep_fibers=True,force_keep_source_ids=[U(20)],rejection_cost=2.5)
    state=np.array([[1.,2.,3.,4.],[5.,6.,7.,100.]])
    configure_gt_enablement(cfg,state,np.array([U(10),U(20)]),[U(20)],False)
    np.testing.assert_array_equal(state,[[1,2,3,4],[5,6,7,4]])
    assert cfg==dict(rejection_cost=2.5)


def test_gt_forced_remains_default_and_does_not_force_ordinary_fibers():
    cfg={};state=np.zeros((2,4))
    configure_gt_enablement(cfg,state,np.array([U(10),U(20)]),[U(20)])
    assert cfg['force_keep_source_ids']==[U(20)]
    np.testing.assert_array_equal(state[:,3],[0,100])


def test_combine_keeps_gt_uuids_and_rejects_overlap(monkeypatch):
    import pytest
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_joint_gt_eval as joint
    def graph(source):
        n=len(source)
        return dict(source_id=np.array(source),family=np.zeros(n,int),initial_k=np.zeros(n),start=np.zeros(n),
            end=np.full(n,10.),phase=np.zeros(n),z=np.zeros(n),radius=np.full(n,100.),lengths=np.full(n,10.),
            node_ds=np.tile([0.,5.,10.],(n,1)),node_phase=np.zeros((n,3)),node_z=np.zeros((n,3)),
            earlier_labels=np.zeros(n,int),width_initial=np.ones(3),links=np.empty((0,4)),
            **{k:np.zeros(0,dtype=bool if k=='parallel' else float) for k in ('ds_a','ds_b','phase_a','phase_b','phase_delta','distance','sign','parallel')},
            i=np.zeros(0,int),j=np.zeros(0,int))
    monkeypatch.setattr(joint.uv,'radial_initial_geometry',lambda d,c:(np.zeros(len(d['family'])),np.ones(3)))
    d,gd=graph([U(10)]),graph([U(20)])
    events=[dict(test_piece=0,gt_piece=0,gt_source=U(20),signed_distance=1.,test_arc=1.,gt_arc=2.,
                 test_angle=0.,gt_angle=0.,phase_delta=0.,parallel=True)]
    out,ids,_=joint.combine(d,gd,events,{})
    assert out['source_id'].tolist()==[U(10),U(20)] and ids.tolist()==[U(20)]
    assert out['i'].tolist()==[0] and out['j'].tolist()==[1]
    with pytest.raises(ValueError,match='disjoint'):
        joint.combine(graph([U(20)]),gd,events,{})
