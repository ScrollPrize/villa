"""Production H/V and sign preparation has no filesystem tensor handoff."""
import json
import math
import os
from pathlib import Path
import numpy as np
import pytest

from .fiber_graph_fixture import measured_store,uid


def test_direct_stage_preparation_returns_arrays_without_tensor_files(tmp_path,monkeypatch):
    from vc_object_store import CollectionReader
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_hv_stage as hv,fiber_normal_sign as normal
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph,original_fiber_source
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fibers_digest
    source=measured_store(tmp_path,[(0,10,20,False,.5),(1,10,30,True,1.)])
    graph=FiberGraph(source)
    with CollectionReader(str(source)) as reader:pin=reader.source
    _,original=original_fiber_source(source)
    features=tmp_path/'input_features.json';prior=tmp_path/'input_priors.json'
    features.write_text(json.dumps(dict(format='fiber-axial-priors',version=1,source_ids=graph.ids.tolist(),
        agreement=[-1,1,0],fiber_source=original,fibers_digest=fibers_digest(source),umbilicus_sha256='test')))
    prior.write_text(json.dumps(dict(format='fiber-outward-priors',source_store=pin,
        source_ids=graph.ids.tolist(),agreement=[1,1,1])))
    def forbidden(*args,**kwargs):raise AssertionError('Direct stages must not serialize or load tensor files')
    monkeypatch.setattr(np,'savez',forbidden);monkeypatch.setattr(np,'savez_compressed',forbidden)
    monkeypatch.setattr(np,'load',forbidden)
    config=dict(database=str(source),features=str(features),priors=str(prior),output_root=str(tmp_path/'no-work-files'))
    data,stats=hv.prepare(config,graph=graph,return_arrays=True)
    np.testing.assert_array_equal(data['i'],graph.i);np.testing.assert_array_equal(data['j'],graph.j)
    np.testing.assert_array_equal(data['relation'],np.where(graph.parallel,1.,-1.))
    np.testing.assert_array_equal(data['degree'],np.bincount(np.r_[graph.i,graph.j],minlength=3))
    assert stats['active_events']==2
    signs,stats=normal.prepare(config,graph=graph,return_arrays=True)
    np.testing.assert_array_equal(signs['i'],[0]);np.testing.assert_array_equal(signs['j'],[1])
    np.testing.assert_array_equal(signs['dot'],[1.])
    assert not np.shares_memory(signs['lengths'],graph.lengths)
    assert stats['tensor_bytes']==0 and not (tmp_path/'no-work-files').exists()


def test_inprocess_axial_evidence_shares_math_without_reference_audits(tmp_path,monkeypatch):
    from dataclasses import replace
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_hv_stage as hv
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph
    source=measured_store(tmp_path,[(0,10,20,False,.5),(1,10,30,True,1.)])
    graph=FiberGraph(source);ids=graph.ids
    evidence=hv.AxialEvidence(graph.source,ids,np.asarray([-1,1,0],np.float32),'axis')
    config=dict(database=str(source),output_root=str(tmp_path/'no-files'))
    def forbidden(*args,**kwargs):raise AssertionError('In-process evidence must not audit or stage full corpora')
    monkeypatch.setattr(hv,'fibers_digest',forbidden)
    monkeypatch.setattr(hv,'verify_feature_source',forbidden)
    monkeypatch.setattr(np,'load',forbidden);monkeypatch.setattr(np,'savez',forbidden)
    data,stats=hv.prepare_evidence(config,evidence,graph=graph)
    np.testing.assert_array_equal(data['a'],evidence.agreement)
    np.testing.assert_array_equal(data['relation'],[-1,1])
    np.testing.assert_array_equal(data['degree'],[2,1,1])
    assert data['a'].dtype==np.float32 and data['relation'].dtype==np.float32
    assert stats['active_fibers']==3 and stats['active_events']==2
    assert stats['feature_identity']==dict(source=graph.source,umbilicus_sha256='axis')
    with pytest.raises(ValueError,match='snapshot'):
        hv.prepare_evidence(config,replace(evidence,source={}),graph=graph)
    with pytest.raises(ValueError,match='IDs'):
        hv.prepare_evidence(config,replace(evidence,ids=ids[::-1]),graph=graph)
    with pytest.raises(TypeError,match='AxialEvidence'):
        hv.prepare_evidence(config,{},graph=graph)
    assert not (tmp_path/'no-files').exists()


def _centers(control,n):
    """Independent transcription of the pre-extraction VC float32 centers."""
    points=np.asarray(control,np.float32);points=points[np.argsort(points[:,2])]
    out=[]
    for z in range(n):
        if z<=points[0,2]:p=points[0].copy()
        elif z>=points[-1,2]:p=points[-1].copy()
        else:
            at=np.searchsorted(points[:,2],np.float32(z),side='left')
            if abs(points[at,2]-np.float32(z))<np.float32(1e-5):p=points[at].copy()
            else:
                a,b=points[at-1],points[at];t=(z-float(a[2]))/(float(b[2])-float(a[2]))
                p=np.array([float(a[0])+t*float(np.float32(b[0]-a[0])),
                            float(a[1])+t*float(np.float32(b[1]-a[1])),z],np.float32)
        p[2]=z;out.append(p)
    return np.asarray(out)


def _old_axial(points,centers,spacing):
    total=0.;weight=0.
    for a,b in zip(points[:-1],points[1:]):
        delta=b-a;length=float(np.sqrt(delta[0]*delta[0]+delta[1]*delta[1]+delta[2]*delta[2]))
        if not length:continue
        tangent=delta/length;count=max(1,math.ceil(length/spacing))
        for j in range(count):
            p=a+delta*((j+.5)/count);z=min(max(int(math.floor(p[2]+.5)),0),len(centers)-1)
            axis=(centers[min(len(centers)-1,z+8)]-centers[max(0,z-8)]).astype(np.float64)
            axis/=np.sqrt(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2])
            dot=np.clip(tangent@axis,-1,1);total+=(length/count)*(2*dot*dot-1.)
        weight+=length
    return np.float32(total/weight)


@pytest.mark.parametrize('workers',[1,4])
def test_native_axial_priors_match_old_quadrature_float32_centers(tmp_path,workers):
    from vc_object_store import fiber_axial_prior_columns
    control=[[20.,30.,0.],[25.1234567,34.,17.],[45.,80.2345678,50.],[50.,90.,79.]]
    path=tmp_path/'axis.json'
    path.write_text(json.dumps(dict(metadata=dict(volume=uid(999)+'-scan',volume_width=200,volume_height=200,
        volume_slices=80,voxelsize_um=2.),control_points=[dict(x=x,y=y,z=z) for x,y,z in control])))
    curves=[np.array([[35.,40.,8.499999],[70.,42.,8.500001],[80.,55.,23.]]),
            np.array([[30.,40.,7.499999],[32.,41.,7.500001],[45.,65.,60.]])]
    curves.append(curves[0][::-1].copy())
    offsets=np.r_[0,np.cumsum([len(p) for p in curves])]
    actual=fiber_axial_prior_columns(offsets,np.concatenate(curves),str(path),uid(999),None,3.25,workers)
    expected=np.array([_old_axial(p,_centers(control,80),3.25) for p in curves])
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1.2e-7)
    assert actual.dtype==np.float32 and not actual.flags.writeable
    scaled=fiber_axial_prior_columns(offsets,np.concatenate(curves),str(path),uid(999),[160,400,400],3.25,workers)
    scaled_expected=np.array([_old_axial(p,_centers(np.asarray(control,np.float32)*2,160),3.25) for p in curves])
    np.testing.assert_allclose(scaled,scaled_expected,rtol=0,atol=1.2e-7)


@pytest.mark.cuda
@pytest.mark.skipif(not os.environ.get('VILLA_GPU_MANAGER_GRANT'),reason='CUDA smoke requires managed GPU')
@pytest.mark.parametrize('stage',['classification','normal_sign'])
def test_wrapped_direct_solve_has_no_npz_handoff(tmp_path,monkeypatch,stage):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_wrapped_stages import solve_stage
    def forbidden(*args,**kwargs):raise AssertionError('Direct GPU solve must not load or save solver tensors')
    monkeypatch.setattr(np,'load',forbidden);monkeypatch.setattr(np,'savez',forbidden)
    data=dict(ids=np.array([uid(1),uid(2)]),i=np.array([0]),j=np.array([1]),lengths=np.array([10.,10.]),
        a=np.array([-1.,1.],np.float32),prior=np.array([1.,1.],np.float32),relation=np.array([-1.],np.float32),
        dot=np.array([1.],np.float32),degree=np.ones(2,np.float32),active=np.arange(2),inherited=np.zeros(2,bool),keep=np.ones(2,np.float32))
    stats=dict(all_fibers=2,inherited_broken=0,original_parallel=0,original_perpendicular=1)
    config=dict(output_root=str(tmp_path/'solve'),write_result_store=False,cost_modes=['uniform'],broken_penalties=[.1],
        entropy_weights=[0.],irls_iterations=3,irls_cg_iterations=3,fixed_keep=stage=='normal_sign')
    summary,saved=solve_stage(config,stage,data=data,stats=stats,return_solution=True)
    assert saved['ids'].tolist()==data['ids'].tolist() and len(summary['runs'])==1
    assert not list((tmp_path/'solve').glob('*.npz'))
    assert (tmp_path/'solve'/'solver-summary.json').exists() and not (tmp_path/'solve'/'summary.json').exists()
