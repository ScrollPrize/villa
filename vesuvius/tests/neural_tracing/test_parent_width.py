from dataclasses import replace
import json
import os
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from test_fiber_annotations import annotated_loader
from test_refinement import line
from vesuvius.neural_tracing.fiber_trace_3d.parent_width import assign_parent_widths, calibrated_width, width_metadata
from vesuvius.neural_tracing.fiber_trace_3d.refinement import (
    RefinementConfig, GeometryStore, RefinementObservation, WidthFeedback,
    BatchedPatchTransform, RefinementObserver, width_evidence, refine_observations,
)
from vesuvius.neural_tracing.fiber_trace_3d.annotations import width_similarity, AnnotationConfig
from vesuvius.neural_tracing.fiber_trace_3d.loader import _Augment3DParams


def test_parent_median_deduplicates_components_and_updates_all_labels(tmp_path):
    record = annotated_loader().records[0]
    records,rows = [],[]
    for i,width in enumerate([100.,100.,130.]):
        p = tmp_path/'train'/f'{i}.json'; p.parent.mkdir(exist_ok=True); p.write_text('{}')
        fiber = replace(record.fiber,path=p,metadata=width_metadata(record.fiber.metadata,width))
        records.append(replace(record,fiber=fiber))
        rows.append(dict(path=f'train/{i}.json',parent='/old/parent.json',parent_sha256='a'*64))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(files=rows)))
    result = assign_parent_widths([*records,*([records[-1]]*5)])
    assert {r.fiber.metadata['width'] for r in result} == {100.}
    assert len({r.fiber.metadata['parent_width_key'] for r in result}) == 1
    assert all(r.fiber.metadata['parent_width_baseline']==100 for r in result)
    assert records[-1].fiber.metadata['width']==130 # no in-place source mutation
    # A source piece with no retained eligible span still contributes one vote.
    only_one=assign_parent_widths([records[-1]],records)
    assert only_one[0].fiber.metadata['width']==100.
    (tmp_path/'manifest.json').write_text(json.dumps(dict(files=rows[:-1])))
    with pytest.raises(ValueError,match='missing parent'):
        assign_parent_widths(records)


def test_width_store_archive_export_frozen_and_tether(tmp_path):
    cfg = RefinementConfig(enabled=True,width_enabled=True,width_learning_rate=1.)
    store = GeometryStore(tmp_path/'state'); key='a'*64; parent='b'*64
    store.register(key,line()[:,:3],{'split':'train'},cfg)
    store.register_width(parent,100.,{'name':'parent'},'train',key)
    # Cap and immutable median tether, not a new tether to the last observation.
    delta,percent = store.update_width(parent,[200.],cfg)
    assert delta==percent==3.125
    assert calibrated_width(120.,100.,[120.],cfg)<120
    assert calibrated_width(120.,100.,[],cfg)==120
    with pytest.raises(ValueError,match='split'):
        store.register_width(parent,100.,{'name':'parent'},'test',key)
    path=tmp_path/'fibers.zip'; descriptor=store.save(path)
    with zipfile.ZipFile(path) as z:
        assert json.loads(z.read(f'{key}/fiber.json'))['width']==103.125
    restored=GeometryStore(tmp_path/'restored'); restored.restore(path,descriptor['sha256'])
    frozen=GeometryStore(restored.root,read_only=True)
    frozen.register_width(parent,100.,{'name':'parent'},'train',key)
    assert frozen.read_width(parent)==103.125
    with pytest.raises(RuntimeError,match='read-only'):
        frozen.update_width(parent,[100.],cfg)


def test_frozen_test_loader_reads_shared_current_width(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d.loader import FiberTrace3DLoader
    loader=annotated_loader()
    record=loader.records[0]
    record=replace(record,fiber=replace(record.fiber,path=tmp_path/'fiber.json'))
    shared=assign_parent_widths([record])[0].fiber.metadata
    store=GeometryStore(tmp_path/'store')
    store.register_width(shared['parent_width_key'],8.,shared['parent_fiber'],'test','a'*64)
    cfg=RefinementConfig(enabled=True,update_enabled=False)
    # A previously refined width, distinct from the immutable median.
    state_path=store.root/f"{shared['parent_width_key']}.width.json"
    state=json.loads(state_path.read_text()); state['current']=9.
    state_path.write_text(json.dumps(state))
    monkeypatch.setenv('VILLA_FIBER_REFINEMENT_STATE',str(store.root))
    monkeypatch.setenv('VILLA_FIBER_REFINEMENT_FROZEN','1')
    monkeypatch.setattr(FiberTrace3DLoader,'_load_records',lambda self:[record])
    frozen=FiberTrace3DLoader(replace(loader.config,refinement=cfg,refinement_split='test'))
    assert frozen._refinement_store is None
    sample=frozen.load_sample(0)
    widths=sample.annotation_targets[:,6]
    np.testing.assert_allclose(widths[widths>0],9.,atol=1e-5)


def observation(cfg=None,spacing=2.,matrix=None):
    c=line(); matrix=np.eye(3) if matrix is None else matrix
    params=_Augment3DParams(0,np.zeros(3),np.zeros(3),matrix,np.linalg.inv(matrix),0,1,1,0,0,None,0,0,None)
    return RefinementObservation('a'*64,0,c,c.copy(),np.tile([0,1,0],(len(c),1)),
        params,spacing,(64,)*3,cfg or RefinementConfig(enabled=True),1,True,
        parent_width_key='b'*64,width_base=100.)


def test_gate_scale_transverse_stretch_and_current_width():
    o=observation(matrix=np.diag([1.,1.,2.]))
    maps=[np.full((64,)*3,100.,dtype=np.float32)] # base100 /spacing2 *width-axis stretch2
    f=WidthFeedback([o],maps,BatchedPatchTransform([o]))
    p=o.current[2:3,:3]; ids=np.array([0]); arc=o.current[2:3,3]; t=np.array([[1.,0.,0.]])
    value,valid=f.sample(p,ids,arc,t)
    assert valid.all() and value[0]==pytest.approx(100.)
    assert f.gate(p,ids,arc,t)[0]==pytest.approx(1.)
    changed=replace(o,width_base=110.)
    f=WidthFeedback([changed],maps,BatchedPatchTransform([changed]))
    assert f.gate(p,ids,arc,t)[0]==pytest.approx(1-10/22)
    widths=np.array([80.,90.,100.,110.,120.,np.nan])
    np.testing.assert_allclose(width_similarity(widths,100.),[0,.5,1,.5,0,0],atol=1e-7)
    np.testing.assert_allclose(width_similarity(torch.tensor(widths),torch.tensor(100.)).numpy(),width_similarity(widths,100.))
    evidence=width_evidence([o],[o.current],[np.ones((64,)*3)],maps)[0]
    np.testing.assert_allclose(evidence,100.)
    assert len(evidence)>1
    assert width_evidence([o],[o.current],[np.zeros((64,)*3)],maps)[0].size==0


def test_smooth_feedback_uses_label_jacobian():
    from vesuvius.neural_tracing.fiber_trace_3d.annotations import transverse_stretch
    loader=annotated_loader(True); record=loader.records[0]
    params=loader._sample_augment_params(record,0,71)
    o=observation(spacing=record.volume_spacing_base)
    o=replace(o,params=params,shape=loader.config.patch_shape_zyx)
    point=np.array([params.cp_volume_zyx])*o.spacing
    tangent=np.array([[1.,0.,0.]]); normal=np.array([[0.,1.,0.]])
    jac=loader._source_to_output_jacobian(point/o.spacing,params)
    stretch=transverse_stretch(jac,tangent,normal,np.array([True]))[0]
    f=WidthFeedback([o],[np.full(o.shape,10.)],BatchedPatchTransform([o]))
    result,valid=f.sample(point,np.array([0]),np.array([o.current[0,3]]),tangent)
    assert valid.all()
    assert result[0]==pytest.approx(10*o.spacing/stretch,rel=1e-6)


def test_shared_width_archive_missing_parent_fails(tmp_path):
    store=GeometryStore(tmp_path/'store'); parent='a'*64
    store.register_width(parent,100.,{'name':'parent'},'train','b'*64)
    archive=tmp_path/'state.zip'; saved=store.save(archive)
    restored=GeometryStore(tmp_path/'restored'); restored.restore(archive,saved['sha256'])
    with pytest.raises(ValueError,match='missing a configured parent width'):
        restored.register_width('c'*64,100.,{'name':'other'},'train','d'*64)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    os.environ.get('FIBER_COOPERATIVE_CUDA_TEST')!='1',reason='manager-gated CUDA only'))])
def test_width_observer_updates_only_accepted_parent_and_detaches(tmp_path,monkeypatch,device):
    cfg=RefinementConfig(enabled=True,width_enabled=True,width_modulation=True,gt_weight=0)
    o=observation(cfg)
    store=GeometryStore(tmp_path/'store'); monkeypatch.setenv('VILLA_FIBER_REFINEMENT_STATE',str(store.root))
    store.register(o.key,o.current[:,:3],{},cfg)
    c=store.read(o.key); o=replace(o,current=c,baseline=store.baseline(o.key),normals=np.tile([0,1,0],(len(c),1)))
    store.register_width(o.parent_width_key,100.,{'name':'parent'},'train',o.key)
    second=replace(o,key='c'*64)
    store.register(second.key,second.current[:,:3],{},cfg)
    store.register_width(o.parent_width_key,100.,{'name':'parent'},'train',second.key)
    stale_curve=o.current.copy(); stale_curve[:,1]+=.25
    stale=replace(o,current=stale_curve)
    observer=RefinementObserver(cfg)
    batch=SimpleNamespace(stream_indices=torch.tensor([0,1,2],device=device),refinement_geometry=(o,second,stale))
    output=torch.zeros(3,7,64,64,64,device=device); output[:,6]=1.
    auxiliary=torch.tensor([55.,60.,80.],device=device)[:,None,None,None,None].expand(3,1,64,64,64).clone().requires_grad_()
    observer.observe_annotations(batch,output,auxiliary,AnnotationConfig(width_weight=.01))
    assert not observer.pending[0][-1].requires_grad
    auxiliary.square().mean().backward()
    assert torch.isfinite(auxiliary.grad).all()
    # Unequal populations: pooled median110, median of patch medians115.
    # Stale proposal's many large values must not influence the parent estimate.
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_trace_3d.refinement.width_evidence',
        lambda *args: [np.array([110.,110.,110.]),np.array([120.]),np.full(100,900.)])
    stats=observer.flush(SimpleNamespace(enabled=False,is_main=True))
    expected=calibrated_width(100.,100.,[110.],cfg)
    assert observer.accepted==2 and observer.stale==1
    assert store.read_width(o.parent_width_key)==pytest.approx(expected)
    assert stats[-2]==pytest.approx(expected-100.)
    assert o.width_base==100. # already queued observations are immutable


def test_median_width_outliers_even_population_and_rate():
    cfg=RefinementConfig(width_gt_weight=4.,width_learning_rate=.01)
    assert calibrated_width(100.,100.,[110.,110.,1000.],cfg)==pytest.approx(100.02)
    assert calibrated_width(100.,100.,[110.,120.],cfg)==pytest.approx(100.03)
    assert calibrated_width(100.,100.,[],cfg)==100.
    for invalid in ([np.nan],[0.],[-1.]):
        with pytest.raises(ValueError):
            calibrated_width(100.,100.,invalid,cfg)


def test_modulation_removes_mismatched_presence_gradient():
    cfg=RefinementConfig(enabled=True,width_modulation=True,gt_weight=0,shape_weight=0)
    o=observation(cfg,spacing=1.)
    presence=np.broadcast_to(np.linspace(0,1,64)[None,:,None],(64,)*3).copy()
    matched=refine_observations([o],[presence],widths=[np.full((64,)*3,100.)])[0]
    mismatch=refine_observations([o],[presence],widths=[np.full((64,)*3,140.)])[0]
    assert np.max(matched[:,1]-o.current[:,1])>0
    np.testing.assert_allclose(mismatch,o.current,atol=1e-10)
