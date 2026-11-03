import json
import numpy as np
import pytest
from functools import partial
from vesuvius.neural_tracing.fiber_trace_3d.fiber_json_solve import convert as convert_verified

VOLUME='20231117143551'
convert=partial(convert_verified,volume=VOLUME,shape_zyx=[20,20,20],expected_manifest_tokens=['test-volume','test-scan'])


def test_stage_requests_preserve_arbitrary_settings_and_order(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_json_solve import stage_requests
    request={'fiber_normal_sign':{'fibers':'{fibers}','output_root':'{output_root}/oriented',
        'steps':17,'lr':.013,'gpu_manager':{'vram_bytes_per_gpu':123456},'force_keep_fibers':False}}
    path=tmp_path/'stage.json';path.write_text(json.dumps(request))
    stages=stage_requests([{'name':'user-label','config_file':str(path)},
        {'name':'native','request':{'commands':[{'binary':'custom','args':['{fibers}']}]}}],tmp_path,'input.vcobj')
    assert [s['name'] for s in stages]==['user-label','native']
    expected=dict(request['fiber_normal_sign'],fibers='input.vcobj',output_root=str(tmp_path/'oriented'))
    assert stages[0]['request']=={'fiber_normal_sign':expected}
    assert stages[1]['request']['commands'][0]['args']==['input.vcobj']
    assert stage_requests([],tmp_path,'input.vcobj')==[]


@pytest.mark.parametrize('stages',[
    [{'name':'../escape','request':{'a':{}}}],
    [{'name':'same','request':{'a':{}}}]*2,
    [{'name':'x','request':{'a':{},'b':{}}}],
    [{'name':'x'}],
    [{'name':'x','request':{'a':{}},'config_file':'unused'}],
])
def test_stage_requests_reject_ambiguous_or_unsafe_input(tmp_path,stages):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_json_solve import stage_requests
    with pytest.raises(ValueError):stage_requests(stages,tmp_path,'input.vcobj')


def test_transfer_151_radial_prior():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import radial_initial_geometry, transfer_radial_initial_geometry
    prior=dict(radius=np.linspace(100,10000,1000),initial_k=np.arange(1000),width_initial=np.ones(100)*100)
    cfg=dict(winding_count=151)
    centers,widths=transfer_radial_initial_geometry(prior,np.array([2000.,2100.]),cfg)
    small=dict(radius=np.array([2000.,2100.]),radial_initial_centers=centers,radial_initial_widths=widths)
    actual,actual_width=radial_initial_geometry(small,cfg)
    assert abs(actual[1]-actual[0])<2
    assert len(actual_width)==151
    np.testing.assert_array_equal(actual_width,radial_initial_geometry(prior,cfg)[1])


def test_trusted_gt_has_no_broken_mass_or_gradient():
    import torch
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import fresh_initial_state
    d=dict(family=np.array([0,0,1,1]),radius=np.array([2000.,2100.,2200.,2300.]))
    state,widths=fresh_initial_state(d,dict(winding_count=151,layout_mode='shared_anchor',force_keep_fibers=True))
    x=torch.tensor(state,requires_grad=True)
    p=x[:,:152].softmax(1)
    assert torch.equal(p[:,-1],torch.zeros(4))
    (p[:,0].sum()+40*p[:,-1].mean()).backward()
    assert torch.equal(x.grad[:,151],torch.zeros(4))


def test_conversion_roundtrip_and_source_guard(tmp_path):
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import read_polylines
    source=tmp_path/'source';source.mkdir()
    uuid='12345678-1234-1234-1234-123456789abc'
    doc=dict(uuid=uuid,coordinate_base_shape_zyx=[20,20,20],line_points=[[1,2,3],[4,5,6]],
             control_points=[dict(segment_to_next=dict(normal_manifest='test-volume/test-scan',interp_mode='trace')),{}],
             hv_classification={'automatic_tag':'H'},branches=[],adjacent_branches=[])
    path=source/'a.json';path.write_text(json.dumps(doc))
    output=tmp_path/'out.vcobj'
    records=convert(source,output)
    assert len(records)==1 and records[0]['id']==uuid
    assert convert(source,output)==records
    [(row,xyz)]=read_polylines(output)
    assert row['id']==uuid and row['name']=='a.json' and row['meta']['annotation']==doc
    np.testing.assert_array_equal(xyz,doc['line_points'])
    doc['line_points'][0][0]=2;path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='Changed source'):
        convert(source,output)
    doc['adjacent_branches']=[{}];path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='explicit links'):
        convert(source,tmp_path/'other.vcobj')
    doc['adjacent_branches']=[]
    del doc['coordinate_base_shape_zyx']
    path.write_text(json.dumps(doc))
    assert len(convert(source,tmp_path/'missing_shape.vcobj'))==1
    doc['coordinates']=dict(volume_uuid=VOLUME,scale_factor=.5);path.write_text(json.dumps(doc))
    convert(source,tmp_path/'scaled.vcobj')
    np.testing.assert_array_equal(read_polylines(tmp_path/'scaled.vcobj')[0][1],np.array(doc['line_points'])*2)
    doc['coordinates']=dict(volume_uuid='other',scale_factor=1);path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='another volume'):
        convert(source,tmp_path/'other_volume.vcobj')
    del doc['coordinates']
    doc['line_points'][0][0]=20
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='outside verified'):
        convert(source,tmp_path/'outside.vcobj')
