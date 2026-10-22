import json
import sqlite3
import numpy as np
import pytest
from functools import partial
from vesuvius.neural_tracing.fiber_trace_3d.fiber_json_solve import convert as convert_verified

convert=partial(convert_verified,expected_manifest_tokens=['test-volume','test-scan'])


def test_stage_requests_preserve_arbitrary_settings_and_order(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_json_solve import stage_requests
    request={'fiber_normal_sign':{'afv':'{afv}','output_root':'{output_root}/oriented',
        'steps':17,'lr':.013,'gpu_manager':{'vram_bytes_per_gpu':123456},'force_keep_fibers':False}}
    path=tmp_path/'stage.json';path.write_text(json.dumps(request))
    stages=stage_requests([{'name':'user-label','config_file':str(path)},
        {'name':'native','request':{'commands':[{'binary':'custom','args':['{afv}']}]}}],tmp_path,'input.afv')
    assert [s['name'] for s in stages]==['user-label','native']
    expected=dict(request['fiber_normal_sign'],afv='input.afv',output_root=str(tmp_path/'oriented'))
    assert stages[0]['request']=={'fiber_normal_sign':expected}
    assert stages[1]['request']['commands'][0]['args']==['input.afv']
    assert stage_requests([],tmp_path,'input.afv')==[]


@pytest.mark.parametrize('stages',[
    [{'name':'../escape','request':{'a':{}}}],
    [{'name':'same','request':{'a':{}}}]*2,
    [{'name':'x','request':{'a':{},'b':{}}}],
    [{'name':'x'}],
    [{'name':'x','request':{'a':{}},'config_file':'unused'}],
])
def test_stage_requests_reject_ambiguous_or_unsafe_input(tmp_path,stages):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_json_solve import stage_requests
    with pytest.raises(ValueError):stage_requests(stages,tmp_path,'input.afv')


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
    template=tmp_path/'template.afv'
    with sqlite3.connect(template) as db:
        db.executescript('''
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
        CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT,family TEXT,point_count INTEGER,length REAL,
        min_x REAL,max_x REAL,min_y REAL,max_y REAL,min_z REAL,max_z REAL,annotation TEXT);
        CREATE TABLE blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER,first_segment INTEGER,points BLOB);
        CREATE VIRTUAL TABLE block_bounds USING rtree(id,min_x,max_x,min_y,max_y,min_z,max_z);
        ''')
        db.execute('INSERT INTO metadata VALUES (?,?)',('root',json.dumps({'coordinate_base_shape_zyx':[20,20,20]})))
    source=tmp_path/'source';source.mkdir()
    doc=dict(coordinate_base_shape_zyx=[20,20,20],line_points=[[1,2,3],[4,5,6]],
             control_points=[dict(segment_to_next=dict(normal_manifest='test-volume/test-scan',interp_mode='trace')),{}],
             hv_classification={'automatic_tag':'H'},branches=[],adjacent_branches=[])
    path=source/'a.json';path.write_text(json.dumps(doc))
    output=tmp_path/'out.afv'
    records=convert(source,template,output)
    assert len(records)==1
    assert convert(source,template,output)==records
    doc['line_points'][0][0]=2;path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='Changed source'):
        convert(source,template,output)
    doc['adjacent_branches']=[{}];path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='explicit links'):
        convert(source,template,tmp_path/'other.afv')
    doc['adjacent_branches']=[]
    del doc['coordinate_base_shape_zyx']
    path.write_text(json.dumps(doc))
    assert len(convert(source,template,tmp_path/'missing_shape.afv'))==1
    doc['line_points'][0][0]=20
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='outside verified'):
        convert(source,template,tmp_path/'outside.afv')
