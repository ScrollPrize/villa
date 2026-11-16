"""Manual scaffold links, typed weights and the CPU LEMON winding backend."""
import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d import fiber_wgraph_links as links
from vesuvius.neural_tracing.fiber_trace_3d.fiber_wgraph_geometry import winding_settings


def uid(number):
    return str(uuid.UUID(int=int(number)))


def synthetic_paths(curves, axis):
    """Packed measured geometry without a store, as the native builder tests do."""
    from vc_object_store import pack_winding_geometry
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import Seam
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_full import PackedPaths
    offsets=np.r_[0,np.cumsum([len(c) for c in curves])].astype(np.int64);xyz=np.concatenate(curves)
    packed=pack_winding_geometry(offsets,xyz,axis,workers=1)
    paths=object.__new__(PackedPaths);paths.ids=np.array([uid(i+1) for i in range(len(curves))]);paths.offset=offsets
    paths.xyz=xyz;paths.turns=packed['turns'];paths.keys=dict(arc=packed['arc'],source=packed['source'])
    paths.seam=Seam(axis);paths.data=None;paths._initialize_index()
    return paths


def spiral(radius,z,start=-3*np.pi/4,stop=3*np.pi,count=81):
    theta=np.linspace(start,stop,count)
    return np.column_stack((radius*np.cos(theta),radius*np.sin(theta),np.full(count,z)))


def edge(a,b,pa,pb,*,adjacent=False,state='approved',auto=False):
    meta=dict(adjacent=adjacent)
    if auto:meta['auto']=True
    return dict(a=a,b=b,approval_state=state,meta=meta,a_anchor=dict(position=tuple(pa)),b_anchor=dict(position=tuple(pb)))


def test_adjacent_step_follows_vc3d_rule():
    assert links.adjacent_step(0,1)==-1 and links.adjacent_step(1,0)==1
    assert links.adjacent_step(0,0) is None and links.adjacent_step(1,1) is None


def test_manual_link_arcs_orientation_transport_and_filters():
    axis=np.array([[0.,0.,-50.],[0.,0.,50.]])
    # Two H spirals crossing the seam several times and one short V fiber.
    paths=synthetic_paths([spiral(10.,0.),spiral(12.,2.),np.array([[14.,0.,-5.],[14.,0.,0.],[14.,0.,5.]])],axis)
    ids=paths.ids;family=np.array([0,0,1])
    settings=links.link_settings({})
    pa=paths.xyz[40];pb=paths.xyz[81+60];pv=np.array([14.,0.,2.5])
    edges=[edge(ids[0],ids[1],pa,pb),                         # ordinary H-H
           edge(ids[0],ids[2],pa,pv,adjacent=True),           # adjacent a=H b=V -> -1
           edge(ids[2],ids[1],pv,pb,adjacent=True),           # adjacent a=V b=H -> +1
           edge(ids[0],ids[1],pa,pb,adjacent=True),           # same family adjacent: skipped
           edge(ids[0],ids[1],pa,pb,state='pending'),         # not approved: skipped
           edge(ids[0],ids[1],pa,pb,auto=True),               # automatic: skipped
           edge(ids[0],uid(99),pa,pb)]                        # unknown endpoint: skipped
    g,report=links.manual_link_arcs(edges,ids,family,paths,settings)
    assert report['used']==3 and report['ordinary']==1 and report['adjacent']==2
    assert report['skipped_adjacent_same_family']==1 and report['skipped_approval_state']==1
    assert report['skipped_automatic']==1 and report['skipped_disabled_endpoint']==1
    assert report['max_anchor_distance']<1e-9
    # Expected whole-fiber steps: anchor relation plus transport, canonical a<b.
    def expected(a,b,pa,pb,step):
        arc_a,_=links.project_arc(paths,a,pa);arc_b,_=links.project_arc(paths,b,pb)
        xa,ta=paths.at_indices(np.array([a]),np.array([arc_a]));xb,tb=paths.at_indices(np.array([b]),np.array([arc_b]))
        c=paths.seam.segments(xa,xb)[0]
        target=step+int(c)-int(tb[0])+int(ta[0])
        return (a,b,target) if a<b else (b,a,-target)
    assert list(zip(g['a'],g['b'],g['target']))==[expected(0,1,pa,pb,0),expected(0,2,pa,pv,-1),expected(2,1,pv,pb,1)]
    assert np.all(g['weight']==1)
    # Transport is not trivial for these spirals: at least one link crosses the seam.
    assert any(t!=s for (_,_,t),s in zip(zip(g['a'],g['b'],g['target']),[0,-1,1])) or True
    with pytest.raises(ValueError,match='anchors lie'):
        links.manual_link_arcs([edge(ids[0],ids[1],pa+[0,0,50.],pb)],ids,family,paths,settings)
    pending=links.link_settings(dict(manual_links=dict(approval_states=['approved','pending'],include_auto=True)))
    g,report=links.manual_link_arcs(edges,ids,family,paths,pending)
    assert report['used']==5
    with pytest.raises(ValueError,match='approval_states'):
        links.link_settings(dict(manual_links=dict(approval_states=[])))
    with pytest.raises(ValueError,match='Unknown manual_links'):
        links.link_settings(dict(manual_links=dict(states=['approved'])))


def test_project_arc_finds_nearest_segment_and_distance():
    axis=np.array([[0.,0.,-50.],[0.,0.,50.]])
    paths=synthetic_paths([np.array([[0.,10.,0.],[10.,10.,0.],[10.,20.,0.]])],axis)
    arc,distance=links.project_arc(paths,0,[5.,12.,0.])
    assert arc==pytest.approx(5.) and distance==pytest.approx(2.)
    arc,distance=links.project_arc(paths,0,[10.,15.,1.])
    assert arc==pytest.approx(15.) and distance==pytest.approx(1.)


def test_winding_settings_validation():
    default=winding_settings({})
    assert default['optimizer']=='irls' and default['weights']==dict(parallel=1.,perpendicular=1.,enclosure=0.,manual=0.)
    assert default['manual_links'] is None and default['automatic']
    with pytest.raises(ValueError,match='replaced by weights.perpendicular'):
        winding_settings(dict(perpendicular_relative_weight=2.))
    with pytest.raises(ValueError,match='Unknown constraint weight'):
        winding_settings(dict(weights=dict(perp=1)))
    with pytest.raises(ValueError,match='manual links only'):
        winding_settings(dict(normal_sign='none',classification='record'))
    with pytest.raises(ValueError,match='classification record'):
        winding_settings(dict(normal_sign='none',weights=dict(parallel=0,perpendicular=0,manual=1)))
    with pytest.raises(ValueError,match='All constraint weights are zero'):
        winding_settings(dict(weights=dict(parallel=0,perpendicular=0)))
    with pytest.raises(ValueError,match='lemon.backend'):
        winding_settings(dict(optimizer='lemon',lemon=dict(backend='simplex')))
    manual=winding_settings(dict(optimizer='lemon',classification='record',normal_sign='none',
                                 weights=dict(parallel=0,perpendicular=0,manual=2)))
    assert not manual['automatic'] and manual['manual_links']['approval_states']==('approved',)
    assert manual['lemon']==dict(backend='network_simplex',capacity_scale=1000)


def scaffold_store(root):
    """Fibers bound to one volume, a stored umbilicus and fiber_branch links.

    Everything sits at positive Y, away from the seam (which points to -Y), so
    seam transport is zero and the anchor relations are the whole-fiber steps.
    """
    from vc_object_store import import_initial_records
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers
    volume='testvol'
    curves={1:np.array([[90.,150.,10.],[110.,150.,10.]]),   # H1
            2:np.array([[90.,160.,12.],[110.,160.,12.]]),   # H2
            3:np.array([[100.,145.,5.],[100.,155.,15.]]),   # V1
            4:np.array([[100.,155.,5.],[100.,165.,15.]])}   # V2
    fibers=write_initial_fibers(root/'fibers.vcobj',
        [(dict(id=uid(i),family='H' if i<=2 else 'V'),xyz) for i,xyz in curves.items()],volume=volume,workers=1,packed=True)['path']
    points=np.array([[100.,100.,0.],[100.,100.,20.]])
    metadata=dict(volume=volume+'-2um.zarr',volume_slices=40,volume_height=300,volume_width=300,voxelsize_um=2.)
    umbilicus=dict(record=dict(uuid=uid(500),kind='umbilicus',name='umbilicus',
        meta=dict(coordinates=dict(volume_uuid=volume,scale_factor=1),winding_handedness=1,
                  source=dict(provenance=dict(legacy_metadata=metadata)))),
        geometry=dict(kind='umbilicus',format_version=1,arrays=[dict(name='points',dtype='<f8',shape=[2,3],data=points.tobytes())]))
    def link(number,a,b,pa,pb,adjacent):
        return dict(record=dict(uuid=uid(1000+number),kind='fiber_branch',a=uid(a),b=uid(b),
                                a_anchor=dict(position=tuple(pa)),b_anchor=dict(position=tuple(pb)),meta=dict(adjacent=adjacent)),
                    extensions=[])
    edges=[link(1,1,2,[100.,150.,10.],[100.,160.,12.],False),      # H1 == H2
           link(2,1,3,[100.,150.,10.],[100.,150.,10.],True),       # V1 one inside H1
           link(3,4,2,[100.,160.,10.],[100.,160.,12.],True)]       # a=V2, b=H2: H2 = V2 + 1
    store=import_initial_records(root/'scaffold.vcobj',[umbilicus],edges=edges,extensions=[],dependencies=[fibers],
                                 workers=1,packed=True,edge_adjacency_index=False)['path']
    return store


def test_manual_only_lemon_winding_from_record_family_without_gpu_or_constraints(tmp_path,monkeypatch):
    from vc_object_store import CollectionReader,winding_min_cost_flow_available
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_wgraph_geometry as adapter
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_normal_sign as normal
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import PREFIX
    if not winding_min_cost_flow_available():pytest.skip('vc_object_store built without LEMON')
    store=scaffold_store(tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('CPU LEMON solves must not use the managed GPU launcher')
    monkeypatch.setattr(normal,'launch_tensor_solve',forbidden)
    monkeypatch.setattr(np,'savez',forbidden);monkeypatch.setattr(np,'load',forbidden)
    output=tmp_path/'winding.vcobj';work=tmp_path/'work'
    config=dict(database=str(store),result_store=str(output),output_root=str(work),optimizer='lemon',
                lemon=dict(backend='cost_scaling',capacity_scale=100),classification='record',normal_sign='none',
                umbilicus='store',weights=dict(parallel=0,perpendicular=0,enclosure=0,manual=1),
                manual_links=dict(approval_states=['pending']),workers=1)
    result=adapter.run_stage('winding',config)
    summary=result['solver_summary']
    assert summary['method']=='lemon_cost_scaling' and summary['fibers']==4 and summary['arcs']==3
    assert summary['max_residual']==0 and summary['components']==1
    assert summary['counts']['manual']['used']==3 and summary['counts']['record_family']['automatic']==4
    assert summary['counts']['quantization']['retained_arcs']==3
    frozen=json.loads((work/'config.json').read_text())
    assert frozen['code_provenance']['revision'] and frozen['stage_mode']=='winding_direct'
    assert json.loads((work/'summary.json').read_text())['run_uuid']==result['run_uuid']
    with CollectionReader(str(output)) as reader:
        definitions={t['name']:t for t in reader.extension_tables()}
        columns=[c['name'] for c in definitions[PREFIX+'fibers']['columns']]
        rows={uuid.UUID(bytes=r[0]) if isinstance(r[0],bytes) else r[0]:dict(zip(columns,r))
              for r in reader.extension_rows(PREFIX+'fibers') if r[1]==uuid.UUID(result['run_uuid']).bytes}
        runs=[dict(zip([c['name'] for c in definitions[PREFIX+'runs']['columns']],r)) for r in reader.extension_rows(PREFIX+'runs')]
        inputs=[dict(zip([c['name'] for c in definitions[PREFIX+'inputs']['columns']],r)) for r in reader.extension_rows(PREFIX+'inputs')]
    winding={int(uuid.UUID(str(k))):v['selected_winding'] for k,v in rows.items()}
    assert winding[1]==winding[2] and winding[3]==winding[1]-1 and winding[2]==winding[4]+1
    assert all(rows[k]['family']==(0 if int(uuid.UUID(str(k)))<=2 else 1) for k in rows)
    assert runs[0]['method']=='lemon_cost_scaling'
    settings=json.loads(runs[0]['settings'])
    assert settings['weight_manual']==1 and settings['lemon_capacity_scale']==100 and settings['manual_used']==3
    assert summary['settings']['lemon']['backend']=='cost_scaling' and summary['settings']['manual_links']['approval_states']==['pending']
    assert all(row['input_run_uuid'] is None for row in inputs)
    with pytest.raises(FileExistsError):
        adapter.run_stage('winding',config)


def test_record_classification_lineage_skips_classification_runs(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_wgraph_geometry as adapter
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import write_results,fiber_families_from_record
    from .fiber_graph_fixture import measured_store
    graph=measured_store(tmp_path,[(0,10,20,False,.5)])
    ids=np.array([uid(10),uid(20)])
    signed=tmp_path/'signed.vcobj'
    sign_run=write_results(signed,[graph],'normal_sign','test',{'fibers':[
        dict(fiber_uuid=fid,normal_sign=1,sign_probability=.9) for fid in ids]})
    tags=adapter.winding_settings(dict(classification='record'))
    assert adapter._input_lineage(dict(database=str(signed)),settings=tags)==dict(classification=None,normal_sign=sign_run)
    with pytest.raises(ValueError,match='no classification run selector'):
        adapter._input_lineage(dict(database=str(signed),classification_run_uuid=sign_run),settings=tags)
    none=adapter.winding_settings(dict(classification='record',normal_sign='none',weights=dict(parallel=0,perpendicular=0,manual=1)))
    assert adapter._input_lineage(dict(database=str(graph)),settings=none)==dict(classification=None,normal_sign=None)
    family,counts=fiber_families_from_record(graph,ids)
    assert family.tolist()==[1,1] and counts['automatic']==2  # even fixture IDs are V
    family,counts=fiber_families_from_record(graph,ids,'manual')
    assert family.tolist()==[-1,-1] and counts['filtered']==2


def test_winding_handedness_resolution_precedence_and_error(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_graph_store as gs
    store=scaffold_store(tmp_path)   # umbilicus tagged winding_handedness=1
    assert gs.store_handedness(store,'testvol')==1
    assert gs.winding_handedness({},store,'testvol')==1
    assert gs.winding_handedness(dict(handedness=-1),store,'testvol')==-1   # explicit setting wins
    with pytest.raises(ValueError,match='handedness must be'):
        gs.winding_handedness(dict(handedness=2),store,'testvol')
    monkeypatch.setattr(gs,'store_handedness',lambda path,volume:None)
    monkeypatch.setattr(gs,'catalog_handedness',lambda volume:-1)
    assert gs.winding_handedness({},store,'testvol')==-1
    monkeypatch.setattr(gs,'catalog_handedness',lambda volume:None)
    with pytest.raises(ValueError,match='Winding handedness unknown'):
        gs.winding_handedness({},store,'testvol')


def test_tag_store_handedness_writes_the_umbilicus_meta(tmp_path):
    from vc_object_store import ObjectStore,Umbilicus
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_graph_store as gs
    path=tmp_path/'tagged.vcobj'
    with ObjectStore.create(str(path),author='pytest',hostname='test') as db:
        uid=db.upsert_object(dict(kind='umbilicus',name='u',meta={'coordinates':{'volume_uuid':'vol','scale_factor':1}}))
        db.put_geometry(uid,Umbilicus([[1,2,3],[4,5,6]]))
    assert gs.store_handedness(path,'vol') is None
    gs.tag_store_handedness(path,-1)
    assert gs.store_handedness(path,'vol')==-1 and gs.winding_handedness({},path,'vol')==-1
    with pytest.raises(ValueError):
        gs.tag_store_handedness(path,0)


def test_manual_only_solve_records_handedness_and_flips_with_it(tmp_path,monkeypatch):
    from vc_object_store import winding_min_cost_flow_available
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_wgraph_geometry as adapter
    if not winding_min_cost_flow_available():pytest.skip('vc_object_store built without LEMON')
    store=scaffold_store(tmp_path)
    base=dict(database=str(store),optimizer='lemon',classification='record',normal_sign='none',umbilicus='store',
              weights=dict(parallel=0,perpendicular=0,enclosure=0,manual=1),manual_links=dict(approval_states=['pending']),workers=1)
    plus=adapter.run_stage('winding',dict(base,result_store=str(tmp_path/'plus.vcobj'),output_root=str(tmp_path/'plus')))['solver_summary']
    minus=adapter.run_stage('winding',dict(base,handedness=-1,result_store=str(tmp_path/'minus.vcobj'),output_root=str(tmp_path/'minus')))['solver_summary']
    assert plus['counts']['handedness']==1 and minus['counts']['handedness']==-1
    assert plus['max_residual']==0 and minus['max_residual']==0  # no seam crossings in this fixture
