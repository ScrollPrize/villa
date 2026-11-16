"""Native winding preparation preserves the existing measured L1 graph."""
from types import SimpleNamespace
import numpy as np
import pytest

from vc_object_store import PackedWindingGeometry,pack_winding_geometry,fiber_signed_order
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import Seam,aggregate_l1_edges
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_benchmark import graph as old_graph
from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_full import PackedPaths


def fixture():
    rng=np.random.default_rng(561)
    axis=np.array([[0.,0.,-50.],[0.,0.,50.]])
    theta=np.linspace(-3*np.pi/4,3*np.pi,81)
    curves=[np.column_stack((radius*np.cos(theta),radius*np.sin(theta),np.full(len(theta),z)))
            for radius,z in ((10.,0.),(12.,2.))]
    curves.append(np.array([[14.,0.,-5.],[14.,0.,0.],[14.,0.,5.]]))
    offsets=np.r_[0,np.cumsum([len(c) for c in curves])].astype(np.int64);xyz=np.concatenate(curves)
    packed=pack_winding_geometry(offsets,xyz,axis,workers=1)
    paths=object.__new__(PackedPaths);paths.ids=np.array(['a','b','c']);paths.offset=offsets
    paths.xyz=xyz;paths.turns=packed['turns'];paths.keys=dict(arc=packed['arc'],source=packed['source'])
    paths.seam=Seam(axis);paths.data=None;paths._initialize_index()
    family=np.array([0,0,1],np.int64);signs=np.array([1,-1,1],np.int64)
    mapping=np.array([0,1,2,-1],np.int64)
    n=421;endpoints=rng.integers(0,4,size=(n,2),dtype=np.int64)
    lengths=np.r_[paths.keys['arc'][offsets[1:]-1],1.]
    arcs=rng.random((n,2))*lengths[endpoints]
    normals_a=rng.normal(size=(n,3));normals_b=rng.normal(size=(n,3))
    normals_a/=np.linalg.norm(normals_a,axis=1)[:,None];normals_b/=np.linalg.norm(normals_b,axis=1)[:,None]
    connector=rng.normal(size=(n,3));connector[::17]=0
    parallel=rng.random(n)<.5
    data=SimpleNamespace(i=endpoints[:,0],j=endpoints[:,1],arc_a=arcs[:,0],arc_b=arcs[:,1],
        distance=rng.integers(0,81,size=n).astype(float)/16,parallel=parallel,
        normal_a=normals_a,normal_b=normals_b,connector=connector,
        valid_a=rng.random(n)>.05,valid_b=rng.random(n)>.05,valid_connector=rng.random(n)>.05)
    native=PackedWindingGeometry(offsets,packed['arc'],xyz,packed['turns'],axis,-np.pi/2)
    return paths,native,data,mapping,family,signs


def signed_reference(connector,a,b):
    length=np.linalg.norm(connector,axis=1);unit=connector/np.maximum(length[:,None],1e-12)
    sa=-np.sum(unit*a,axis=1);sb=-np.sum(unit*b,axis=1)
    known=(sa*sb>0)&(np.minimum(abs(sa),abs(sb))>=.1)&(length>0)
    return np.where(known,np.sign(sa),0).astype(np.float32)


def reference(paths,data,mapping,family,signs,config):
    parts=[];used=expanded=0;n=len(data.i)
    for first in range(0,n,config['constraint_batch']):
        batches=[];last=min(n,first+config['constraint_batch'])
        for begin in range(first,last,config['read_batch']):
            sl=slice(begin,min(last,begin+config['read_batch']))
            a=mapping[data.i[sl]];b=mapping[data.j[sl]];par=data.parallel[sl]
            keep=(a>=0)&(b>=0)&data.valid_a[sl]&data.valid_b[sl]&data.valid_connector[sl]
            keep&=par==(family[a.clip(0)]==family[b.clip(0)])
            a=a[keep];b=b[keep]
            if not len(a):continue
            d=dict(i=a,j=b,ds_a=data.arc_a[sl][keep],ds_b=data.arc_b[sl][keep],family=family,
                parallel=par[keep],distance=data.distance[sl][keep],
                sign=signed_reference(data.connector[sl][keep],data.normal_a[sl][keep]*signs[a,None],
                                      data.normal_b[sl][keep]*signs[b,None]))
            paths.data=d;g,transport=old_graph(config,d,paths)
            used+=int(transport['valid'].sum());expanded+=len(g['a'])
            batches.append(aggregate_l1_edges(**{key:g[key] for key in ('a','b','target','weight')}))
        if batches:parts.append(aggregate_l1_edges(**{key:np.concatenate([b[key] for b in batches]) for key in ('a','b','target','weight')}))
    return aggregate_l1_edges(**{key:np.concatenate([p[key] for p in parts]) for key in ('a','b','target','weight')}),used,expanded


@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('scale,offset,weight',[(1.,.5,1.),(1.03,.4,1.3)])
def test_native_winding_graph_matches_existing_partitioned_preparation(workers,scale,offset,weight):
    paths,native,data,mapping,family,signs=fixture()
    config=dict(constraint_batch=97,read_batch=19,lasagna_winding_scale=scale,
                perpendicular_offset=offset,perpendicular_relative_weight=weight)
    expected,used,expanded=reference(paths,data,mapping,family,signs,config)
    native_config=dict(config,perpendicular_weight=weight);native_config.pop('perpendicular_relative_weight')
    actual=native.prepare_winding_graph(data.i,data.j,mapping,family,signs,
        data.arc_a,data.arc_b,data.distance,data.parallel,data.normal_a,data.normal_b,data.connector,
        data.valid_a,data.valid_b,data.valid_connector,workers=workers,**native_config)
    for key in ('a','b','target'):np.testing.assert_array_equal(actual[key],expected[key])
    np.testing.assert_allclose(actual['weight'],expected['weight'],rtol=2e-15,atol=2e-15)
    if scale==1:np.testing.assert_array_equal(actual['weight'],expected['weight'])
    assert actual['read']==len(data.i) and actual['used']==used and actual['expanded']==expanded
    assert not actual['weight'].flags.writeable


def test_shared_native_signed_order_matches_numpy():
    _,_,data,_,_,_=fixture()
    np.testing.assert_array_equal(fiber_signed_order(data.connector,data.normal_a,data.normal_b),
                                  signed_reference(data.connector,data.normal_a,data.normal_b))


def filtered_events(data,mapping,family,signs):
    """The native builder's event filters, applied to the whole fixture."""
    a=mapping[data.i];b=mapping[data.j]
    keep=(a>=0)&(b>=0)&data.valid_a&data.valid_b&data.valid_connector
    keep&=data.parallel==(family[a.clip(0)]==family[b.clip(0)])
    a=a[keep];b=b[keep]
    return dict(i=a,j=b,ds_a=data.arc_a[keep],ds_b=data.arc_b[keep],family=family,parallel=data.parallel[keep],
                distance=data.distance[keep],sign=signed_reference(data.connector[keep],data.normal_a[keep]*signs[a,None],
                                                                  data.normal_b[keep]*signs[b,None]))


@pytest.mark.parametrize('workers',[1,4])
def test_native_enclosure_relations_match_python_reference(workers):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_enclosure_graph import extract_relations
    paths,native,data,mapping,family,signs=fixture()
    d=filtered_events(data,mapping,family,signs);paths.data=d
    relations,rejected=extract_relations(d,paths)
    actual=native.prepare_winding_graph(data.i,data.j,mapping,family,signs,
        data.arc_a,data.arc_b,data.distance,data.parallel,data.normal_a,data.normal_b,data.connector,
        data.valid_a,data.valid_b,data.valid_connector,workers=workers,constraint_batch=97,read_batch=19,enclosures=True)
    expected=sorted((r['h'],r['v'],r['anchor_step']) for r in relations)
    assert list(zip(actual['enclosure_a'],actual['enclosure_b'],actual['enclosure_target']))==expected
    counts=actual['enclosure_counts']
    reasons=[r['reason'] for r in rejected]
    assert counts['accepted']==len(relations) and counts['pairs']==len(relations)+len(rejected)
    assert counts['one_sided']==reasons.count('one_sided') and counts['contradictory']==reasons.count('contradictory')
    assert counts['nonunique']==reasons.count('nonunique')
    assert counts['pairs']>0 and len(rejected)>0  # random evidence is mostly rejected; acceptance is tested below
    with pytest.raises(ValueError):  # no arc type at all is a configuration error
        native.prepare_winding_graph(data.i,data.j,mapping,family,signs,
            data.arc_a,data.arc_b,data.distance,data.parallel,data.normal_a,data.normal_b,data.connector,
            data.valid_a,data.valid_b,data.valid_connector,parallel_weight=0.,perpendicular_weight=0.)


def test_native_enclosure_accepts_exact_pair_with_one_turn_apart_witnesses():
    """One outward and one inward witness whose anchors coincide one turn apart give an exact H->V step."""
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_enclosure_graph import extract_relations
    paths,native,data,mapping,family,signs=fixture()
    turn=2*np.pi*10.  # the H fixture fiber is a circle of radius 10: one turn returns to the same point
    data=SimpleNamespace(i=np.array([0,0]),j=np.array([2,2]),arc_a=np.array([1.,1.+turn]),arc_b=np.array([2.5,2.5]),
        distance=np.array([.3,.3]),parallel=np.array([False,False]),
        normal_a=np.array([[-1.,0,0],[1.,0,0]]),normal_b=np.array([[-1.,0,0],[1.,0,0]]),connector=np.array([[1.,0,0],[1.,0,0]]),
        valid_a=np.array([True,True]),valid_b=np.array([True,True]),valid_connector=np.array([True,True]))
    d=filtered_events(data,mapping,family,signs);paths.data=d
    assert d['sign'].tolist()==[1,-1]
    relations,rejected=extract_relations(d,paths)
    assert len(relations)==1 and not rejected and relations[0]['h']==0 and relations[0]['v']==2
    actual=native.prepare_winding_graph(data.i,data.j,mapping,family,signs,
        data.arc_a,data.arc_b,data.distance,data.parallel,data.normal_a,data.normal_b,data.connector,
        data.valid_a,data.valid_b,data.valid_connector,workers=1,enclosures=True)
    assert actual['enclosure_counts']=={'pairs':1,'accepted':1,'one_sided':0,'contradictory':0,'nonunique':0}
    assert actual['enclosure_a'].tolist()==[0] and actual['enclosure_b'].tolist()==[2]
    assert actual['enclosure_target'].tolist()==[relations[0]['anchor_step']]
    # Enclosures do not depend on the perpendicular weight: zero drops those arcs but keeps the relation.
    only=native.prepare_winding_graph(data.i,data.j,mapping,family,signs,
        data.arc_a,data.arc_b,data.distance,data.parallel,data.normal_a,data.normal_b,data.connector,
        data.valid_a,data.valid_b,data.valid_connector,workers=1,parallel_weight=0.,perpendicular_weight=0.,enclosures=True)
    assert only['used']==0 and len(only['a'])==0 and only['enclosure_target'].tolist()==actual['enclosure_target'].tolist()


def test_events_without_winding_distance_witness_enclosures_but_emit_no_arcs():
    """NaN distances (geometry-only extraction) give no automatic arc, still exact enclosure bounds."""
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_enclosure_graph import extract_relations
    paths,native,data,mapping,family,signs=fixture()
    turn=2*np.pi*10.
    data=SimpleNamespace(i=np.array([0,0,0]),j=np.array([2,2,1]),arc_a=np.array([1.,1.+turn,3.]),arc_b=np.array([2.5,2.5,4.]),
        distance=np.array([np.nan,np.nan,np.nan]),parallel=np.array([False,False,True]),
        normal_a=np.array([[-1.,0,0],[1.,0,0],[0,1.,0]]),normal_b=np.array([[-1.,0,0],[1.,0,0],[0,1.,0]]),
        connector=np.array([[1.,0,0],[1.,0,0],[1.,0,0]]),
        valid_a=np.array([True]*3),valid_b=np.array([True]*3),valid_connector=np.array([True]*3))
    actual=native.prepare_winding_graph(data.i,data.j,mapping,family,signs,
        data.arc_a,data.arc_b,data.distance,data.parallel,data.normal_a,data.normal_b,data.connector,
        data.valid_a,data.valid_b,data.valid_connector,workers=1,enclosures=True)
    assert len(actual['a'])==0 and actual['used']==0 and actual['without_distance']==3
    assert actual['enclosure_counts']['accepted']==1 and actual['enclosure_a'].tolist()==[0] and actual['enclosure_b'].tolist()==[2]
    d=filtered_events(data,mapping,family,signs);paths.data=d
    relations,_=extract_relations(d,paths)
    assert actual['enclosure_target'].tolist()==[relations[0]['anchor_step']]
    with pytest.raises(ValueError):  # a negative distance is still an error
        bad=np.array([-1.,np.nan,np.nan])
        native.prepare_winding_graph(data.i,data.j,mapping,family,signs,data.arc_a,data.arc_b,bad,data.parallel,
            data.normal_a,data.normal_b,data.connector,data.valid_a,data.valid_b,data.valid_connector,workers=1)
