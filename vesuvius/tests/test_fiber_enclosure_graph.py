import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_enclosure_graph import extract_relations, components, graph_output
from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_signs import fit_sign


class Paths:
    def __init__(self, ta, tb, c):
        self.ta=np.asarray(ta); self.tb=np.asarray(tb); self.c=np.asarray(c); self.seam=self
    def ends(self, side):
        return np.zeros((len(self.ta),3)), self.ta if side=='a' else self.tb
    def segments(self,a,b): return self.c


def data():
    return dict(i=np.array([0,0]),j=np.array([1,1]),family=np.array([0,1]),
                sign=np.array([1,-1]),parallel=np.array([False,False]))


def test_unique_enclosure_and_connector_cancellation():
    d=data()
    r,bad=extract_relations(d,Paths([0,1],[0,0],[0,0]))
    assert not bad and r[0]['anchor_step']==0 and r[0]['local_step']==0
    # Rechart V by one turn; connector transports compensate exactly.
    r2,_=extract_relations(d,Paths([0,1],[1,1],[1,1]))
    assert r2==r
    # Reversed H traversal shifts its anchor gauge but preserves unique relation.
    r3,_=extract_relations(d,Paths([-1,0],[0,0],[0,0]))
    assert r3[0]['anchor_step']==-1


def test_bounds_nonunique_contradictory_and_swapped_endpoints():
    d=data()
    assert extract_relations(d,Paths([0,2],[0,0],[0,0]))[1][0]['reason']=='nonunique'
    assert extract_relations(d,Paths([0,0],[0,0],[0,0]))[1][0]['reason']=='contradictory'
    d['i'],d['j']=d['j'],d['i'];d['sign']*=-1
    assert extract_relations(d,Paths([0,0],[0,1],[0,0]))[0][0]['anchor_step']==0


def test_components_keep_isolates_and_conflicts():
    r=[dict(h=0,v=1,anchor_step=0),dict(h=2,v=1,anchor_step=0),
       dict(h=2,v=3,anchor_step=0),dict(h=0,v=3,anchor_step=1)]
    labels, q, groups, conflicts=components(5,r)
    assert sorted(map(len,groups))==[1,4] and conflicts


def test_modes_do_not_modify_ordinary():
    ordinary=dict(a=np.array([0]),b=np.array([1]),target=np.array([.3]),
                  weight=np.ones(1),one_sided=np.zeros(1,bool))
    r=[dict(h=0,v=1,anchor_step=0,lower_event=2,upper_event=3)]
    assert len(graph_output(ordinary,r,'only')['a'])==1
    assert len(graph_output(ordinary,r,'additional')['a'])==2
    assert graph_output(ordinary,r,'ordinary')['target'][0]==.3
    assert 'constraint_type' not in ordinary


def test_reference_fit_is_exact_and_unsupported_is_not_guessed():
    assert fit_sign([])[0]==0
    assert fit_sign([1,-1])[0]==0
    assert fit_sign([-.8,-.9,.1])[0]==-1


def test_actual_spiral_polyline_transport():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow import Seam
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_min_flow_benchmark import Paths as FiberPaths
    angle=np.linspace(0,4*np.pi,1001)
    xyz=np.column_stack(((10+angle)*np.cos(angle),(10+angle)*np.sin(angle),np.zeros(len(angle))))
    v=np.array([[10+np.pi,-.01,-1],[10+np.pi,-.01,1]])
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0),axis=1))]
    d=data();d.update(source_id=np.array([0,1]),start=np.zeros(2),
                      node_ds=np.array([[0,arc[-1]],[0,2]]),ds_a=np.array([0,arc[500]]),ds_b=np.ones(2))
    for cut in (-np.pi/2,.1,2.):
        p=object.__new__(FiberPaths);p.data=d;p.seam=Seam([[0,0,-1],[0,0,1]],cut)
        p.paths={0:(arc,xyz,p.seam.polyline(xyz)),1:(np.array([0.,2.]),v,p.seam.polyline(v))}
        r,bad=extract_relations(d,p)
        assert not bad and len(r)==1 and r[0]['local_step']==0



def test_native_cross_sign_reduction():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_reference_constraint_signs import reduce_evidence
    result=reduce_evidence(np.array([1,2,3]),np.array([1,1,2,2]),np.array([.9,.8,-.9,-.8]))
    assert [r['normal_sign'] for r in result]==[1,-1,0]
    assert [r['constraints'] for r in result]==[2,2,0]
    assert result[0]['agreement']==1
