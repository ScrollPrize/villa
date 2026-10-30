"""Neural policy extraction parity and opt-in width rejection."""
from collections import defaultdict
from pathlib import Path
import ast
import subprocess
import numpy as np
import pytest
from scipy.spatial import cKDTree
from vesuvius.neural_tracing.tracers.fibers.tracing import volume_policy as p

def original():
    root=next(d for d in Path(__file__).parents if (d/'volume-cartographer').is_dir())
    try:
        source=subprocess.check_output(['git','show',
            '84ee8852c:vesuvius/src/vesuvius/neural_tracing/tracers/fibers/tracing/volume.py'],
            cwd=root,text=True,stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        pytest.skip('historical extraction oracle is unavailable in this checkout')
    tree=ast.parse(source)
    names={'unit','tangents','runs','outside_ct_cut','KDForest','Owned','LivePaths'}
    tree.body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    ns=dict(np=np,cKDTree=cKDTree,defaultdict=defaultdict,PARALLEL=.8)
    exec(compile(tree,'historical_volume','exec'),ns)
    return ns

def test_neural_caller_and_short_piece():
    source=Path(p.__file__).with_name('volume.py')
    compile(source.read_text(),str(source),'exec')
    class MustNotQuery:
        def owner(self,*args,**kwargs):
            raise AssertionError('a too-short piece cannot survive trimming')
    assert p.novel_parts(np.zeros((47,3)),MustNotQuery(),2,24,16,48)==[]

def test_owned_exact_extraction_parity():
    old=original()['Owned'](32,merge=50);new=p.Owned(32,merge=50)
    rng=np.random.default_rng(12)
    for i in range(20):
        line=np.cumsum(rng.normal(size=(30,3)),axis=0)+rng.uniform(-60,60,3)
        old.add(line,i);new.add(line,i)
        queries=line+rng.normal(size=line.shape)
        directions=p.tangents(line)
        np.testing.assert_array_equal(old.owner(queries,directions,2),new.owner(queries,directions,2))

def test_live_exact_extraction_parity():
    old=original()['LivePaths'](2);new=p.LivePaths(2)
    rng=np.random.default_rng(13);a=rng.normal(size=(32,3))
    ids=np.arange(32)
    for step in range(70):
        b=a+rng.normal(size=a.shape)
        np.testing.assert_array_equal(old.hits(b,p.unit(b-a),ids,step),
                                      new.hits(b,p.unit(b-a),ids,step))
        old.add(a,b,ids,step);new.add(a,b,ids,step)
        if step==20:old.drop([1,2]);new.drop([1,2])
        a=b

def test_cached_deferral_witness():
    points=np.array([[0.,0,0],[10,0,0],[30,0,0]])
    running=np.array([[1.,0,0],[11,0,0]])
    witness=p.deferred_owner(points,running,3,[100,100,100],[100,200],.2)
    np.testing.assert_array_equal(witness,[0,-1,-1])
    np.testing.assert_array_equal(witness>=0,p.deferred_mask(points,running,3,
        [100,100,100],[100,200],.2))
    # The first candidate becomes eligible when its sole compatible blocker ends.
    np.testing.assert_array_equal(p.deferred_owner(points,running[1:],3,
        [100,100,100],[200],.2),[-1,-1,-1])

def test_width_owned_and_live():
    line=np.array([[0.,0,0],[1,0,0],[2,0,0]])
    owned=p.Owned(8);owned.add(line,7,100)
    d=np.array([[1.,0,0]])
    assert owned.owner([[1.,0,0]],d,2,widths=100,tolerance=.2)[0]==7
    assert owned.owner([[1.,0,0]],d,2,widths=50,tolerance=.2)[0]==-1
    assert owned.owner([[1.,0,0]],d,2,widths=50,tolerance=0)[0]==7
    live=p.LivePaths(2)
    live.add(line[:1],line[2:],np.array([7]),1,np.array([100.]))
    assert live.hits(line[1:2],d,np.array([8]),2,np.array([100.]),.2)[0]
    assert not live.hits(line[1:2],d,np.array([8]),2,np.array([50.]),.2)[0]
    assert not live.hits(line[1:2],d,np.array([7]),2,np.array([100.]),.2)[0]
    assert not live.hits(line[1:2],d,np.array([8]),1,np.array([100.]),.2)[0]

def test_width_boundary_unknown_and_defer():
    np.testing.assert_array_equal(p.width_compatible([120,121,80,79,np.nan],100,.2),
                                  [True,False,True,False,True])
    assert p.seed_deferred([0,0,0],[[1,0,0]],2,100,[100],.2)
    assert not p.seed_deferred([0,0,0],[[1,0,0]],2,100,[50],.2)
    rng=np.random.default_rng(19)
    points=rng.normal(size=(100,3));running=rng.normal(size=(17,3))
    w=rng.uniform(1,5,100);rw=rng.uniform(1,5,17)
    np.testing.assert_array_equal(p.deferred_mask(points,running,1,w,rw,.2),
        [p.seed_deferred(x,running,1,width,rw,.2) for x,width in zip(points,w)])

def test_trimming_exact_rules():
    line=np.stack([np.arange(100.),np.zeros(100),np.zeros(100)],1)
    owned=p.Owned(32);owned.add(line,1,100)
    assert p.novel_parts(line,owned,2,24,16,48,widths=100,tolerance=.2)==[]
    assert p.novel_parts(line,owned,2,24,16,48,widths=50,tolerance=.2)==[(0,100)]
    empty=p.Owned(32)
    assert p.novel_parts(line,empty,2,24,16,48)==[(0,100)]
