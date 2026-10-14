import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fiber_surface_untangle import signed_areas,untangle,FeasibleStep
from opt_loss_fixed_uv import SlidingUVTargets


def fixture():
    y,x=np.mgrid[:9,:9]
    xyz=np.stack([x*100.,y*100.,x*0.],-1).astype(np.float32)
    normals=np.zeros((8,8,3),np.float32);normals[...,2]=1
    t=SlidingUVTargets([[2,2],[6,6]],[[200,200,0],[600,600,0]],[100,100],(9,9),
        arc=[0,100],scales=[[1000,1000]]*2,source=[0,0],step=100)
    return xyz,normals,t


def test_explicit_untangling():
    x,n,t=fixture();x[4,4,0]+=180
    assert (signed_areas(torch.tensor(x),torch.tensor(n),100)<=0).any()
    uv=t.current_uv().detach().clone()
    result,history=untangle(x,t,n,100,halos=(2,),steps=400,lr=.03)
    assert (signed_areas(torch.tensor(result),torch.tensor(n),100)>0).all()
    np.testing.assert_array_equal(result[[0,-1]],x[[0,-1]])
    np.testing.assert_array_equal(result[:,[0,-1]],x[:,[0,-1]])
    torch.testing.assert_close(t.current_uv(),uv)


def test_feasible_backtracking_restores_joint_params():
    x,n,_=fixture()
    class Model:
        def __init__(self): self.p=torch.nn.Parameter(torch.tensor(x))
        def _grid_xyz(self): return self.p[None]
    m=Model();uv=torch.nn.Parameter(torch.zeros(1))
    opt=torch.optim.Adam([m.p,uv])
    guard=FeasibleStep(m.p.detach(),torch.tensor(n),100)
    saved=guard.before(opt,m)
    with torch.no_grad(): m.p[4,4,0]+=1000;uv.add_(10)
    guard.after(opt,m,saved)
    assert guard.damped==1
    assert (signed_areas(m.p,torch.tensor(n),100)>=.001).all()
    torch.testing.assert_close(uv,torch.tensor([10*guard.smallest_fraction]))


def test_failed_step_full_rollback():
    x,n,_=fixture()
    class Model:
        def __init__(self): self.p=torch.nn.Parameter(torch.tensor(x))
        def _grid_xyz(self): return self.p[None]
    m=Model();opt=torch.optim.Adam([m.p]);guard=FeasibleStep(m.p.detach(),torch.tensor(n),100,max_backtracks=0)
    saved=guard.before(opt,m)
    with torch.no_grad(): m.p.fill_(float('nan'))
    guard.after(opt,m,saved)
    assert guard.rejected==1
    np.testing.assert_array_equal(m.p.detach(),x)
