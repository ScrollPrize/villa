"""History broad-phase skips only values that already round to float32 zero."""
import math

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.shared.fast_sample import sample_crop, sample_scalar_crop


@pytest.mark.parametrize('shape', [(9,11,7), (1,3,1)])
def test_scalar_interpolation_exact_inside_and_across_zero_padded_edges(shape):
    rng = np.random.default_rng(17)
    raw = rng.integers(0,256,(1,*shape),dtype=np.uint8)
    grid = rng.uniform(-3,12,(400,3))
    grid = np.concatenate((grid, [[0,0,0], [shape[2]-1,shape[1]-1,shape[0]-1]]))
    expected = np.zeros(len(grid),np.float32)
    for i,xyz in enumerate(grid):
        z,y,x = xyz[::-1]
        iz,iy,ix = math.floor(z),math.floor(y),math.floor(x)
        fz,fy,fx = z-iz,y-iy,x-ix
        value = 0.
        for dz in (0,1):
            for dy in (0,1):
                for dx in (0,1):
                    if 0 <= iz+dz < shape[0] and 0 <= iy+dy < shape[1] and 0 <= ix+dx < shape[2]:
                        weight = (fz if dz else 1-fz)*(fy if dy else 1-fy)*(fx if dx else 1-fx)
                        value += weight*raw[0,iz+dz,iy+dy,ix+dx]
        expected[i] = value*(1./255.)
    out = np.empty(len(grid),np.float32)
    sample_scalar_crop(raw,np.zeros(3),np.zeros(3),np.eye(3),grid,out)
    np.testing.assert_array_equal(out,expected)


def brute_history(grid, hist, mask, sigma, segments):
    """Exhaustive scalar reference, including the sampler's legacy forward cutoff."""
    result=np.zeros(len(grid),np.float32)
    forward=max((p[2] for p,m in zip(hist,mask) if m>0),default=-1e30)
    if segments and len(hist) and mask[0]>0:
        forward=max(forward,0.)
    for i,(a,b,c) in enumerate(grid):
        if c-forward>7.1*sigma:
            continue
        best=1e30
        for k,p in enumerate(hist):
            if mask[k]<=0:
                continue
            d0,d1,d2=a-p[0],b-p[1],c-p[2]
            if segments and (k==0 or mask[k-1]>0):
                v=(hist[k-1] if k else np.zeros(3))-p
                v0,v1,v2=v
                vv=v0*v0+v1*v1+v2*v2
                t=min(max((d0*v0+d1*v1+d2*v2)/max(vv,1e-12),0.),1.)
                d0-=t*v0;d1-=t*v1;d2-=t*v2
            best=min(best,d0*d0+d1*d1+d2*d2)
        if best<1e29:
            result[i]=math.exp(-best/(2.*sigma*sigma))
    return result


@pytest.mark.parametrize('mode',['points','segments'])
@pytest.mark.parametrize('sigma',[.05,.35,1.,4.])
@pytest.mark.parametrize('case',['curved','gap','missing','empty','origin','duplicate'])
def test_history_bounds_match_exhaustive_float32(mode,sigma,case):
    rng=np.random.default_rng(713)
    hist=np.cumsum(rng.normal(size=(9,3)),axis=0)
    hist[:,2]=-np.arange(1,10)
    mask=np.ones(len(hist))
    if case=='gap':
        mask[[0,3,4,8]]=0
        hist[mask==0]=np.nan
    elif case=='missing':
        mask[:]=0
    elif case=='empty':
        hist=hist[:0];mask=mask[:0]
    elif case=='origin':
        hist[:]=[40.,30.,-20.]
    elif case=='duplicate':
        hist[:]=[0.,0.,-1.]
    # Include Gaussian subnormal tails and locations beyond the bounding box,
    # plus the origin-to-first-point segment and a masked gap.
    grid=np.concatenate([rng.uniform(-40,40,(300,3)),rng.normal(size=(300,3)),
                         np.array([[r*sigma,0.,-1.] for r in (7.,14.,14.3,14.5,15.9,16.,16.1,20.)]),
                         np.array([[0.,0.,0.],[20.,15.,-10.],[40.,30.,-20.]])])
    result=sample_crop(np.zeros((1,3,3,3),np.uint8),np.zeros(3),np.zeros(3),np.eye(3),
                       grid,hist,mask,2,sigma,mode)
    np.testing.assert_array_equal(result[-1],brute_history(grid,hist,mask,sigma,mode=='segments'))
