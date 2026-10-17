"""Public elastic geometry, strength, and replay contracts."""
import pickle
import struct
import numpy as np
import pytest
from vc_reader import augment as A


def displacement(plan):
    """Inspect the first geometric step in the trusted VCA1 replay record."""
    data=plan._native.dump(); offset=4
    def number():
        nonlocal offset
        value=struct.unpack_from('<Q',data,offset)[0]; offset+=8; return value
    def sequence(dtype):
        nonlocal offset
        count=number(); value=np.frombuffer(data,dtype=dtype,count=count,offset=offset).copy()
        offset+=count*np.dtype(dtype).itemsize; return value
    def text():
        nonlocal offset
        count=number(); offset+=count
    for _ in range(number()): text(); sequence('<u8'); number(); number()
    assert number()==1
    sequence('<u8'); assert number()==0
    shape=tuple(sequence('<u8').astype(int)); sequence('<f8'); sequence('<f8')
    field=sequence('<f8')
    return np.zeros((*shape,len(shape))) if not len(field) else field.reshape(*shape,len(shape))


@pytest.mark.parametrize('shape',[(17,23),(13,17,23),(1,7,11),(1,1)])
@pytest.mark.parametrize('strength',[0.,.1,8.])
def test_strength_bounds_replay_and_independent_resampling(shape,strength):
    ndi=pytest.importorskip('scipy.ndimage')
    rng=np.random.default_rng(12)
    image=rng.normal(size=(2,*shape))
    labels=(np.arange(np.prod(shape),dtype=np.uint64)+2**63).reshape(1,*shape)
    sample=dict(image=image,mask=labels)
    pipeline=A.Pipeline([A.Elastic(sigma=4,magnitude=strength,border=True)])
    plan=pipeline.sample(sample,seed=123)
    field=displacement(plan)
    np.testing.assert_allclose(field.mean(axis=tuple(range(len(shape)))),0,atol=2e-14)
    rms=np.sqrt(np.mean(np.sum(field**2,axis=-1)))
    assert rms<=strength+1e-12
    if strength==.1 and np.prod(shape)>1: assert rms==pytest.approx(.1)
    maxima=np.stack([np.abs(np.diff(field,axis=a)).reshape(-1,len(shape)).max(axis=0,initial=0)
                     for a in range(len(shape))])
    assert maxima.sum(axis=0).max()<=.8+1e-12
    coords=np.indices(shape,dtype=float)+field.transpose(-1,*range(len(shape)))
    out=pipeline(sample,seed=123)
    for channel in range(2):
        expected=ndi.map_coordinates(image[channel],coords,order=1,mode='nearest')
        np.testing.assert_allclose(out['image'][channel],expected,atol=3e-14,rtol=3e-14)
    nearest=tuple(np.rint(np.clip(coords[a],0,n-1)).astype(int) for a,n in enumerate(shape))
    np.testing.assert_array_equal(out['mask'][0],labels[0][nearest])
    for restored in [plan.apply(sample),pickle.loads(pickle.dumps(plan)).apply(sample)]:
        for key in sample: np.testing.assert_array_equal(restored[key],out[key])
    if strength==0:
        for key in sample: np.testing.assert_array_equal(out[key],sample[key])


@pytest.mark.parametrize('shape',[(17,23),(13,17,23)])
def test_points_and_normals_use_the_same_deformation(shape):
    d=len(shape); center=tuple(n//2 for n in shape)
    pipeline=A.Pipeline([A.Elastic(sigma=4,magnitude=.5,border=True)],targets={
        'image':A.Image(),'points':A.Points(order='zyx'[-d:]),
        'normals':A.NormalField(components='zyx'[-d:])})
    normals=np.zeros((d,*shape)); normals[0]=1
    sample=dict(image=np.ones((1,*shape)),points=np.zeros((1,d)),normals=normals)
    plan=pipeline.sample(sample,seed=7); field=displacement(plan)
    sample['points'][0]=np.asarray(center)+field[center]
    out=plan.apply(sample)
    np.testing.assert_allclose(out['points'][0],center,atol=1e-7)
    j=np.eye(d)
    for a in range(d):
        lo=list(center); hi=list(center); lo[a]-=1; hi[a]+=1
        j[:,a]+=(field[tuple(hi)]-field[tuple(lo)])/2
    expected=j[0]/np.linalg.norm(j[0])
    np.testing.assert_allclose(out['normals'][(slice(None),*center)],expected,atol=1e-13)


@pytest.mark.parametrize('sigma',[-1,np.nan,np.inf])
def test_invalid_scale(sigma):
    with pytest.raises(ValueError):
        A.Pipeline([A.Elastic(sigma=sigma)])({'image':np.zeros((1,5,7))},seed=0)


def test_zero_scale_and_distribution_reproducibility():
    sample={'image':np.arange(35,dtype=float).reshape(1,5,7)}
    for scale in [0,2.375,(2,8)]:
        pipeline=A.Pipeline([A.Elastic(sigma=scale,magnitude=(.1,.5))])
        a=pipeline(sample,seed=9)['image']; b=pipeline(sample,seed=9)['image']
        np.testing.assert_array_equal(a,b)
        assert not np.array_equal(a,pipeline(sample,seed=10)['image'])
