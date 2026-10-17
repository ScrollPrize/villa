"""Normalization math, data-dependent replay, and resident CPU/CUDA chains."""
import pickle

import numpy as np
import pytest

from vc_reader import augment as A
from vc_reader.augment.cuda import is_available


@pytest.fixture(params=["cpu", "cuda"])
def backend(request):
    if request.param == "cuda" and not is_available():
        pytest.skip("CUDA build/device unavailable")
    return request.param


def reference(image, mode):
    out = np.zeros_like(image)
    for source, dest in zip(image, out):
        values = source.astype(np.float64)
        mask = values != 0 if mode == "zscore" else np.ones(values.shape, dtype=bool)
        selected = values[mask]
        if not selected.size or selected.min() == selected.max():
            continue
        if mode == "minmax":
            scale = 2. if selected.max() > 0 and selected.min() < -np.finfo(float).max+selected.max() else 1.
            scaled = selected/scale
            dest[mask] = (scaled-scaled.min())/(scaled.max()-scaled.min())
        else:
            magnitude = np.max(np.abs(selected))
            scaled = selected/magnitude if magnitude > 1e150 or magnitude < 1e-150 else selected
            dest[mask] = (scaled-scaled.mean())/scaled.std(ddof=0)
    return out


@pytest.mark.parametrize("mode", ["minmax", "zscore"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(2,7,9), (2,5,7,9)])
def test_normalization_reference_and_replay(backend, mode, dtype, shape):
    image = np.random.default_rng(123).normal(3,2,shape).astype(dtype)
    image.ravel()[::5] = 0
    sample = dict(image=image, mask=(image > 4).astype(np.int16))
    original = image.copy()
    pipeline = A.Pipeline([A.Normalize(mode)],backend=backend)
    actual = pipeline(sample,seed=17)
    tol = 3e-7 if dtype == np.float32 else 2e-12
    np.testing.assert_allclose(actual["image"],reference(image,mode),rtol=tol,atol=tol)
    np.testing.assert_array_equal(actual["mask"],sample["mask"])
    np.testing.assert_array_equal(image,original)
    assert actual["image"].dtype == dtype
    assert not np.shares_memory(actual["image"],image)
    plan = pickle.loads(pickle.dumps(pipeline.sample(sample,seed=17)))
    np.testing.assert_array_equal(plan.apply(sample)["image"],actual["image"])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(pipeline))(sample,seed=17)["image"],actual["image"])
    changed = dict(sample,image=image*image)
    np.testing.assert_array_equal(plan.apply(changed)["image"],pipeline(changed,seed=17)["image"])


@pytest.mark.parametrize("mode", ["minmax", "zscore"])
@pytest.mark.parametrize("values", [
    [0.,0.,0.,0.], [7.,7.,7.,7.], [0.,7.,7.,0.], [-0.,0.,-0.,0.],
    [-1e308,1e308,0.,5e307], [1e-310,2e-310,0.,3e-310],
    [1e12,1e12+.25,0.,1e12+.5],
])
def test_constants_and_extreme_float64(backend,mode,values):
    image = np.array(values).reshape(1,2,2)
    actual = A.Pipeline([A.Normalize(mode)],backend=backend)(dict(image=image),seed=0)["image"]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual,reference(image,mode),rtol=2e-12,atol=2e-12)
    if mode == "zscore":
        assert np.all(actual[image == 0] == 0)


def test_population_variance_and_large_reductions(backend):
    image = (np.arange(65*129*129)%200).astype(np.float64).reshape(1,65,129,129)
    pipeline = A.Pipeline([A.Normalize()],backend=backend)
    actual = pipeline(dict(image=image),seed=5)["image"]
    selected = actual[image != 0]
    assert abs(selected.mean()) < 2e-12
    assert abs(selected.std(ddof=0)-1) < 2e-12
    assert np.all(actual[image == 0] == 0)
    np.testing.assert_array_equal(actual,pipeline(dict(image=image),seed=5)["image"])


def test_normalization_channels_targets_and_validity(backend):
    image = np.arange(120,dtype=np.float32).reshape(2,3,4,5)
    valid = np.ones((1,3,4,5),np.uint8); valid[:,:,0,:] = 0
    sample = dict(image=image,other=image+1,padding_mask=valid)
    targets = dict(image=A.Image(fill=-2),other=A.Image(valid=""),padding_mask=A.ValidMask())
    pipeline = A.Pipeline([A.Normalize("minmax",on="image",channels=(1,))],targets=targets,backend=backend)
    actual = pipeline(sample,seed=0)
    expected = image.copy(); expected[1:] = reference(image[1:],"minmax")
    expected[:, :,0,:] = -2
    np.testing.assert_allclose(actual["image"],expected,rtol=2e-7,atol=2e-7)
    np.testing.assert_array_equal(actual["other"],sample["other"])


def test_normalization_in_chain_and_compact_boundary(backend):
    image = (np.arange(315)%256).astype(np.float32).reshape(1,5,7,9)
    transforms = [A.MedianFilter(3),A.Gamma(.8),A.Normalize("minmax"),
                  A.GaussianNoise(.1),A.Normalize("zscore"),A.BrightnessMultiply(.5)]
    expected = A.Pipeline(transforms,compact=False)(dict(image=image),seed=42)
    pipeline = A.Pipeline(transforms,backend=backend)
    actual = pipeline(dict(image=image.copy()),seed=42,inplace=True)
    np.testing.assert_allclose(actual["image"],expected["image"],rtol=2e-6,atol=3e-7)


@pytest.mark.parametrize("value", [np.nan,np.inf,-np.inf])
def test_nonfinite_rejected(backend,value):
    image = np.array([0,1,value,3],np.float64).reshape(1,2,2)
    with pytest.raises(ValueError,match="finite"):
        A.Pipeline([A.Normalize()],backend=backend)(dict(image=image),seed=0)


def test_nonfinite_from_previous_step_rejected(backend):
    image = np.full((1,3,4,5),1e38,np.float32)
    with pytest.raises(ValueError,match="finite"):
        A.Pipeline([A.BrightnessMultiply(100),A.Normalize()],backend=backend)(dict(image=image),seed=0)


def test_validation_and_skipping(backend):
    with pytest.raises(ValueError,match="mode"):
        A.Normalize("unknown")
    image = np.arange(12,dtype=np.float32).reshape(1,3,4)
    for transform in (A.Normalize(p=0),A.Normalize(p_channel=0)):
        np.testing.assert_array_equal(A.Pipeline([transform],backend=backend)(dict(image=image),seed=1)["image"],image)
    with pytest.raises(ValueError,match="float"):
        A.Pipeline([A.Normalize()],backend=backend)(dict(image=image.astype(np.uint8)),seed=1)
