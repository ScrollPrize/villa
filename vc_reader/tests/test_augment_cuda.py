"""CPU/CUDA semantic parity; GPU tests skip cleanly on CPU installations."""
import importlib
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest

from vc_reader import augment as A
from vc_reader.augment.cuda import is_available


def benchmark_cases():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("bench_augment").cases((7, 7, 7))


CASES = benchmark_cases()
gpu = pytest.mark.skipif(not is_available(), reason="CUDA build/device unavailable")


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    for key, ref in expected.items():
        if not isinstance(ref, np.ndarray):
            assert actual[key] == ref
            continue
        out = actual[key]
        assert out.dtype == ref.dtype and out.shape == ref.shape
        assert np.isfinite(out).all()
        if ref.dtype.kind == "f":
            # CUDA libdevice transcendentals can round differently than libm.
            atol, rtol = (3e-7, 2e-6) if ref.dtype == np.float32 else (2e-13, 2e-12)
            np.testing.assert_allclose(out, ref, atol=atol, rtol=rtol)
        else:
            np.testing.assert_array_equal(out, ref)


def verify(transforms, sample, targets=None, seed=123):
    cpu = A.Pipeline(transforms, targets=targets, compact=False)
    cuda = A.Pipeline(transforms, targets=targets, backend="cuda")
    original = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sample.items()}
    expected, actual = cpu(sample, seed=seed), cuda(sample, seed=seed)
    compare(actual, expected)
    replay = cuda.sample(sample, seed=seed)
    for repeated in (cuda(sample, seed=seed), replay.apply(sample), pickle.loads(pickle.dumps(replay)).apply(sample)):
        for key, value in actual.items():
            if isinstance(value, np.ndarray):
                assert value.tobytes() == repeated[key].tobytes()
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            assert sample[key].tobytes() == value.tobytes()
            assert not np.shares_memory(actual[key], sample[key])
    return actual


def test_backend_selection():
    assert A.Pipeline([]).backend == "cpu"
    automatic = A.Pipeline([], backend="auto")
    assert automatic.backend == ("cuda" if is_available() else "cpu")
    assert pickle.loads(pickle.dumps(automatic)).backend == automatic.backend
    with pytest.raises(ValueError, match="backend"):
        A.Pipeline([], backend="typo")
    if not is_available():
        with pytest.raises(ValueError, match="unavailable"):
            A.Pipeline([], backend="cuda")
    else:
        with pytest.raises(ValueError, match="not supported"):
            A.Pipeline([A.ClipOutliers()], backend="cuda")


@gpu
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_benchmark_cases(case, dtype):
    rng = np.random.default_rng(235)
    image = rng.uniform(-.3, 1.3, (2, 7, 7, 7)).astype(dtype)
    sample = dict(image=image, metadata="kept")
    if case.inputs == "both":
        sample["mask"] = rng.integers(-3, 6, image.shape, dtype=np.int16)
    if case.inputs == "binary":
        sample["image"] = (image > .5).astype(dtype)
    verify(case.native, sample)


@gpu
@pytest.mark.parametrize("shape", [(2, 1, 7), (1, 3, 5), (2, 1, 2, 5), (1, 3, 4, 5)])
@pytest.mark.parametrize("size", [1, 2, 3, 4, 7])
def test_median_boundaries(shape, size):
    image = np.random.default_rng(44).normal(size=shape).astype(np.float32)
    out = verify([A.MedianFilter(size)], dict(image=image))
    cpu = A.Pipeline([A.MedianFilter(size)], compact=False)(dict(image=image), seed=123)
    np.testing.assert_array_equal(out["image"], cpu["image"])


@gpu
@pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int8, np.int16, np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.parametrize("transform", [A.Flip("x",p_axis=1), A.Transpose("zxy"), A.Rotate90(("yx",),(1,),resize=True), A.Elastic(1, .5), A.ThickSlice(.5,axes="x")])
def test_label_dtypes_and_validity(dtype, transform):
    rng = np.random.default_rng(234)
    image = rng.uniform(size=(2,3,4,5)).astype(np.float32)
    mask = (np.arange(60).reshape(1,3,4,5)%5).astype(dtype)
    if np.dtype(dtype).itemsize == 8:
        mask += np.array(2**60, dtype=dtype)  # No float conversion of IDs.
    valid = (rng.uniform(size=(1,3,4,5)) > .3).astype(np.uint8)
    verify([transform], dict(image=image,mask=mask,padding_mask=valid))


@gpu
@pytest.mark.parametrize("transform", [
    A.Gamma(.7,p_invert=1,p_retain_stats=1),
    A.GaussianBlur((.1,2),shared_axes=False),
    A.GaussianBlur(1,boundary="reflect"),
    A.MedianFilter(3,quantize=True),
    A.Sharpen(.5,p_clamp=1),
    A.BrightnessGradient(2,loc=.5,strength=.4,clip=True),
    A.BlankRectangle(3,value="mean",count=3),
    A.BlankRectangle(3,value="median",count=3),
    A.Smear(axis="y",shift=(-2,),previous=2),
    A.LocalSmoothing(2,loc=.5,strength=.6,sigma=1),
    A.LowResolution(.4,axes="yx",shared_axes=False),
    A.RandomBinaryOperator(channels=(0,1),radius=2,operations=("close",)),
    A.OneOf([A.Gamma(.8),A.GaussianBlur(1)]),
])
def test_2d_variants(transform):
    image = np.random.default_rng(45).uniform(size=(2,5,7)).astype(np.float32)
    verify([transform],dict(image=image))


@gpu
def test_chain_and_inplace():
    image = np.random.default_rng(234).uniform(size=(2,7,7,7)).astype(np.float32)
    transforms = [A.GaussianNoise(.02,shared_noise=True),A.MedianFilter(3),A.Elastic(1,.5),A.LocalGamma(2,loc=.5,gamma=.8),A.Flip("x",p_axis=1)]
    verify(transforms,dict(image=image))
    pipeline = A.Pipeline(transforms,backend="cuda")
    expected = pipeline(dict(image=image),seed=3)
    actual = pipeline(dict(image=image.copy()),seed=3,inplace=True,profile=True)
    assert len(actual.pop("_aug_perf")) == len(transforms)
    compare(actual,expected)


@gpu
@pytest.mark.parametrize("image", [np.zeros((1,3,3,3),np.float32), np.full((1,3,3,3),1e37,np.float32)])
def test_constant_and_wide_intermediates(image):
    for op in [A.GaussianBlur(1),A.LocalSmoothing(2),A.Sharpen(.1),A.Gamma(.8,p_retain_stats=1),A.MedianFilter(3)]:
        verify([op],dict(image=image))


@gpu
@pytest.mark.parametrize("op", [A.Elastic(1,.8,border=True), A.Elastic(1,.8), A.Transpose("yx")])
def test_2d_geometry_and_targets(op):
    image = np.random.default_rng(24).normal(size=(1,5,7)).astype(np.float64)
    coords = np.indices((5,7)).astype(np.float64)
    sample = dict(image=image, mask=(image>0).astype(np.int8), padding_mask=(image>-.5),
                  points=np.array([[2.,3.],[1.,1.]]), normals=np.ones((2,5,7)), coords=coords)
    targets = dict(image=A.Image(fill=-1), mask=A.Label(ignore_value=-2), padding_mask=A.ValidMask(),
                   points=A.Points(order="yx"),normals=A.NormalField(components="yx"),
                   coords=A.CoordinateField(components="yx",frame="local"))
    verify([op],sample,targets)


@gpu
@pytest.mark.parametrize("seed", [0,1,2,2**63+9])
def test_probabilities_and_channel_selection(seed):
    image = np.random.default_rng(23).uniform(size=(3,4,5,6)).astype(np.float32)
    verify([A.OneOf([A.Gamma((.7,1.2),p=.8,channels=(0,2)),
                      A.GaussianNoise((.01,.1),shared_noise=True,p_channel=.6)],p=.7),
            A.GaussianBlur((.3,1),shared=True,shared_axes=False,channels=(1,))],
           dict(image=image),seed=seed)


@gpu
def test_noncontiguous_and_invalid_inputs():
    image = np.arange(60,dtype=np.float32).reshape(1,3,4,5)
    verify([A.MedianFilter(3)],dict(image=image[:,:,:,::-1]))
    pipe = A.Pipeline([A.MedianFilter(3)],backend="cuda")
    with pytest.raises(ValueError,match="contiguous"):
        pipe(dict(image=image[:,:,:,::-1]),seed=1,inplace=True)
    image[0,0,0,0] = np.nan
    with pytest.raises(ValueError,match="nonfinite"):
        pipe(dict(image=image),seed=1)


@gpu
@pytest.mark.parametrize("seed", [123,124,125])
def test_elastic_dense_validity_support(seed):
    image = np.random.default_rng(32).normal(size=(1,16,16,16)).astype(np.float32)
    verify([A.Elastic(3,1)],dict(image=image,padding_mask=np.ones_like(image,dtype=np.uint8)),seed=seed)


@gpu
def test_resident_shape_changes_and_data_dependent_statistics():
    image = np.random.default_rng(32).uniform(size=(2,3,5,7)).astype(np.float32)
    verify([A.Transpose("xyz"),A.Rotate90(("yx",),(1,),resize=True),
            A.BrightnessAdd(.4),A.Gamma(.8,p_retain_stats=1),A.Contrast(1.2),
            A.BlankRectangle(2,value="mean",count=2),A.LocalContrast(2,factor=.8),
            A.LowResolution(.5),A.RandomBinaryOperator(channels=(0,1),operations=("open",))],
           dict(image=image))


@gpu
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_binary_operator_preserves_label_value_validation(dtype):
    image = np.zeros((1,3,4,5),dtype=np.float32)
    mask = np.full(image.shape,2**60,dtype=dtype)
    for backend in ("cpu","cuda"):
        pipeline = A.Pipeline([A.RandomBinaryOperator(on="mask",radius=1)],backend=backend)
        with pytest.raises(ValueError,match="value-based label transforms"):
            pipeline(dict(image=image,mask=mask),seed=123)


@gpu
def test_resident_invocations_have_independent_buffers():
    from concurrent.futures import ThreadPoolExecutor
    sample = dict(image=np.random.default_rng(23).uniform(size=(2,7,9,11)).astype(np.float32))
    pipeline = A.Pipeline([A.Elastic(2,.5),A.LocalGamma(2),A.MedianFilter(3)],backend="cuda")
    seeds = list(range(8))
    expected = [pipeline(sample,seed=seed) for seed in seeds]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda seed: pipeline(sample,seed=seed),seeds))
    for out, ref in zip(actual,expected):
        np.testing.assert_array_equal(out["image"],ref["image"])
