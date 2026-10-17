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
    host = verify([op],sample,targets)
    from vc_reader.cuda import asarray
    device = A.Pipeline([op],targets=targets,backend="cuda")({k: asarray(v) for k, v in sample.items()},seed=123)
    compare({k: v.numpy() for k, v in device.items()},host)


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


@gpu
def test_cuda_normalization_matches_float64_formula():
    # Values pair around 1500, so every statistic is exact in any reduction
    # order. Outputs must equal float32((x - center) / divisor) bit for bit.
    from vc_reader.cuda import asarray
    rng = np.random.default_rng(17)
    v = rng.integers(1, 1500, 200_000).astype(np.float64)
    x = np.concatenate([v, 3000 - v, np.zeros(1000)])
    x = rng.permutation(x).astype(np.float32).reshape(1, 8, 125, 401)
    nonzero = x[x != 0].astype(np.float64)
    center = nonzero.sum() / nonzero.size
    deviation = np.sqrt(((nonzero - center) ** 2).sum() / nonzero.size)
    zscore = np.where(x == 0, 0, (x.astype(np.float64) - center) / deviation).astype(np.float32)
    low, high = float(x.min()), float(x.max())
    minmax = ((x.astype(np.float64) - low) / (high - low)).astype(np.float32)
    for mode, expected in (("zscore", zscore), ("minmax", minmax)):
        pipe = A.Pipeline([A.Normalize(mode)], targets={"image": A.Image()}, compact=False, backend="cuda")
        out = pipe({"image": asarray(x)}, seed=0)["image"].numpy()
        np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))


def _directional_sample(rng, shape=(6, 7, 8)):
    image = rng.normal(size=(1, *shape)).astype(np.float32)
    vectors = rng.normal(size=(3, *shape))
    return dict(
        image=image, mask=(image > 0).astype(np.uint8),
        points=np.array([[1., 2., 3.], [4.5, 5.25, .5], [0., 0., 0.], [5., 6., 7.]]),
        directions=np.array([[1., 0., 0.], [.2, -.4, .9]]),
        flow=vectors, normals=vectors / np.linalg.norm(vectors, axis=0),
        coords=np.indices(shape).astype(np.float64))


DIRECTIONAL_TARGETS = dict(
    image=A.Image(), mask=A.Label(), points=A.Points(order="xyz"),
    directions=A.Vectors(components="zyx", normalize=True), flow=A.VectorField(components="yxz"),
    normals=A.NormalField(), coords=A.CoordinateField(frame="local"))


@gpu
@pytest.mark.parametrize("op", [A.Flip(), A.Rotate90(), A.Transpose(), A.Elastic(2, .8), A.Elastic(2, .8, border=True)])
@pytest.mark.parametrize("seed", [0, 7])
def test_resident_directional_and_sparse_targets(op, seed):
    from vc_reader.cuda import asarray
    sample = _directional_sample(np.random.default_rng(seed))
    if repr(op).startswith("Elastic"):
        sample.pop("directions")  # sparse vectors have no location for a nonlinear map
    targets = {k: v for k, v in DIRECTIONAL_TARGETS.items() if k in sample}
    host = verify([op], sample, targets, seed=seed)
    pipeline = A.Pipeline([op], targets=targets, backend="cuda")
    device = pipeline({k: asarray(v) for k, v in sample.items()}, seed=seed)
    compare({k: v.numpy() for k, v in device.items()}, host)


@gpu
def test_resident_points_outside_modes():
    from vc_reader.cuda import asarray
    image = np.zeros((1, 4, 5, 6), np.float32)
    points = np.array([[0., 0., 0.], [3., 4., 5.], [3.5, 4., 5.], [-.5, 1., 1.]])
    for outside, expected_rows in (("keep", 4), ("drop", 2)):
        targets = dict(image=A.Image(), points=A.Points(outside=outside))
        pipeline = A.Pipeline([A.Flip(axes="z", p_axis=1)], targets=targets, backend="cuda")
        cpu = A.Pipeline([A.Flip(axes="z", p_axis=1)], targets=targets)
        out = pipeline(dict(image=asarray(image), points=asarray(points)), seed=0)["points"]
        ref = cpu(dict(image=image, points=points), seed=0)["points"]
        assert out.shape == (expected_rows, 3)
        np.testing.assert_array_equal(out.numpy(), ref)
    targets = dict(image=A.Image(), points=A.Points(outside="error"))
    pipeline = A.Pipeline([A.Flip(axes="z", p_axis=1)], targets=targets, backend="cuda")
    with pytest.raises(ValueError, match="outside"):
        pipeline(dict(image=asarray(image), points=asarray(points)), seed=0)


@gpu
def test_resident_sparse_vectors_reject_nonlinear():
    from vc_reader.cuda import asarray
    image = np.zeros((1, 6, 6, 6), np.float32)
    targets = dict(image=A.Image(), directions=A.Vectors())
    pipeline = A.Pipeline([A.Elastic(2, 1)], targets=targets, backend="cuda")
    with pytest.raises(ValueError, match="locations"):
        pipeline(dict(image=asarray(image), directions=asarray(np.ones((2, 3)))), seed=0)


def _blobs(rng, shape, labels=3):
    from itertools import product
    volume = np.zeros(shape, np.int16)
    grid = np.indices(shape)
    for label in range(1, labels + 1):
        center = [rng.uniform(0, n) for n in shape]
        radius = rng.uniform(2, max(shape) / 2)
        inside = sum((g - c) ** 2 for g, c in zip(grid, center)) <= radius ** 2
        volume[inside] = label
    volume[tuple(slice(0, 2) for _ in shape)] = -1
    return volume


@gpu
@pytest.mark.parametrize("shape", [(9, 23, 31), (40, 37)])
@pytest.mark.parametrize("options", [dict(), dict(tube=False), dict(open=True, close=False), dict(ignore={"mask": -1})])
def test_medial_surface_matches_cpu(shape, options):
    from vc_reader.cuda import asarray
    rng = np.random.default_rng(len(shape))
    mask = _blobs(rng, shape)[None]
    sample = dict(image=rng.normal(size=mask.shape).astype(np.float32), mask=mask)
    op = A.MedialSurface(**options)
    host = verify([op], sample)
    assert host["mask_skel"].any()
    device = A.Pipeline([op], backend="cuda")({k: asarray(v) for k, v in sample.items()}, seed=0)
    compare({k: v.numpy() for k, v in device.items()}, host)


def _torch_oriented_blur(volume, axis, along, across):
    """vesuvius fiber_trace_3d._anisotropic_blur_3d, verbatim semantics."""
    torch = pytest.importorskip("torch")
    import math
    import torch.nn.functional as F

    def kernel1d(sigma):
        radius = max(1, int(math.ceil(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, dtype=torch.float64)
        k = torch.exp(-0.5 * (coords / sigma) ** 2)
        return k / k.sum().clamp_min(1.0e-12)

    def blur1d(vol, u, sigma):
        if sigma <= 0:
            return vol
        k = kernel1d(sigma); radius = (k.numel() - 1) // 2
        d, h, w = vol.shape
        base = torch.stack(torch.meshgrid(*(torch.arange(n, dtype=torch.float64) for n in (d, h, w)), indexing="ij"), -1)
        u = u / torch.linalg.vector_norm(u).clamp_min(1e-12)
        acc = torch.zeros_like(vol)
        for offset, weight in zip(range(-radius, radius + 1), k):
            q = base + float(offset) * u.view(1, 1, 1, 3)
            grid = torch.stack([q[..., 2] * (2.0 / max(w - 1, 1)) - 1.0, q[..., 1] * (2.0 / max(h - 1, 1)) - 1.0,
                                q[..., 0] * (2.0 / max(d - 1, 1)) - 1.0], -1).unsqueeze(0)
            acc = acc + F.grid_sample(vol.view(1, 1, d, h, w), grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0] * weight
        return acc

    axis = torch.as_tensor(axis, dtype=torch.float64); axis = axis / torch.linalg.vector_norm(axis).clamp_min(1e-12)
    ref = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    if abs(float(axis @ ref)) > 0.9:
        ref = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    a = torch.linalg.cross(axis, ref); a = a / torch.linalg.vector_norm(a)
    b = torch.linalg.cross(axis, a); b = b / torch.linalg.vector_norm(b)
    out = torch.as_tensor(volume, dtype=torch.float64)
    for u, s in ((axis, along), (a, across), (b, across)):
        out = blur1d(out, u, s)
    return out.numpy()


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=gpu)])
@pytest.mark.parametrize("axis", [(1., 0., 0.), (.2, .7, -.4), (0., .05, 1.)])
def test_oriented_blur_matches_vesuvius_reference(backend, axis):
    image = np.random.default_rng(3).normal(size=(1, 9, 11, 13)).astype(np.float32)
    op = A.OrientedBlur(sigma_along=1.7, sigma_across=.6, axis=axis)
    out = A.Pipeline([op], backend=backend)(dict(image=image), seed=0)["image"]
    expected = _torch_oriented_blur(image[0].astype(np.float64), axis, 1.7, .6)
    np.testing.assert_allclose(out[0], expected, rtol=1e-5, atol=1e-5)


@gpu
def test_oriented_blur_random_axis_and_target_axis_parity():
    from vc_reader.cuda import asarray
    rng = np.random.default_rng(5)
    image = rng.normal(size=(2, 8, 9, 10)).astype(np.float32)
    verify([A.OrientedBlur(p_channel=.7)], dict(image=image), seed=11)
    targets = dict(image=A.Image(), fiber=A.Vectors(components="xyz"))
    sample = dict(image=image, fiber=np.array([[.3, -.2, .9]]))
    ops = [A.Flip(axes="zx", p_axis=1), A.OrientedBlur(1.2, .5, axis="fiber")]
    host = verify(ops, sample, targets)
    device = A.Pipeline(ops, targets=targets, backend="cuda")({k: asarray(v) for k, v in sample.items()}, seed=123)
    compare({k: v.numpy() for k, v in device.items()}, host)
    # The axis follows the flipped vector: equal to a fixed blur with that axis after the flip.
    flipped = A.Pipeline(ops[:1], targets=targets)(sample, seed=123)
    z, y, x = flipped["fiber"][0][::-1]
    fixed = A.Pipeline([A.OrientedBlur(1.2, .5, axis=(z, y, x))])(dict(image=flipped["image"]), seed=0)["image"]
    np.testing.assert_allclose(host["image"], fixed, rtol=1e-6, atol=1e-6)


def test_oriented_blur_validation():
    with pytest.raises(ValueError, match="axis"):
        A.OrientedBlur(axis=(0, 0, 0))
    with pytest.raises(ValueError, match="3D"):
        A.Pipeline([A.OrientedBlur()])(dict(image=np.zeros((1, 4, 5), np.float32)), seed=0)


@pytest.mark.parametrize("backend", ["cpu", pytest.param("cuda", marks=gpu)])
def test_gaussian_blur_nearest_boundary(backend):
    image = np.random.default_rng(8).normal(size=(1, 6, 7, 8))
    sigma, truncate = .9, 6.
    out = A.Pipeline([A.GaussianBlur(sigma, truncate=truncate, boundary="nearest")], backend=backend)(dict(image=image), seed=0)["image"]
    # Full-width truncation as documented; edge samples replicate.
    extent = sigma * truncate + .5; size = int(np.round(extent))
    if size % 2 == 0:
        size += 1 if extent - size >= 0 else -1
    radius = max(0, (size - 1) // 2)
    taps = np.exp(-.5 * (np.arange(-radius, radius + 1) / sigma) ** 2); taps /= taps.sum()
    expected = image[0]
    for axis in range(3):
        padded = np.pad(expected, [(radius, radius) if a == axis else (0, 0) for a in range(3)], mode="edge")
        expected = sum(w * np.take(padded, np.arange(k, k + expected.shape[axis]), axis=axis) for k, w in enumerate(taps))
    np.testing.assert_allclose(out[0], expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="boundary"):
        A.GaussianBlur(boundary="wrap")
