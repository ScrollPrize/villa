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


# References for the extended modes follow the vesuvius NumPy definitions
# (image_proc.intensity.normalization, ink_detection.data.normalization),
# evaluated in float64.
def _selected(values, valid):
    return values[valid] if valid is not None and valid.any() else values.reshape(-1)


def _robust_reference(values, valid, lower, upper, clip):
    selected = _selected(values, valid)
    std, lo, hi = selected.std(), selected.min(), selected.max()
    if clip:
        lo, hi = np.percentile(selected, lower), np.percentile(selected, upper)
        values = np.clip(values, lo, hi)
        selected = np.clip(selected, lo, hi)
    median = np.median(selected)
    scale = 1.4826 * np.median(np.abs(selected - median))
    if not np.isfinite(scale) or scale < 1e-6:
        candidates = [abs(std), abs(hi - lo) / 2]
        scale = next((c for c in candidates if np.isfinite(c) and c >= 1e-6), 1.)
    return np.nan_to_num((values - median) / scale, nan=0., posinf=0., neginf=0.)


def _percentile_reference(values, valid, lower, upper, median):
    selected = _selected(values, valid)
    lo, hi = np.percentile(selected, lower), np.percentile(selected, upper)
    values = np.clip(values, lo, hi)
    if median:
        scale = .5 * (hi - lo)
        scale = scale if np.isfinite(scale) and scale >= 1e-6 else 1.
        return (values - np.median(np.clip(selected, lo, hi))) / scale
    scale = hi - lo
    if not np.isfinite(scale) or scale < 1e-6:
        return np.zeros_like(values)
    return (values - lo) / scale


def _zscore_reference(values, valid, eps):
    selected = _selected(values, valid)
    if selected.min() == selected.max():
        return np.zeros_like(values)
    return (values - selected.mean()) / max(selected.std(), eps)


def _extended(image, valid, mode, **options):
    out = np.empty_like(image, dtype=np.float64)
    for c, values in enumerate(image.astype(np.float64)):
        v = None if valid is None else valid[0 if len(valid) == 1 else c].astype(bool)
        if mode == "robust":
            out[c] = _robust_reference(values, v, options.get("lower", 1), options.get("upper", 99), options.get("clip", True))
        elif mode == "percentile":
            out[c] = _percentile_reference(values, v, options.get("lower", 1), options.get("upper", 99), options.get("median", False))
        elif mode == "zscore":
            out[c] = _zscore_reference(values, v, options.get("eps", 0))
        else:
            low, high = options.get("bounds", (-np.inf, np.inf))
            out[c] = np.nan_to_num((np.clip(values, low, high) - options.get("center", 0)) / options.get("scale", 1),
                                   nan=0., posinf=0., neginf=0.)
    return out


EXTENDED = [
    ("zscore", dict(nonzero=False)),
    ("zscore", dict(nonzero=False, eps=1e-8)),
    ("robust", {}),
    ("robust", dict(lower=5, upper=95)),
    ("robust", dict(clip=False)),
    ("percentile", dict(lower=0.5, upper=99.5)),
    ("percentile", dict(lower=2, upper=98, median=True)),
    ("percentile", dict(lower=0, upper=100)),
    ("fixed", dict(center=127.5, scale=127.5)),
    ("fixed", dict(center=90.25, scale=31.5, bounds=(12., 230.))),
]


@pytest.mark.parametrize("mode,options", EXTENDED)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("masked", [False, True])
def test_extended_modes_match_vesuvius_references(backend, mode, options, dtype, masked):
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (2, 9, 11, 13)).astype(dtype)
    image[:, :2] = 0  # zeros are ordinary values outside the default z-score
    valid = np.ones((1, 9, 11, 13), np.uint8); valid[:, :, :3] = 0
    targets = dict(image=A.Image(fill=-7), padding_mask=A.ValidMask())
    if mode == "fixed" and masked:
        pytest.skip("fixed normalization has no statistics")
    transform = A.Normalize(mode, masked=masked, **options) if mode != "fixed" else A.Normalize(mode, **options)
    sample = dict(image=image, padding_mask=valid)
    actual = A.Pipeline([transform], targets=targets, backend=backend)(sample, seed=3)["image"]
    expected = _extended(image, valid if masked else None, mode, **options)
    expected[:, valid[0] == 0] = -7
    tol = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
    assert actual.dtype == dtype


@pytest.mark.parametrize("mode", ["zscore", "robust", "percentile"])
def test_masked_statistics_without_valid_voxels_use_every_voxel(backend, mode):
    image = np.random.default_rng(2).normal(5, 3, (1, 6, 7, 8)).astype(np.float32)
    options = dict(nonzero=False) if mode == "zscore" else {}
    targets = dict(image=A.Image(valid=None), padding_mask=A.ValidMask())
    sample = dict(image=image, padding_mask=np.zeros((1, 6, 7, 8), np.uint8))
    masked = A.Pipeline([A.Normalize(mode, masked=True, **options)], targets=targets, backend=backend)(sample, seed=0)
    plain = A.Pipeline([A.Normalize(mode, **options)], targets=targets, backend=backend)(sample, seed=0)
    np.testing.assert_allclose(masked["image"], plain["image"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(plain["image"][0], _extended(image, None, mode, **options)[0], rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("mode", ["robust", "percentile"])
def test_order_statistics_constant_and_tiny_spans(backend, mode):
    for image in (np.full((1, 4, 5), 3., np.float32), np.array([[[0, 1e-7, 2e-7, 0, 1e-7]]], np.float32)):
        actual = A.Pipeline([A.Normalize(mode)], backend=backend)(dict(image=image), seed=0)["image"]
        np.testing.assert_allclose(actual, _extended(image, None, mode), rtol=2e-6, atol=2e-6)


def test_extended_mode_options_are_validated():
    with pytest.raises(ValueError, match="does not apply"):
        A.Normalize("minmax", eps=1)
    with pytest.raises(ValueError, match="does not apply"):
        A.Normalize("fixed", masked=True)
    with pytest.raises(ValueError, match="percentiles"):
        A.Normalize("robust", lower=60, upper=40)
    with pytest.raises(ValueError, match="scale"):
        A.Normalize("fixed", scale=0)
    with pytest.raises(ValueError, match="bounds"):
        A.Normalize("fixed", bounds=(2, 1))


def test_extended_modes_on_resident_device_arrays():
    if not is_available():
        pytest.skip("CUDA build/device unavailable")
    from vc_reader import cuda
    if not cuda.is_available():
        pytest.skip("CUDA reader unavailable")
    image = np.random.default_rng(4).integers(0, 256, (1, 16, 17, 18)).astype(np.float32)
    valid = np.ones((1, 16, 17, 18), np.uint8); valid[..., :4] = 0
    targets = dict(image=A.Image(), padding_mask=A.ValidMask())
    for mode, options in EXTENDED:
        transform = A.Normalize(mode, **options) if mode == "fixed" else A.Normalize(mode, masked=True, **options)
        pipeline = A.Pipeline([transform], targets=targets, backend="cuda")
        host = pipeline(dict(image=image, padding_mask=valid), seed=1)["image"]
        device = pipeline(dict(image=cuda.asarray(image), padding_mask=cuda.asarray(valid)), seed=1)["image"].numpy()
        np.testing.assert_array_equal(device, host)
