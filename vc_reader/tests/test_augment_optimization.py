"""Independent references for the shared kernels used by the optimization pass."""
import numpy as np
import pytest

from vc_reader import augment as A


@pytest.mark.parametrize("shape", [(7, 11), (5, 7, 11), (1, 3, 2)])
@pytest.mark.parametrize("radius", [0, 1, 3])
@pytest.mark.parametrize("operation", ["dilate", "erode", "open", "close"])
@pytest.mark.parametrize("neutral", [False, True])
@pytest.mark.parametrize("dtype", [np.int16, np.uint64, np.float64])
def test_morphology_small_and_large_footprints(shape, radius, operation, neutral, dtype):
    ndi = pytest.importorskip("scipy.ndimage")
    labels = np.random.default_rng(19).integers(-1, 4, size=(2, *shape), dtype=np.int16)
    labels = (np.maximum(labels, 0) if dtype == np.uint64 else labels).astype(dtype)
    coords = np.indices((2 * radius + 1,) * len(shape)) - radius
    footprint = (coords * coords).sum(axis=0) <= radius * radius
    expected = []
    for channel in labels:
        binary = channel > 0
        if operation in ("dilate", "close"):
            binary = ndi.binary_dilation(binary, structure=footprint, border_value=0)
        if operation != "dilate":
            binary = ndi.binary_erosion(binary, structure=footprint, border_value=int(neutral))
        if operation == "open":
            binary = ndi.binary_dilation(binary, structure=footprint, border_value=0)
        expected.append(binary * channel.max())
    out = A.Pipeline([A.Morphology(operation, radius=radius, neutral_border=neutral)])(
        {"mask": labels}, seed=1)["mask"]
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == labels.dtype


def test_morphology_preserves_negative_zero():
    labels = np.full((1, 3, 5), -2., dtype=np.float64)
    out = A.Pipeline([A.Morphology("close")])({"mask": labels}, seed=1)["mask"]
    assert (out == 0).all()
    assert np.signbit(out).all()


@pytest.mark.parametrize("size", [2, 3, 4, 7])
def test_median_tracks_large_rank_jumps_and_empty_bins(size):
    ndi = pytest.importorskip("scipy.ndimage")
    image = np.broadcast_to(((np.arange(257) // 3) % 2) * 255., (1, 5, 9, 257)).copy()
    # Include all 256 palette values globally, but leave large gaps in most
    # local histograms. Moving across stripes jumps between the extreme ranks.
    image[0, 0, 0, :256] = np.arange(256)
    expected = ndi.median_filter(image[0], size=size, mode="reflect")
    output = A.Pipeline([A.MedianFilter(size)])({"image": image}, seed=123)["image"]
    np.testing.assert_array_equal(output[0], expected)


@pytest.mark.parametrize("shape", [(9, 11), (7, 9, 11), (1, 3, 2)])
@pytest.mark.parametrize("border", [False, True])
@pytest.mark.parametrize("scale", [.8, 1.2])
def test_affine_interior_and_padding_reference(shape, border, scale):
    ndi = pytest.importorskip("scipy.ndimage")
    image = np.random.default_rng(17).normal(size=(2, *shape))
    center = (np.asarray(shape) - 1) / 2
    matrix = np.eye(len(shape)) / scale
    offset = center - (center + .25) / scale
    expected = np.stack([
        ndi.affine_transform(c, matrix, offset, order=1, prefilter=False,
                             mode="nearest" if border else "grid-constant", cval=-.75)
        for c in image
    ])
    out = A.Pipeline([A.Affine(scale=scale, translate=.25, border=border)],
                     targets={"image": A.Image(fill=-.75)})(
        {"image": image}, seed=1)["image"]
    np.testing.assert_allclose(out, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("shape", [(7, 11), (5, 7, 11)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_local_gamma_separable_field_reference(shape, dtype):
    special = pytest.importorskip("scipy.special")
    image = np.random.default_rng(21).normal(size=(2, *shape)).astype(dtype)
    field = np.ones(shape)
    for axis, extent in enumerate(shape):
        x = np.arange(extent, dtype=np.float64)
        kernel = .5 * (special.erf((x + .5 - .4 * extent) / (2. * np.sqrt(2.)))
                       - special.erf((x - .5 - .4 * extent) / (2. * np.sqrt(2.))))
        broadcast = [1] * len(shape)
        broadcast[axis] = extent
        field *= kernel.reshape(broadcast)
    field = (field - field.min()) / (field.max() - field.min())
    expected = []
    for channel in image.astype(np.float64):
        low, span = channel.min(), channel.max() - channel.min()
        norm = (channel - low) / span
        expected.append(((norm * (1 - field) + norm ** .8 * field) * span + low).astype(dtype))
    out = A.Pipeline([A.LocalGamma(scale=2, loc=.4, gamma=.8)])({"image": image}, seed=1)["image"]
    np.testing.assert_allclose(out, expected, rtol=2e-6 if dtype == np.float32 else 2e-14,
                               atol=2e-6 if dtype == np.float32 else 2e-14)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("count", [1, 3, 200001])
def test_paired_noise_matches_rician_components(dtype, count):
    # Both transforms sample the same channel seed. Rician consumes one normal
    # pair per voxel; Gaussian consumes the same pair on adjacent voxels.
    gaussian = A.Pipeline([A.GaussianNoise(.25)])
    rician = A.Pipeline([A.RicianNoise(.25)])
    components = gaussian({"image": np.zeros((1, 1, 2 * count), dtype)}, seed=17)["image"].ravel()
    sample = {"image": np.zeros((1, 1, count), dtype)}
    actual = rician(sample, seed=17)["image"].ravel()
    expected = np.hypot(components[::2].astype(np.float64), components[1::2])
    np.testing.assert_allclose(actual, expected, rtol=2e-7 if dtype == np.float32 else 3e-15, atol=0)
    odd = gaussian(sample, seed=17)["image"].ravel()
    np.testing.assert_array_equal(odd, components[:count])
    if count > 1000:
        assert abs(components.mean()) < .002
        assert abs(components.std() - .25) < .002
        assert abs(np.corrcoef(components[::2], components[1::2])[0, 1]) < .01
        assert abs(np.mean(actual.astype(np.float64) ** 2) - 2 * .25 ** 2) < .002


@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 3, 7), (4, 5, 8), (3, 4, 17),
                                   (1, 9, 35), (3, 4, 32), (2, 5, 33), (4, 5, 65)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_median3_network_exact_values_boundaries_and_tails(shape, dtype):
    ndi = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(91)
    limit = np.finfo(dtype).max
    for image in (rng.normal(size=(2, *shape)).astype(dtype),
                  rng.choice(np.array([-limit, -1, 0, 1, limit], dtype=dtype), size=(2, *shape))):
        original = image.copy()
        expected = np.stack([ndi.median_filter(c, size=3, mode="reflect") for c in image])
        actual = A.Pipeline([A.MedianFilter(3)])({"image": image}, seed=1)["image"]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(image, original)
        assert actual.dtype == dtype


@pytest.mark.parametrize("shape", [(1, 7), (5, 17), (1, 3, 9), (7, 9, 19)])
@pytest.mark.parametrize("sigma", [.3, 1., 3.])
@pytest.mark.parametrize("boundary", ["reflect", "mirror"])
def test_float32_blur_reference(shape, sigma, boundary):
    ndi = pytest.importorskip("scipy.ndimage")
    image = np.random.default_rng(53).normal(size=(2, *shape)).astype(np.float32)
    expected = np.stack([ndi.gaussian_filter(c.astype(np.float64), sigma,
                         radius=int((6 * sigma + .5) // 2), mode=boundary) for c in image])
    actual = A.Pipeline([A.GaussianBlur(sigma, boundary=boundary)])({"image": image}, seed=1)["image"]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=5e-7)


@pytest.mark.parametrize("transform", [A.GaussianBlur(1), A.LocalSmoothing(2, loc=.4, strength=.7, sigma=1),
                                       A.Sharpen(.25, p_clamp=0), A.Sharpen(1e40, p_clamp=1)])
@pytest.mark.parametrize("scale", [1., np.finfo(np.float32).max / 16])
def test_float32_filters_retain_wide_fallback(transform, scale):
    image = (np.random.default_rng(83).uniform(-1, 1, (1, 3, 7, 19)) * scale).astype(np.float32)
    pipeline = A.Pipeline([transform])
    expected = pipeline({"image": image.astype(np.float64)}, seed=7)["image"].astype(np.float32)
    actual = pipeline({"image": image}, seed=7)["image"]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=scale * 2e-6)


def test_rician_large_finite_float32_does_not_overflow_when_squared():
    image = np.full((1, 3, 7), 1e30, dtype=np.float32)
    actual = A.Pipeline([A.RicianNoise(.1)])({"image": image}, seed=7)["image"]
    np.testing.assert_array_equal(actual, image)


@pytest.mark.parametrize("gamma", [.5, 1., 1.7])
@pytest.mark.parametrize("retain", [0, 1])
@pytest.mark.parametrize("invert", [0, 1])
def test_float32_gamma_preserves_statistics_and_matches_double_reference(gamma, retain, invert):
    image = np.random.default_rng(29).normal(size=(2, 5, 7, 11)).astype(np.float32)
    pipeline = A.Pipeline([A.Gamma(gamma, p_retain_stats=retain, p_invert=invert)])
    actual = pipeline({"image": image}, seed=17)["image"]
    reference = pipeline({"image": image.astype(np.float64)}, seed=17)["image"]
    np.testing.assert_allclose(actual, reference, rtol=3e-6, atol=2e-6)
    if retain:
        np.testing.assert_allclose(actual.mean(axis=(1, 2, 3)), image.mean(axis=(1, 2, 3)), atol=1e-7)
        np.testing.assert_allclose(actual.std(axis=(1, 2, 3)), image.std(axis=(1, 2, 3)), rtol=2e-7)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_median3_signed_zero_ties_use_general_selector(dtype):
    # Promoting exact float values to double retains the same comparison ties.
    image = np.random.default_rng(3).choice([-0., 0., -1., 1.], size=(1, 4, 5, 33)).astype(dtype)
    pipeline = A.Pipeline([A.MedianFilter(3)])
    actual = pipeline({"image": image}, seed=1)["image"]
    reference = pipeline({"image": image.astype(np.float64)}, seed=1)["image"].astype(dtype)
    assert actual.tobytes() == reference.tobytes()


@pytest.mark.parametrize("gamma", [1e-50, .5, 1e40])
def test_float32_gamma_extreme_normalized_values(gamma):
    image = np.array([0, 1e-40, 1e-20, 1, 1e30], dtype=np.float32).reshape(1, 1, 5)
    pipeline = A.Pipeline([A.Gamma(gamma)])
    actual = pipeline({"image": image}, seed=3)["image"]
    expected = pipeline({"image": image.astype(np.float64)}, seed=3)["image"].astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0)


@pytest.mark.parametrize("gamma", [16., 100., 1e8])
def test_float32_gamma_sensitive_base_near_one(gamma):
    # Normalizing across [-1, 1] produces a value halfway between the last two
    # floats below/at one; rounding it to one is unsafe for a large exponent.
    image = np.array([-1, np.nextafter(np.float32(1), np.float32(0)), 1], dtype=np.float32).reshape(1, 1, 3)
    pipeline = A.Pipeline([A.Gamma(gamma)])
    actual = pipeline({"image": image}, seed=3)["image"]
    expected = pipeline({"image": image.astype(np.float64)}, seed=3)["image"].astype(np.float32)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0)
