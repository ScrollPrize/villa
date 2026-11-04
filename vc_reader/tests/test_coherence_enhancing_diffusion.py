"""Independent CED reference, conservation, and CPU/CUDA pipeline contracts."""
from pathlib import Path
import pickle

import numpy as np
import pytest
from scipy.ndimage import correlate1d

from build_support import load_build
load_build()
from vc_reader import transforms as A


@pytest.fixture(params=["cpu", "cuda"])
def backend(request):
    if request.param == "cuda" and not A._transforms.cuda_available():
        pytest.skip("CUDA device required")
    return request.param


def reference(image, *, sigma=1, rho=2, lambda_=.01, exponent=2, alpha=.01,
              step_size=.1, num_steps=10):
    """SciPy correlations, NumPy eigh, and explicit flux/divergence on one channel."""
    value = image.astype(np.float64)
    d = value.ndim
    def smooth(v, width):
        if width == 0:
            return v.copy()
        radius = int(3*width)
        weights = np.exp(-np.arange(-radius, radius+1)**2/(2*width**2))
        weights /= weights.sum()
        for axis in range(d):
            v = correlate1d(v, weights, axis=axis, mode="nearest")
        return v
    derivative = np.array([2, 1, -16, -27, 0, 27, 16, -1, -2])/96
    smoothing = np.array([1, 4, 6, 4, 1])/16
    for _ in range(num_steps):
        base = smooth(value, sigma)
        gradients = []
        for a in range(d):
            g = base
            for b in range(d):
                g = correlate1d(g, derivative if a == b else smoothing, axis=b, mode="nearest")
            gradients.append(g)
        tensor = np.empty((*value.shape, d, d))
        for a in range(d):
            for b in range(d):
                tensor[..., a, b] = smooth(gradients[a]*gradients[b], rho)
        eigenvalues, vectors = np.linalg.eigh(tensor)
        gap = eigenvalues[..., -1:] - eigenvalues
        rates = np.full_like(gap, alpha)
        positive = gap > 0
        rates[positive] += (1-alpha)*np.exp(-(lambda_/gap[positive])**exponent)
        tensor = np.einsum("...ij,...j,...kj->...ik", vectors, rates, vectors)
        forward = np.stack([np.diff(value, axis=a, append=np.take(value, [-1], axis=a)) for a in range(d)], axis=-1)
        flux = np.einsum("...ij,...j->...i", tensor, forward)
        divergence = np.zeros_like(value)
        for a in range(d):
            f = flux[..., a].copy()
            edge = [slice(None)]*d
            edge[a] = -1
            f[tuple(edge)] = 0
            divergence += np.diff(f, axis=a, prepend=np.zeros_like(np.take(f, [0], axis=a)))
        value += step_size*divergence
    return value


@pytest.mark.parametrize("shape", [(7, 9), (5, 7, 9), (1, 4, 6), (1, 1)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scales", [(0, 0), (.7, 1.1)])
def test_reference_and_conservation(backend, shape, dtype, scales):
    image = np.random.default_rng(31).normal(size=(1, *shape)).astype(dtype)
    before = image.copy()
    options = dict(sigma=scales[0], rho=scales[1], lambda_=.002, num_steps=3)
    expected = reference(image[0], **options)
    actual = A.Pipeline([A.CoherenceEnhancingDiffusion(**options)], backend=backend)({"image": image}, seed=3)["image"]
    assert actual.shape == image.shape and actual.dtype == image.dtype
    np.testing.assert_array_equal(image, before)
    tol = 3e-6 if dtype == np.float32 else 2e-12
    np.testing.assert_allclose(actual[0], expected, atol=tol, rtol=tol)
    assert abs(actual.sum(dtype=np.float64)-image.sum(dtype=np.float64)) <= tol*image.size
    assert np.linalg.norm(actual) <= np.linalg.norm(image)+tol


@pytest.mark.parametrize("shape", [(1, 7, 9), (1, 5, 7, 9)])
def test_constant_zero_steps_and_isotropic_limit(backend, shape):
    constant = np.full(shape, 2.5, np.float32)
    pipe = A.Pipeline([A.CoherenceEnhancingDiffusion()], backend=backend)
    np.testing.assert_array_equal(pipe({"image": constant}, seed=1)["image"], constant)
    image = np.random.default_rng(3).normal(size=shape).astype(np.float32)
    zero = A.Pipeline([A.CoherenceEnhancingDiffusion(num_steps=0)], backend=backend)
    np.testing.assert_array_equal(zero({"image": image}, seed=1)["image"], image)
    # alpha=1 must reduce to the ordinary no-flux discrete heat equation.
    expected = image.copy()
    for axis in range(1, image.ndim):
        forward = np.diff(image, axis=axis, append=np.take(image, [-1], axis=axis))
        expected += .1*np.diff(forward, axis=axis, prepend=np.zeros_like(np.take(image, [0], axis=axis)))
    isotropic = A.Pipeline([A.CoherenceEnhancingDiffusion(alpha=1, num_steps=1)], backend=backend)
    np.testing.assert_allclose(isotropic({"image": image}, seed=1)["image"], expected, atol=2e-6)


def test_selection_replay_and_labels(backend):
    image = np.random.default_rng(4).uniform(size=(2, 5, 7, 9)).astype(np.float32)
    mask = (image[:1] > .5).astype(np.uint8)
    sample = {"image": image, "mask": mask}
    pipe = A.Pipeline([A.CoherenceEnhancingDiffusion(channels=[1], num_steps=2)], backend=backend)
    plan = pipe.sample(sample, seed=42)
    result = pipe.apply(sample, plan)
    np.testing.assert_array_equal(result["image"][0], image[0])
    np.testing.assert_array_equal(result["mask"], mask)
    assert np.any(result["image"][1] != image[1])
    replay = pickle.loads(pickle.dumps(plan)).apply(sample)
    np.testing.assert_array_equal(result["image"], replay["image"])
    disabled = A.Pipeline([A.CoherenceEnhancingDiffusion(p=0)], backend=backend)
    np.testing.assert_array_equal(disabled(sample, seed=42)["image"], image)


@pytest.mark.parametrize("options", [dict(sigma=-1), dict(rho=float("nan")), dict(rho=1000),
    dict(lambda_=0), dict(lambda_=float("inf")), dict(exponent=0), dict(alpha=0), dict(alpha=1.1),
    dict(step_size=0), dict(step_size=.17), dict(step_size=float("nan")), dict(num_steps=-1)])
def test_invalid_parameters(options):
    with pytest.raises((ValueError, TypeError)):
        A.CoherenceEnhancingDiffusion(**options)


def test_real_ct_crop(backend):
    fixture = Path(__file__).resolve().parents[2]/"vesuvius/tests/ink_detection/fixtures/ink_roi_case.npz"
    if not fixture.exists():
        pytest.skip("monorepo CT fixture required")
    with np.load(fixture) as data:
        image = data["ct"][4:13, 24:48, 24:48].astype(np.float32)[None]
    image /= 255
    options = dict(num_steps=4, lambda_=.0001)
    result = A.Pipeline([A.CoherenceEnhancingDiffusion(**options)], backend=backend)({"image": image}, seed=0)["image"]
    np.testing.assert_allclose(result[0], reference(image[0], **options), atol=3e-6, rtol=3e-6)
    assert np.isfinite(result).all() and np.any(result != image)


def test_cuda_resident_and_batch(backend):
    if backend != "cuda":
        pytest.skip("resident CUDA path")
    torch = pytest.importorskip("torch")
    image = np.random.default_rng(7).normal(size=(2, 1, 5, 7, 9)).astype(np.float32)
    op = A.CoherenceEnhancingDiffusion(num_steps=2)
    cpu = A.Pipeline([op])
    pipe = A.Pipeline([op], backend="cuda")
    expected = np.stack([cpu({"image": v}, seed=1)["image"] for v in image])
    device = torch.from_numpy(image).cuda()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result = pipe.batch({"image": device}, seeds=[1, 1])["image"]
    stream.synchronize()
    assert result.is_cuda
    np.testing.assert_allclose(result.cpu().numpy(), expected, atol=3e-6, rtol=3e-6)
    np.testing.assert_array_equal(device.cpu().numpy(), image)
