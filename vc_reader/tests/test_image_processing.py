"""Numerical and gradient contracts for native image processing and mesh labels."""
import numpy as np
import pytest

from build_support import load_build
load_build()

import vc_reader.transforms as A
from vc_reader.geometry import rasterize_mesh_slice, transform_points


def _backends():
    return ["cpu", "cuda"] if A._transforms.cuda_available() else ["cpu"]


def _torch_tensor_reference(image, sigma, rho):
    """Full convolutions provide an independent reference for separable native passes."""
    import torch
    import torch.nn.functional as F
    dim = image.ndim - 2
    conv = F.conv2d if dim == 2 else F.conv3d
    def smooth(value, width):
        if width <= 0:
            return value
        radius = int(3 * width)
        x = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
        weights = torch.exp(-x.square() / (2 * width * width))
        weights = weights / weights.sum()
        kernel = weights[:, None] * weights[None, :]
        if dim == 3:
            kernel = kernel[:, :, None] * weights[None, None, :]
        kernel = kernel[None, None].expand(value.shape[1], 1, *kernel.shape)
        return conv(value, kernel, padding=radius, groups=value.shape[1])
    image = smooth(image, sigma)
    d = image.new_tensor([2, 1, -16, -27, 0, 27, 16, -1, -2]) / 96
    s = image.new_tensor([1, 4, 6, 4, 1]) / 16
    gradients = []
    for axis in range(dim):
        taps = [d if a == axis else s for a in range(dim)]
        kernel = taps[0][:, None] * taps[1][None, :]
        if dim == 3:
            kernel = kernel[:, :, None] * taps[2][None, None, :]
        gradients.append(conv(image, kernel[None, None], padding=tuple(len(t)//2 for t in taps)))
    return smooth(torch.cat([gradients[a] * gradients[b] for a in range(dim) for b in range(a, dim)], dim=1), rho)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("sigma,rho", [(0, 0), (.8, 1.1)])
def test_structure_tensor_values_and_backward(dim, sigma, rho):
    torch = pytest.importorskip("torch")
    from vc_reader.torch import structure_tensor
    for device in _backends():
        image = torch.tensor(np.random.default_rng(7).normal(size=(2, 1) + (7,)*dim), dtype=torch.float32, device=device, requires_grad=True)
        reference = _torch_tensor_reference(image, sigma, rho)
        actual = structure_tensor(image, sigma=sigma, rho=rho)
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
        cotangent = torch.tensor(np.random.default_rng(3).normal(size=actual.shape), dtype=image.dtype, device=device)
        expected_grad, = torch.autograd.grad(reference, image, cotangent)
        actual_grad, = torch.autograd.grad(actual, image, cotangent)
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-6, rtol=3e-5)
        assert actual_grad.abs().sum() > 0


def test_structure_tensor_gradcheck():
    torch = pytest.importorskip("torch")
    from vc_reader.torch import structure_tensor
    image = torch.tensor(np.random.default_rng(9).normal(size=(1, 1, 3, 4)), dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: structure_tensor(x, sigma=.4, rho=.5, scale=.375), (image,), fast_mode=True)


@pytest.mark.parametrize("shape", [(9, 11), (7, 9, 11)])
@pytest.mark.parametrize("fill", [0, 1, None])
def test_distance_transform_scipy_boundaries(shape, fill):
    scipy = pytest.importorskip("scipy.ndimage")
    mask = np.random.default_rng(42).integers(0, 2, size=shape, dtype=np.uint8) if fill is None else np.full(shape, fill, np.uint8)
    expected = scipy.distance_transform_edt(1-mask) - scipy.distance_transform_edt(mask)
    for backend in _backends():
        pipeline = A.Pipeline([A.DistanceTransform(implicit_background=True)], targets={"label": A.Label()}, backend=backend)
        actual = pipeline({"label": mask[None]}, seed=0)["label_sdt"][0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def _ridge_reference(image):
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(image, 2)
    smoothed = (smoothed-smoothed.min()) / (smoothed.max()-smoothed.min())
    d = image.ndim
    h = np.zeros((*image.shape, d, d), dtype=np.float64)
    for q in range(d):
        first = np.gradient(smoothed, axis=q, edge_order=2)
        for p in range(q+1):
            h[..., p, q] = h[..., q, p] = np.gradient(first, axis=p, edge_order=2)
    zero = np.trace(h, axis1=-2, axis2=-1) == 0
    values = np.linalg.eigvalsh(h * 36)
    values = np.take_along_axis(values, np.argsort(np.abs(values), axis=-1), axis=-1)
    values[zero] = 0
    largest = np.abs(values[..., -1])
    second = np.abs(values[..., -2])
    denominator = largest if d == 2 else np.sqrt(second*largest)
    rb = np.abs(values[..., 0]) / np.where(denominator == 0, 1e-10, denominator)
    ra = second / np.where(largest == 0, 1e-10, largest)
    response = (1-np.exp(-.5*np.sum(values**2, axis=-1)/1.5**2))*np.exp(-.5*rb**2/.5**2)
    if d == 3:
        response *= np.exp(-.5*ra**2/.5**2)
    response[values[..., -1] > 0] = 0
    return response


@pytest.mark.parametrize("shape", [(17, 19), (13, 15, 17)])
def test_ridges_match_numpy_reference(shape):
    pytest.importorskip("scipy.ndimage")
    image = np.random.default_rng(123).random(shape, dtype=np.float32)
    expected = _ridge_reference(image)
    for backend in _backends():
        pipeline = A.Pipeline([A.GaussianBlur(2, boundary="reflect", truncate=4, full_width=False), A.Normalize("minmax"), A.Ridges()], backend=backend)
        actual = pipeline({"image": image[None]}, seed=0)["image"][0]
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-4)
        np.testing.assert_array_equal(actual > .5, expected > .5)


def test_mesh_slice_order_frames_and_invalid_indices():
    vertices = np.array([[1, 1, -1], [1, 5, 1], [5, 1, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)
    labels = np.array([2, 9], dtype=np.uint16)
    attributes = dict(vertex_normals=np.tile([0, 0, 1], (3, 1)).astype(np.float32),
                      face_normals=np.tile([0, 0, 1], (2, 1)).astype(np.float32), vertex_normal_flags=np.ones(2, bool),
                      vertex_tangents=np.tile([1, 0, 0], (3, 1)).astype(np.float32), vertex_bitangents=np.tile([0, 1, 0], (3, 1)).astype(np.float32),
                      face_tangents=np.tile([1, 0, 0], (2, 1)).astype(np.float32), face_bitangents=np.tile([0, 1, 0], (2, 1)).astype(np.float32), vertex_tangent_flags=np.ones(2, bool))
    cpu = rasterize_mesh_slice(vertices, faces, labels, 0, (7, 7), **attributes)
    expected = np.zeros((7, 7), dtype=np.uint16)
    expected[3, 1] = expected[2, 2] = expected[1, 3] = 9
    np.testing.assert_array_equal(cpu[0], expected)
    np.testing.assert_allclose(cpu[1][expected > 0], np.tile([0, 0, 1], (3, 1)))
    np.testing.assert_allclose(cpu[2][expected > 0], np.tile(np.eye(3), (3, 1, 1)))
    if "cuda" in _backends():
        from vc_reader.cuda import asarray
        gpu = rasterize_mesh_slice(asarray(vertices), asarray(faces), asarray(labels), 0, (7, 7), **{k: asarray(v) for k, v in attributes.items()})
        for actual, reference in zip(gpu, cpu):
            np.testing.assert_allclose(actual.numpy(), reference, atol=1e-6)
    with pytest.raises(ValueError, match="must be matrices"):
        rasterize_mesh_slice(np.array(1, np.float32), faces, labels, 0, (7, 7))
    with pytest.raises(ValueError, match="invalid mesh"):
        rasterize_mesh_slice(vertices, np.array([[0, 1, 8]], np.int32), labels[:1], 0, (7, 7))


def test_transform_points_normalizes_vectors():
    points = np.array([[1, 1, 0], [0, 0, 0]], np.float32)
    affine = np.array([[2, 0, 0, 99], [0, 1, 0, 99], [0, 0, 1, 99]], np.float32)
    result = transform_points(points, affine, vectors=True, normalize=True)
    np.testing.assert_allclose(result[0], [2/np.sqrt(5), 1/np.sqrt(5), 0], atol=1e-7)
    np.testing.assert_array_equal(result[1], 0)


def test_real_scroll_structure_and_normalization():
    from pathlib import Path
    torch = pytest.importorskip("torch")
    from vc_reader.torch import structure_tensor
    fixture = Path(__file__).resolve().parents[2] / "vesuvius/tests/ink_detection/fixtures/ink_roi_case.npz"
    if not fixture.exists():
        pytest.skip("monorepo real-scroll fixture required")
    with np.load(fixture) as data:
        crop = data["ct"][4:13, 24:48, 24:48].astype(np.float32)
    reference = (crop-crop.mean(dtype=np.float64))/crop.std(dtype=np.float64)
    for backend in _backends():
        actual = A.Pipeline([A.Normalize("zscore", nonzero=False, eps=1e-8)], backend=backend)({"image": crop[None]}, seed=0)["image"][0]
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-6)
        image = torch.from_numpy(crop)[None, None].to(backend).requires_grad_()
        expected = _torch_tensor_reference(image, .8, 1.1)
        result = structure_tensor(image, sigma=.8, rho=1.1)
        torch.testing.assert_close(result, expected, atol=.003, rtol=3e-5)
        expected_grad, = torch.autograd.grad(expected.square().mean(), image)
        actual_grad, = torch.autograd.grad(result.square().mean(), image)
        torch.testing.assert_close(actual_grad, expected_grad, atol=.001, rtol=1e-4)
