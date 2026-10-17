"""Distance transforms and structure tensors against the vesuvius references."""
import numpy as np
import pytest

from vc_reader import augment as A
from vc_reader.augment.cuda import is_available

ndimage = pytest.importorskip("scipy.ndimage")


@pytest.fixture(params=["cpu", "cuda"])
def backend(request):
    if request.param == "cuda" and not is_available():
        pytest.skip("CUDA build/device unavailable")
    return request.param


def run(transforms, sample, backend, targets=None, device=False):
    pipeline = A.Pipeline(transforms, targets=targets, backend=backend)
    if device:
        from vc_reader.cuda import asarray
        out = pipeline({k: asarray(v) for k, v in sample.items()}, seed=0)
        return {k: v.numpy() for k, v in out.items()}
    return pipeline(sample, seed=0)


def blobs(shape, seed=0, count=4):
    rng = np.random.default_rng(seed)
    grid = np.indices(shape)
    labels = np.zeros(shape, np.uint8)
    for _ in range(count):
        center = rng.uniform(0, np.array(shape))
        radius = rng.uniform(2, max(shape) / 3)
        labels[sum((g - c) ** 2 for g, c in zip(grid, center)) <= radius ** 2] = 1
    return labels


# vesuvius image_proc.geometry.structure_tensor.StructureTensorComputer, in float64.
def computer_reference(volume, sigma, rho):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    x = torch.as_tensor(volume, dtype=torch.float64)[None, None]
    d = x.dim() - 2

    def gaussian(s):
        radius = int(3 * s)
        coords = torch.arange(-radius, radius + 1, dtype=torch.float64)
        g = torch.exp(-coords ** 2 / (2 * s * s)); g = g / g.sum()
        k = g
        for _ in range(d - 1):
            k = k[..., None] * g
        return k[None, None], radius

    conv = F.conv3d if d == 3 else F.conv2d
    if sigma > 0:
        k, pad = gaussian(sigma); x = conv(x, k, padding=pad)
    dk = torch.tensor([2., 1., -16., -27., 0., 27., 16., -1., -2.], dtype=torch.float64)
    sk = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float64)
    grads = []
    for axis in range(d):
        k = None
        for b in range(d):
            v = (dk if b == axis else sk).view(*[-1 if i == b else 1 for i in range(d)])
            k = v if k is None else k * v
        k = k / (96 * 16 ** (d - 1))
        grads.append(conv(x, k[None, None], padding=tuple(4 if b == axis else 2 for b in range(d))))
    pairs = [(a, b) for a in range(d) for b in range(a, d)]
    J = torch.cat([grads[a] * grads[b] for a, b in pairs], dim=1)
    if rho > 0:
        k, pad = gaussian(rho)
        J = conv(J, k.expand(J.shape[1], *k.shape[1:]).contiguous(), padding=pad, groups=J.shape[1])
    return J[0].numpy()


# vesuvius neural_tracing.tracers.shared.ct_structure.ct_structure_tensor (zyx order).
def ct_reference(cube, center, derivative_sigma, integration_sigma, spacing=1.):
    image = np.asarray(cube, dtype=np.float64) / 255.
    gradients = []
    for axis in range(3):
        order = [0, 0, 0]; order[axis] = 1
        gradients.append(ndimage.gaussian_filter(image, derivative_sigma / spacing, order=order, mode="nearest", truncate=3.) / spacing)
    weights = [np.exp(-.5 * ((np.arange(n) - c) / (integration_sigma / spacing)) ** 2) for n, c in zip(image.shape, center)]
    weight = weights[0][:, None, None] * weights[1][None, :, None] * weights[2][None, None, :]
    weight /= weight.sum()
    return np.array([[np.sum(weight * a * b) for b in gradients] for a in gradients])


@pytest.mark.parametrize("shape", [(11, 13, 17), (23, 19)])
@pytest.mark.parametrize("mode", ["signed", "inside", "outside"])
def test_distance_transform_matches_scipy(backend, shape, mode):
    labels = blobs(shape)[None]
    out = run([A.DistanceTransform(mode=mode)], dict(image=np.zeros(labels.shape, np.float32), mask=labels), backend)
    inside = ndimage.distance_transform_edt(labels[0])
    outside = ndimage.distance_transform_edt(1 - labels[0])
    expected = {"signed": outside - inside, "inside": inside, "outside": outside}[mode]
    np.testing.assert_allclose(out["mask_sdt"][0], expected, rtol=1e-6, atol=1e-5)
    assert out["mask_sdt"].dtype == np.float32


def test_distance_transform_normals_match_aux_surface_normals(backend):
    skimage = pytest.importorskip("skimage.filters")
    labels = blobs((14, 15, 16), seed=3)[None]
    out = run([A.DistanceTransform(normals="_normals", components="xyz")],
              dict(image=np.zeros(labels.shape, np.float32), mask=labels), backend)
    sdt = ndimage.distance_transform_edt(1 - labels[0]) - ndimage.distance_transform_edt(labels[0])
    gradients = np.stack([skimage.scharr(sdt, axis=2), skimage.scharr(sdt, axis=1), skimage.scharr(sdt, axis=0)])
    expected = gradients / np.maximum(np.sqrt((gradients ** 2).sum(0, keepdims=True)), 1e-6)
    np.testing.assert_allclose(out["mask_normals"], expected, rtol=1e-5, atol=1e-5)


def test_distance_transform_absent_class_and_validation(backend):
    image = np.zeros((1, 4, 5, 6), np.float32)
    for value in (0, 1):
        out = run([A.DistanceTransform()], dict(image=image, mask=np.full((1, 4, 5, 6), value, np.uint8)), backend)
        assert not out["mask_sdt"].any()
    with pytest.raises(ValueError, match="mode"):
        A.DistanceTransform(mode="euclid")


@pytest.mark.parametrize("shape", [(12, 14, 15), (21, 18)])
@pytest.mark.parametrize("sigma,rho", [(1., 1.5), (0., 0.), (.7, 2.)])
def test_pavel_components_match_structure_tensor_computer(backend, shape, sigma, rho):
    image = np.random.default_rng(1).normal(size=(1, *shape)).astype(np.float32)
    out = run([A.StructureTensor(sigma=sigma, rho=rho)], dict(image=image), backend)
    expected = computer_reference(image[0].astype(np.float64), sigma, rho)
    np.testing.assert_allclose(out["image_st"], expected, rtol=2e-5, atol=2e-6)
    assert out["image_st"].shape == expected.shape


@pytest.mark.parametrize("field", ["normal", "tangent"])
def test_eigenvector_fields(backend, field):
    torch = pytest.importorskip("torch")
    labels = blobs((13, 14, 15), seed=5)[None]
    sample = dict(image=np.zeros(labels.shape, np.float32), mask=labels)
    ops = [A.DistanceTransform(), A.StructureTensor(on="mask_sdt", field=field, components="xyz", confidence="gap")]
    out = run(ops, sample, backend)
    sdt = ndimage.distance_transform_edt(1 - labels[0]) - ndimage.distance_transform_edt(labels[0])
    J = computer_reference(sdt, 1., 1.5)
    zz, zy, zx, yy, yx, xx = J
    matrices = torch.as_tensor(np.stack([np.stack([xx, yx, zx], -1), np.stack([yx, yy, zy], -1), np.stack([zx, zy, zz], -1)], -2))
    values, vectors = torch.linalg.eigh(matrices)
    values, vectors = values.numpy(), vectors.numpy()
    column = -1 if field == "normal" else 0
    expected = np.moveaxis(vectors[..., column], -1, 0)
    actual = out["mask_sdt_st"]
    # Compare up to sign where the eigenvalue is well separated.
    gap = (values[..., 2] - values[..., 1]) / np.maximum(values[..., 2], 1e-300) if field == "normal" else \
        (values[..., 1] - values[..., 0]) / np.maximum(values[..., 2], 1e-300)
    clear = (gap > 1e-3) & (values[..., 2] > 1e-6)
    cosine = np.abs((actual * expected).sum(0))
    assert clear.sum() > 100
    np.testing.assert_allclose(cosine[clear], 1, atol=1e-4)
    np.testing.assert_allclose(out["gap"][0][clear], gap[clear], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=0)[clear], 1, atol=1e-6)


def test_point_tensor_matches_ct_structure_tensor(backend):
    cube = np.random.default_rng(9).integers(0, 256, (1, 30, 26, 26)).astype(np.float32)
    center = (11., 12.5, 12.5)
    op = A.StructureTensor(derivative="gaussian", sigma=2., rho=8., scale=1 / 255., at=center, output="tensor")
    out = run([op], dict(image=cube), backend)
    np.testing.assert_allclose(out["tensor"], ct_reference(cube[0], center, 2., 8.), rtol=1e-5, atol=1e-10)


def test_gaussian_dense_components(backend):
    image = np.random.default_rng(2).normal(size=(1, 10, 11, 12))
    op = A.StructureTensor(derivative="gaussian", sigma=1.3, rho=2., truncate=3.)
    out = run([op], dict(image=image), backend)
    grads = [ndimage.gaussian_filter(image[0], 1.3, order=[int(a == b) for b in range(3)], mode="nearest", truncate=3.) for a in range(3)]
    pairs = [(a, b) for a in range(3) for b in range(a, 3)]
    expected = np.stack([ndimage.gaussian_filter(grads[a] * grads[b], 2., mode="nearest", truncate=3.) for a, b in pairs])
    np.testing.assert_allclose(out["image_st"], expected, rtol=1e-9, atol=1e-12)


def test_structure_validation():
    with pytest.raises(ValueError, match="sigma"):
        A.StructureTensor(derivative="gaussian", sigma=0)
    with pytest.raises(ValueError, match="point"):
        A.StructureTensor(at=(1, 2, 3), field="normal")
    with pytest.raises(ValueError, match="field"):
        A.StructureTensor(field="eigen")


def test_resident_device_inputs_match_host_cuda():
    if not is_available():
        pytest.skip("CUDA build/device unavailable")
    labels = blobs((16, 17, 18), seed=4)[None]
    image = np.random.default_rng(6).normal(size=labels.shape).astype(np.float32)
    sample = dict(image=image, mask=labels)
    ops = [A.Flip(), A.DistanceTransform(normals="_normals"),
           A.StructureTensor(on="mask_sdt", field="normal", confidence="gap"),
           A.StructureTensor(on="image", output="tensor", derivative="gaussian", sigma=1.5, rho=4., at=(8, 8.5, 9))]
    host = run(ops, sample, "cuda")
    device = run(ops, sample, "cuda", device=True)
    assert host.keys() == device.keys()
    for key in host:
        np.testing.assert_array_equal(device[key], host[key])
    cpu = run(ops, sample, "cpu")
    for key in ("mask_sdt", "gap", "tensor"):
        np.testing.assert_allclose(device[key], cpu[key], rtol=2e-5, atol=2e-5)
