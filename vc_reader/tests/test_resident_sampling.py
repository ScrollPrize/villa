"""Resident sampling contracts: coordinates, masks, gradients and CPU/CUDA parity."""
import numpy as np
import pytest
import torch
from build_support import load_build
load_build()
from vc_reader import cuda, tifxyz
from vc_reader.resident import GridField, SurfaceTransfer, project_mask, sample_array, sample_field_tile
from vc_reader.torch import sample_array as torch_sample, surface_points


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('shape', [(7, 9), (5, 7, 9)])
@pytest.mark.parametrize('boundary', ['zeros', 'border'])
def test_dense_values_and_first_gradients(device, shape, boundary):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    rng = np.random.default_rng(12)
    values = torch.tensor(rng.normal(size=(2, 3, *shape)), dtype=torch.float32, device=device, requires_grad=True)
    points = torch.tensor(rng.uniform(-1.5, np.array(shape)+.5, (2, 43, len(shape))), dtype=torch.float32, device=device, requires_grad=True)
    actual = torch_sample(values, points, boundary=boundary)
    scale = torch.tensor(shape, device=device)-1
    grid = (2*points/scale-1).flip(-1).reshape(2, 43, *([1]*(len(shape)-1)), len(shape))
    expected = torch.nn.functional.grid_sample(values, grid, padding_mode=boundary, align_corners=True).reshape(2, 3, 43)
    cot = torch.tensor(rng.normal(size=actual.shape), dtype=torch.float32, device=device)
    grads = torch.autograd.grad(actual, (values, points), cot, retain_graph=True)
    references = torch.autograd.grad(expected, (values, points), cot)
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=2e-5)
    for a, b in zip(grads, references):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=3e-5)


def test_constant_nearest_nan_and_empty_queries():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    points = np.array([[1, 1.5], [-.1, 1], [np.nan, 0], [1e30, 0]], np.float32)
    np.testing.assert_array_equal(sample_array(values, points, order=0, boundary='constant', fill=-7), [6, -7, -7, -7])
    assert sample_array(values, np.empty((0, 2), np.float32)).shape == (0,)
    with pytest.raises(ValueError):
        sample_array(values, points, order=3)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_bicubic_positions_tangents_and_holes(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    r, c = np.mgrid[:8, :9].astype(np.float32)
    xyz = np.stack((c, r, 20+.1*r*r+.3*np.sin(c)), -1)
    valid = np.ones(r.shape, bool); valid[3, 3] = False
    rc = np.array([[1.2, 4.3], [4.2, 5.8], [2.7, 2.2], [-.1, 0], [7, 8]], np.float32)
    expected, supported, col, row = tifxyz.Surface(xyz, mask=valid).points_at(rc[:, :1], rc[:, 1:], tangents=True)
    p, dr, dc, ok = surface_points(torch.tensor(xyz[None], device=device), torch.tensor(valid[None], device=device), torch.tensor(rc[None], device=device))
    np.testing.assert_array_equal(ok.cpu()[0], supported[:, 0])
    keep = supported[:, 0]
    for actual, reference in ((p, expected), (dr, row), (dc, col)):
        np.testing.assert_allclose(actual.cpu()[0][keep], reference[:, 0][keep], atol=3e-5, rtol=2e-5)
        assert (actual.cpu()[0][~keep] == 0).all()


def test_triangle_surface_and_linear_field_border_validity():
    xyz = np.array([[[0, 0, 1], [1, 0, 2]], [[0, 1, 3], [1, 1, 9]]], np.float32)
    field = GridField(xyz, np.ones((2, 2), bool), interpolation='triangles')
    values, valid = field.sample(np.array([[.2, .3], [.8, .7]], np.float32))
    np.testing.assert_allclose(values, [[.3, .2, 1.7], [.7, .8, 5.8]], atol=2e-6)
    assert valid.all()
    out, ok = sample_field_tile(np.array([[1, 2]], np.float32), np.array([[False, True]]), (1, 4), 0, 1, 0, 4)
    np.testing.assert_allclose(out, [[.5, 1.5, 2, 2]])
    np.testing.assert_array_equal(ok, [[False, False, True, True]])


def test_cuda_projection_and_transfer(torch_cuda):
    r, c = np.mgrid[:6, :7].astype(np.float32)
    xyz = np.stack((c, r, 3+.05*r*r), -1)
    mask = np.ones(r.shape, bool)
    normals = np.zeros_like(xyz); normals[..., 2] = 1
    args = (mask, xyz, normals, mask, (0, 0, 0), (10, 10, 10))
    expected = project_mask(*args, half_thickness=2)
    gpu_args = tuple(cuda.asarray(a) for a in args[:4])+args[4:]
    actual = project_mask(*gpu_args, half_thickness=2).numpy()
    np.testing.assert_array_equal(actual, expected)
    transfer = SurfaceTransfer(target_fields=tuple(xyz[..., a] for a in range(3)), target_valid=mask,
        source_fields=tuple(xyz[..., a] for a in range(3)), source_valid=mask,
        uv_rows=r, uv_cols=c, uv_valid=mask, label_shape=(12, 14), output_shape=(12, 14), max_distance=1)
    cpu = transfer.rasterize((0, 12, 0, 14)); gpu = transfer.cuda().rasterize((0, 12, 0, 14))
    for a, b in zip(cpu[:3], gpu[:3]):
        np.testing.assert_allclose(a, b.numpy(), atol=2e-5)
    assert cpu[3] == gpu[3]


def test_cuda_nondefault_stream_and_coordinate_only_gradient(torch_cuda):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        values = torch.arange(125, device='cuda', dtype=torch.float32).reshape(1, 1, 5, 5, 5)
        points = torch.tensor([[[1.25, 2.5, 3.75]]], device='cuda', requires_grad=True)
        result = torch_sample(values, points)
        result.sum().backward()
        torch.testing.assert_close(result, torch.tensor([[[47.5]]], device='cuda'))
        torch.testing.assert_close(points.grad, torch.tensor([[[25., 5., 1.]]], device='cuda'))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compiled_sampling_and_surface_queries(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    # AOT exercises both forward and native backward without graph breaks.
    compiled = torch.compile(torch_sample, backend='aot_eager', fullgraph=True)
    values = torch.arange(125, device=device, dtype=torch.float32).reshape(1, 1, 5, 5, 5).requires_grad_()
    points = torch.tensor([[[1.25, 2.5, 3.75]]], device=device, requires_grad=True)
    result = compiled(values, points)
    result.sum().backward()
    torch.testing.assert_close(points.grad, torch.tensor([[[25., 5., 1.]]], device=device))
    r, c = torch.meshgrid(torch.arange(5., device=device), torch.arange(6., device=device), indexing='ij')
    xyz = torch.stack((c, r, 10+r*r), -1)[None]
    mask = torch.ones(xyz.shape[:-1], device=device, dtype=torch.bool)
    rc = torch.tensor([[[1.25, 2.5]]], device=device)
    compiled_surface = torch.compile(surface_points, backend='aot_eager', fullgraph=True)
    for a, b in zip(compiled_surface(xyz, mask, rc), surface_points(xyz, mask, rc)):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_integer_and_singleton_coordinate_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    values = torch.tensor([[[[2., 5., 9.]]]], device=device, requires_grad=True)
    points = torch.tensor([[[0., 0.], [0., 1.], [0., 2.], [-1., 1.]]], device=device, requires_grad=True)
    out = torch_sample(values, points, boundary='border')
    out.sum().backward()
    torch.testing.assert_close(out, torch.tensor([[[2., 5., 9., 5.]]], device=device))
    torch.testing.assert_close(points.grad, torch.tensor([[[0., 0.], [0., 4.], [0., 0.], [0., 4.]]], device=device))
    torch.testing.assert_close(values.grad, torch.tensor([[[[1., 2., 1.]]]], device=device))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_nearest_just_below_half_does_not_round_up(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    values = torch.tensor([[[[2., 5.]]]], device=device)
    half = np.float32(.5)
    points = torch.tensor([[[0., np.nextafter(half, np.float32(0))], [0., half]]], device=device)
    torch.testing.assert_close(torch_sample(values, points, order=0), torch.tensor([[[2., 5.]]], device=device))
