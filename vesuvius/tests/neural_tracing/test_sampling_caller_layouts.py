"""Caller layouts and application masks around vc_reader's default sampling."""
import numpy as np
import torch

from vc_reader import tifxyz
from vesuvius.neural_tracing.fiber_trace_2d.augmentation import sample_xy_maps_bilinear
from vesuvius.neural_tracing.fiber_trace_2d.loader import _sample_grid_points_hwc, _resample_coord_tensor_batch_like_augmentation
from vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps import (
    _sample_local_surface_coords_from_uv_tensor,
    _sample_displacement_at_local_coords_tensor,
    _surface_to_stored_uv_samples_lattice,
)
from vesuvius.neural_tracing.winding_models.merge_winding_field import _patch_samples
from vesuvius.neural_tracing.winding_models.phase_overlap_sync import _sample_cached_phase


def test_half_ties_singleton_axis_and_coordinate_channel_order():
    values = torch.tensor([[[10., 20., 30.], [12., 24., 36.]]])
    valid = torch.tensor([[False, True]])
    sampled, supported = _sample_grid_points_hwc(values, valid, torch.tensor([[.5, 0.], [-1., 0.]]))
    torch.testing.assert_close(sampled, torch.tensor([[11., 22., 33.], [0., 0., 0.]]))
    assert supported.tolist() == [True, False]
    points = torch.tensor([[[[.5, 0.], [1., 0.]]]])
    sampled, supported = _resample_coord_tensor_batch_like_augmentation(values[None], valid[None], points)
    torch.testing.assert_close(sampled[0, 0], torch.tensor([[11., 22., 33.], [12., 24., 36.]]))
    assert supported.all()


def test_map_sampling_gradients_and_invalid_points():
    maps = torch.tensor([[[[0., 0.], [2., 0.]], [[0., 4.], [2., 4.]]]], requires_grad=True)
    points = torch.tensor([[[.25, .75], [2., 1.]]], requires_grad=True)
    values, valid = sample_xy_maps_bilinear(maps, points)
    torch.testing.assert_close(values[0, 0], torch.tensor([.5, 3.]))
    assert valid.tolist() == [[True, False]]
    assert values[0, 1].isnan().all()
    values[0, 0].sum().backward()
    torch.testing.assert_close(points.grad[0, 0], torch.tensor([2., 4.]))


def test_surface_coordinates_use_shared_quad_validity():
    uv = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    xyz = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]], np.float32)
    queries = torch.tensor([[.5, .5], [3., 3.]])
    values, valid = _sample_local_surface_coords_from_uv_tensor(uv, xyz, queries)
    torch.testing.assert_close(values[0], torch.tensor([4., 5., 6.]))
    assert valid.tolist() == [True, False]
    _, valid = _sample_local_surface_coords_from_uv_tensor(uv[:3], xyz[:3], queries)
    assert not valid.any()


def test_displacement_sampling_keeps_zyx_query_order():
    z, y, x = torch.meshgrid(*(torch.arange(4.) for _ in range(3)), indexing="ij")
    field = torch.stack((z, 2 * y, 3 * x))[None]
    sampled, valid = _sample_displacement_at_local_coords_tensor(field, torch.tensor([[.5, 1.5, 2.5], [4., 1., 1.]]))
    torch.testing.assert_close(sampled, torch.tensor([[.5, 3., 7.5], [0., 0., 0.]]))
    assert valid.tolist() == [True, False]


def test_surface_lattice_points_follow_stored_uv():
    row, col = np.mgrid[:5, :5].astype(np.float32)
    xyz = np.stack((col, row, col * col), -1)
    uv, points, _ = _surface_to_stored_uv_samples_lattice(xyz, np.ones(row.shape, bool), (0, 0), 2, 2)
    np.testing.assert_array_equal(points, xyz[uv[:, 0] * 2, uv[:, 1] * 2])


def test_winding_phase_uses_boundary_voxel_and_shared_float32_sampling():
    class Frame:
        spacing = 1
        def to_slab(self, points):
            return points
    z, y, x = np.mgrid[:4, :4, :4]
    phase = (z + 2 * y + 3 * x).astype(np.float32)
    value, density, _ = _sample_cached_phase(phase, np.ones_like(phase, bool), Frame(), [1.5, 1.5, 1.5], 1)
    assert value == 9 and density == 3
    value, _, _ = _sample_cached_phase(phase, np.ones_like(phase, bool), Frame(), [3, 2, 2], 1)
    assert value == 13


def test_winding_patch_sampling_uses_default_bicubic(tmp_path):
    row, col = np.mgrid[:5, :5].astype(np.float32)
    xyz = np.stack((col + 10, row + 10, col * col + 10), -1)
    tifxyz.write(tmp_path / "sheet", tifxyz.Surface(xyz))
    samples, vertices = _patch_samples(tmp_path / "sheet", erode=0, subdiv=2)
    # Query row=1.25, column=1.25 is the first sample in quad (1,1).
    np.testing.assert_allclose(samples[(1 * 4 + 1) * 4], [11.25, 11.25, 11.5625], atol=1e-5)
    np.testing.assert_array_equal(vertices, xyz.reshape(-1, 3))
