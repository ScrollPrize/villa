import numpy as np
import torch
from vc_reader import tifxyz
from vesuvius.neural_tracing.heatmap_single_point.dataset import Patch, HeatmapDatasetV2, get_zyx_from_patch
from vesuvius.neural_tracing.winding_models.winding_model_dataset import _sample_quad_fields


def test_heatmap_queries_and_quad_normals_share_bicubic_geometry():
    r, c = np.mgrid[:7, :8].astype(np.float32)
    xyz = np.stack((10+.15*r*r, r+20, c+30), -1)
    patch = Patch(torch.from_numpy(xyz), torch.tensor([.5, .5]), None)
    rc = torch.tensor([[2.25, 3.75], [3.5, 4.5]])
    expected, _ = tifxyz.Surface(xyz).points_at(rc[:, :1].numpy(), rc[:, 1:].numpy())
    np.testing.assert_allclose(get_zyx_from_patch(rc, patch), expected[:, 0], atol=3e-5)
    dataset = object.__new__(HeatmapDatasetV2)
    mask = torch.zeros_like(patch.valid_quad_mask); mask[2, 3] = True
    points, normals = dataset._sample_points_from_quads(patch, mask, normals=True)
    assert points.shape == normals.shape == (4, 3)
    torch.testing.assert_close(normals.norm(dim=-1), torch.ones(4))
    # Normal orientation follows column tangent cross row tangent in ZYX.
    assert (normals[:, 0] < 0).all()


def test_winding_cells_keep_linear_bounds_and_channel_order():
    corners = [np.array([[1, 2, 3], [4, 5, 6]], np.float64)+offset for offset in (0, 2, 4, 6)]
    result = _sample_quad_fields(*corners, np.array([0., .25, 1.]), np.array([0., .75, 1.]))
    expected = corners[0][:, None] + np.array([0, 3.5, 6])[None, :, None]
    np.testing.assert_allclose(result, expected, atol=2e-6)
