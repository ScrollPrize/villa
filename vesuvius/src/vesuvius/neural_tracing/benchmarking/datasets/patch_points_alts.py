"""
Alternate implementations of HeatmapDatasetV2._get_patch_points_in_crop for benchmarking.

Profile reference:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
     14466    0.013    0.000   12.498    0.001 dataset.py:225(_get_patch_points_in_crop)
     15264    6.867    0.000   11.626    0.001 dataset.py:221(_get_quads_in_crop)

Goals: reduce redundant per-call work (bounding-box rejection, precomputed sampling
grid) while keeping outputs identical to the base implementation.
"""

from typing import Dict, Tuple

import torch

from vesuvius.neural_tracing.dataset import HeatmapDatasetV2


class BBoxFilteredPatchPoints(HeatmapDatasetV2):
    """
    Early-exit when the quad bounding box does not intersect the crop. The fallback
    path is the exact base implementation.
    """

    def __init__(self, config, patches_for_split):
        super().__init__(config, patches_for_split)
        quad_bboxes: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for patch in self._patches:
            valid_centers = patch.quad_centers[patch.valid_quad_mask]
            if len(valid_centers) == 0:
                bbox_min = torch.zeros(3, dtype=patch.quad_centers.dtype)
                bbox_max = bbox_min
            else:
                bbox_min = valid_centers.min(dim=0).values
                bbox_max = valid_centers.max(dim=0).values
            quad_bboxes[id(patch)] = (bbox_min, bbox_max)
        self._quad_bboxes = quad_bboxes

    def _get_patch_points_in_crop(self, patch, min_corner_zyx, crop_size):
        bbox_min, bbox_max = self._quad_bboxes[id(patch)]
        crop_min = min_corner_zyx.to(dtype=bbox_min.dtype)
        crop_size_tensor = torch.as_tensor(crop_size, dtype=bbox_min.dtype)
        crop_max = crop_min + crop_size_tensor
        if (bbox_max < crop_min).any() or (bbox_min >= crop_max).any():
            return torch.empty((0, 3), dtype=torch.float32)
        return super()._get_patch_points_in_crop(patch, min_corner_zyx, crop_size)


class PrecomputedSamplingPatchPoints(HeatmapDatasetV2):
    """
    Precompute per-patch sampling tensors (quad corners, u/v grids) so each call
    only masks and lerps instead of rebuilding stacks.
    """

    def __init__(self, config, patches_for_split):
        super().__init__(config, patches_for_split)
        sampling: Dict[int, Dict[str, torch.Tensor]] = {}
        for patch in self._patches:
            top_left = patch.zyxs[:-1, :-1]
            top_right = patch.zyxs[:-1, 1:]
            bottom_left = patch.zyxs[1:, :-1]
            bottom_right = patch.zyxs[1:, 1:]
            quad_corners = torch.stack([
                torch.stack([top_left, top_right], dim=2),
                torch.stack([bottom_left, bottom_right], dim=2),
            ], dim=2)  # shape: (h-1, w-1, 2, 2, 3)

            points_per_side = (1 / patch.scale + 0.5).int()
            v_points = torch.arange(points_per_side[0], dtype=torch.float32) / points_per_side[0]
            u_points = torch.arange(points_per_side[1], dtype=torch.float32) / points_per_side[1]

            sampling[id(patch)] = {
                "quad_corners": quad_corners,
                "v_points": v_points,
                "u_points": u_points,
            }
        self._sampling = sampling

    def _get_patch_points_in_crop(self, patch, min_corner_zyx, crop_size):
        quad_in_crop = self._get_quads_in_crop(patch, min_corner_zyx, crop_size)
        if not torch.any(quad_in_crop):
            return torch.empty((0, 3), dtype=torch.float32)

        info = self._sampling[id(patch)]
        filtered_quads_zyxs = info["quad_corners"][quad_in_crop]

        points_covering_quads = torch.lerp(
            filtered_quads_zyxs[:, None, 0, :],
            filtered_quads_zyxs[:, None, 1, :],
            info["v_points"][None, :, None, None],
        )
        points_covering_quads = torch.lerp(
            points_covering_quads[:, :, None, 0],
            points_covering_quads[:, :, None, 1],
            info["u_points"][None, None, :, None],
        )

        return points_covering_quads.view(-1, 3)


class BBoxPrecomputedPatchPoints(PrecomputedSamplingPatchPoints):
    """
    Combine AABB rejection with precomputed sampling grids.
    """

    def __init__(self, config, patches_for_split):
        super().__init__(config, patches_for_split)
        quad_bboxes: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for patch in self._patches:
            valid_centers = patch.quad_centers[patch.valid_quad_mask]
            if len(valid_centers) == 0:
                bbox_min = torch.zeros(3, dtype=patch.quad_centers.dtype)
                bbox_max = bbox_min
            else:
                bbox_min = valid_centers.min(dim=0).values
                bbox_max = valid_centers.max(dim=0).values
            quad_bboxes[id(patch)] = (bbox_min, bbox_max)
        self._quad_bboxes = quad_bboxes

    def _get_patch_points_in_crop(self, patch, min_corner_zyx, crop_size):
        bbox_min, bbox_max = self._quad_bboxes[id(patch)]
        crop_min = min_corner_zyx.to(dtype=bbox_min.dtype)
        crop_size_tensor = torch.as_tensor(crop_size, dtype=bbox_min.dtype)
        crop_max = crop_min + crop_size_tensor
        if (bbox_max < crop_min).any() or (bbox_min >= crop_max).any():
            return torch.empty((0, 3), dtype=torch.float32)
        return super()._get_patch_points_in_crop(patch, min_corner_zyx, crop_size)
