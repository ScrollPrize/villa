"""
Alternate implementations of HeatmapDatasetV2._get_perturbed_zyx_from_patch for benchmarking.

The profiler shows this helper dominates dataloader time:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       399    0.147    0.000   21.268    0.053 dataset.py:301(_get_perturbed_zyx_from_patch)
       399    0.049    0.000   19.402    0.049 dataset.py:255(_get_cached_patch_points)

These subclasses aim to reduce the repeated patch-point work without changing any outputs.
"""

import torch

from vesuvius.neural_tracing.dataset import HeatmapDatasetV2, get_zyx_from_patch

# default timings mean 47.14 ms | median 48.07 ms | min 25.49 ms | max 62.91 ms


# mean 18.12 ms | median 5.93 ms | min 0.24 ms | max 59.35 ms
class CachedPatchPoints(HeatmapDatasetV2):
    """
    Cache _get_cached_patch_points across multiple perturbations that share the same
    (patch, center_ij, crop) context. The normal calculation and random draws are unchanged,
    so outputs match the base implementation.
    """

    def __init__(self, config, patches_for_split):
        super().__init__(config, patches_for_split)
        self._perturb_cache_key = None
        self._perturb_cache_value = None

    def _make_cache_key(self, patch, center_ij, min_corner_zyx, crop_size):
        return (
            id(patch),
            tuple(center_ij.tolist()),
            tuple(min_corner_zyx.tolist()),
            int(crop_size),
        )

    def _get_cached_patch_points_cached(self, patch, center_ij, min_corner_zyx, crop_size):
        key = self._make_cache_key(patch, center_ij, min_corner_zyx, crop_size)
        if key != self._perturb_cache_key:
            self._perturb_cache_key = key
            self._perturb_cache_value = super()._get_cached_patch_points(patch, center_ij, min_corner_zyx, crop_size)
        return self._perturb_cache_value

    def _get_perturbed_zyx_from_patch(self, point_ij, patch, center_ij, min_corner_zyx, crop_size, is_center_point=False):
        if is_center_point:
            perturbed_ij = point_ij
            perturbed_zyx = get_zyx_from_patch(point_ij, patch)
        else:
            offset_magnitude = torch.rand([]) * self._uv_max_perturbation
            offset_angle = torch.rand([]) * 2 * torch.pi
            offset_uv_voxels = offset_magnitude * torch.tensor([torch.cos(offset_angle), torch.sin(offset_angle)])

            offset_2d = offset_uv_voxels * patch.scale
            perturbed_ij = point_ij + offset_2d

            perturbed_ij = torch.clamp(perturbed_ij, torch.zeros([]), torch.tensor(patch.zyxs.shape[:2]) - 1.001)

            if not patch.valid_quad_mask[*perturbed_ij.int()]:
                return get_zyx_from_patch(point_ij, patch)

            perturbed_zyx = get_zyx_from_patch(perturbed_ij, patch)

        i, j = perturbed_ij.int()
        h, w = patch.zyxs.shape[:2]

        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and patch.valid_vertex_mask[ni, nj]:
                    neighbors.append(patch.zyxs[ni, nj])

        if len(neighbors) < 3:
            final_zyx = perturbed_zyx
        else:
            normal = torch.linalg.cross(neighbors[1] - neighbors[0], neighbors[2] - neighbors[0], dim=-1)
            normal_norm = torch.norm(normal)
            if normal_norm > 1e-6:
                normal = normal / normal_norm
                normal_offset_magnitude = (torch.rand([]) * 2 - 1) * self._w_max_perturbation

                cached_patch_points = self._get_cached_patch_points_cached(patch, center_ij, min_corner_zyx, crop_size)

                while abs(normal_offset_magnitude) >= 1.0:
                    nearest_patch_distance = self._get_distance_to_nearest_patch_cached(perturbed_zyx, cached_patch_points)
                    if abs(normal_offset_magnitude) <= nearest_patch_distance * self._main_component_distance_factor:
                        break
                    normal_offset_magnitude *= 0.8
                else:
                    normal_offset_magnitude = 0.

                final_zyx = perturbed_zyx + normal_offset_magnitude * normal
            else:
                final_zyx = perturbed_zyx

        return final_zyx

# mean 16.02 ms | median 2.11 ms | min 0.24 ms | max 58.73 ms
class CachedPatchPointsVectorizedDistance(CachedPatchPoints):
    """
    Same cache as CachedPatchPoints, but compute nearest-patch distance in one
    vectorised pass instead of a Python loop over patches.
    """

    def _get_distance_to_nearest_patch_cached(self, point_zyx, cached_patch_points):
        if not cached_patch_points:
            return float('inf')
        all_points = torch.cat(cached_patch_points, dim=0)
        if all_points.numel() == 0:
            return float('inf')
        return torch.norm(all_points - point_zyx, dim=-1).min().item()

